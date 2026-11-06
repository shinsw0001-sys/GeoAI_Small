"""변화 라벨 도출 (job:ChangeLabelDeriveJob, 진입점 change-labels).

입력: 분석 준비 데이터(gai:AnalysisReadyData, 다시기)
매개변수(둘 중 하나)
  label_sources: 시점별 벡터 목록 [{acquired_at, buildings: 건물 윤곽 GeoJSON, landcover: 토지피복 GeoJSON(선택)}]
                 첫 시점과 마지막 시점을 대조한다.
  label_source : 변화 벡터 GeoJSON. 지물 속성 class 가 라벨 개념(지역명 또는 한글 이름)이다. 그대로 래스터화한다.
도출 규칙 (분야 온톨로지 dom:Labels 의 개념 정의를 따른다)
  1) 두 시점 건물을 교집합 대 합집합 비율(match_iou, 0.1) 이상인 쌍끼리 큰 값부터 1:1 대응시킨다.
  2) 이후 시점에만 있는 건물 → 건물 신축(이후 윤곽). 이전 시점에만 있는 건물 → 건물 멸실(이전 윤곽).
  3) 대응 쌍의 비율이 modified_iou(0.8) 미만 → 건물 변경. 두 윤곽의 대칭 차 중 min_change_m2(4) 이상 부분.
  4) 두 시점 토지피복 다각형이 겹치고 분류(class)가 다르면 → 토지 변화(겹친 부분). 건물 변화 구역은 뺀다.
  래스터화 순서: 토지 변화 → 건물 멸실 → 건물 변경 → 건물 신축 (뒤가 앞을 덮는다).
산출: 라벨 세트(gai:LabelSet) = manifest.json + masks.npz + change-reference.geojson(도출한 변화 다각형)
"""

from __future__ import annotations

import json

from geoai_common.objectstore import get_json
from rasterio.crs import CRS
from shapely import STRtree
from shapely.geometry import mapping
from shapely.ops import unary_union

from .data import load_npz
from .labels import class_value, find_ard, polygons, rasterize_tiles, read_geojson, save_labelset, source_crs
from .runtime import ToolContext, ToolError, ToolOutput

CONCEPTS = ("LandChange", "DemolishedBuilding", "ModifiedBuilding", "NewBuilding")  # 래스터화 순서


def _parts(geom, min_area: float) -> list:
    if geom.is_empty:
        return []
    geoms = getattr(geom, "geoms", [geom])
    return [g for g in geoms if g.geom_type in ("Polygon", "MultiPolygon") and g.area >= min_area]


def match_buildings(before: list, after: list, match_iou: float) -> list[tuple[int, int, float]]:
    """IoU 내림차순 탐욕 1:1 대응. (이전 번호, 이후 번호, IoU)"""
    tree = STRtree(after)
    pairs = []
    for i, p in enumerate(before):
        for j in tree.query(p):
            q = after[int(j)]
            inter = p.intersection(q).area
            if inter <= 0:
                continue
            iou = inter / (p.area + q.area - inter)
            if iou >= match_iou:
                pairs.append((iou, i, int(j)))
    used_b, used_a, out = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        if i not in used_b and j not in used_a:
            used_b.add(i)
            used_a.add(j)
            out.append((i, j, iou))
    return out


def derive_changes(
    before: list,
    after: list,
    land_before: list[tuple],
    land_after: list[tuple],
    *,
    match_iou: float = 0.1,
    modified_iou: float = 0.8,
    min_area: float = 4.0,
) -> dict[str, list]:
    """변화 유형별 다각형 목록. before/after: 건물 다각형, land_*: (다각형, 분류 이름)."""
    pairs = match_buildings(before, after, match_iou)
    matched_b = {i for i, _, _ in pairs}
    matched_a = {j for _, j, _ in pairs}
    out: dict[str, list] = {c: [] for c in CONCEPTS}
    out["NewBuilding"] = [g for j, g in enumerate(after) if j not in matched_a and g.area >= min_area]
    out["DemolishedBuilding"] = [g for i, g in enumerate(before) if i not in matched_b and g.area >= min_area]
    for i, j, iou in pairs:
        if iou < modified_iou:
            out["ModifiedBuilding"].extend(_parts(before[i].symmetric_difference(after[j]), min_area))

    if land_before and land_after:
        tree = STRtree([g for g, _ in land_after])
        changed = []
        for g1, c1 in land_before:
            for k in tree.query(g1):
                g2, c2 = land_after[int(k)]
                if c1 != c2:
                    changed.extend(_parts(g1.intersection(g2), min_area))
        building_change = [g for c in ("NewBuilding", "DemolishedBuilding", "ModifiedBuilding") for g in out[c]]
        if changed:
            land = unary_union(changed)
            if building_change:
                land = land.difference(unary_union(building_change))
            out["LandChange"] = _parts(land, min_area)
    return out


def run(ctx: ToolContext) -> ToolOutput:
    ard = find_ard(ctx)
    manifest = get_json(ctx.store, ard.storage_uri)
    valid = load_npz(ctx.store, manifest["tiles_uri"])["valid"]
    dst_crs = CRS.from_user_input(manifest["crs"])
    classes = ctx.spec.label_classes
    by_local = {c["iri"].rsplit("#", 1)[-1]: int(c["value"]) for c in classes}
    fallback_crs = ctx.spec.parameters.get("label_crs")
    stats: dict = {"features": 0, "fixed_invalid": 0, "dropped_empty": 0}
    min_area = ctx.param("min_change_m2", 4.0)

    direct = ctx.spec.parameters.get("label_source")
    sources = ctx.spec.parameters.get("label_sources") or []
    ctx.progress(5, "분석 준비 데이터·시점별 벡터 읽기")
    if sources:
        missing = [c for c in CONCEPTS if c not in by_local]
        if missing:
            raise ToolError("label_scheme_mismatch", f"분야 라벨 체계에 변화 개념이 없습니다: {', '.join(missing)}")
        dated = sorted((s for s in sources if s.get("buildings")), key=lambda s: str(s.get("acquired_at", "")))
        if len(dated) < 2:
            raise ToolError(
                "insufficient_acquisitions", "label_sources 에 건물 윤곽이 있는 시점이 2개 이상 필요합니다."
            )
        first, last = dated[0], dated[-1]

        def load(uri: str | None) -> list[tuple]:
            if not uri:
                return []
            gj = read_geojson(ctx, uri)
            return polygons(gj, source_crs(gj, fallback_crs), dst_crs, stats)

        ctx.progress(15, f"건물 윤곽 대조 ({first.get('acquired_at')} ↔ {last.get('acquired_at')})")
        b1, b2 = [g for g, _ in load(first["buildings"])], [g for g, _ in load(last["buildings"])]
        attr = ctx.spec.parameters.get("landcover_attribute", "class")
        l1 = [(g, str(p.get(attr))) for g, p in load(first.get("landcover"))]
        l2 = [(g, str(p.get(attr))) for g, p in load(last.get("landcover"))]
        ctx.progress(25, "변화 유형 도출")
        changes = derive_changes(
            b1, b2, l1, l2,
            match_iou=ctx.param("match_iou", 0.1), modified_iou=ctx.param("modified_iou", 0.8), min_area=min_area,
        )  # fmt: skip
        extra = {
            "label_sources": sources,
            "compared": [first.get("acquired_at"), last.get("acquired_at")],
            "derivation": "building-match+landcover-overlay",
        }
        stats.update(
            {
                "buildings_before": len(b1),
                "buildings_after": len(b2),
                "landcover_before": len(l1),
                "landcover_after": len(l2),
            }
        )
    elif direct:
        gj = read_geojson(ctx, direct)
        changes = {c: [] for c in CONCEPTS}
        unmatched = 0
        local_of = {v: k for k, v in by_local.items()}
        for g, props in polygons(gj, source_crs(gj, fallback_crs), dst_crs, stats):
            v = class_value(props, classes) if props.get("class") is not None else None
            if v is None:
                unmatched += 1
                continue
            changes.setdefault(local_of[v], []).append(g)
        stats["unmatched_class"] = unmatched
        extra = {"label_source": direct, "derivation": "direct"}
    else:
        raise ToolError("missing_parameter", "label_sources(시점별 벡터) 또는 label_source(변화 벡터)가 필요합니다.")

    counts = {c: len(changes.get(c, [])) for c in CONCEPTS}
    stats["change_polygons"] = counts
    shapes = [(g, by_local[c]) for c in CONCEPTS if c in by_local for g in changes.get(c, [])]
    if not shapes:
        raise ToolError("no_change_found", "두 시점 사이에서 도출된 변화가 없습니다. 시점별 벡터를 확인하세요.")

    ctx.progress(40, f"타일 래스터화 ({len(manifest['tiles'])}장)")
    masks = rasterize_tiles(ctx, shapes, manifest, valid)
    if not (masks > 0).any():
        raise ToolError("label_no_overlap", "도출한 변화 다각형이 영상 범위와 겹치지 않습니다.")

    label_of = {c["iri"].rsplit("#", 1)[-1]: c["label"] for c in classes}
    epsg = dst_crs.to_epsg()
    reference = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"concept": c, "class": label_of.get(c, c), "area_m2": round(g.area, 1)},
                "geometry": mapping(g),
            }
            for c in CONCEPTS
            for g in changes.get(c, [])
        ],
    }
    if epsg:
        reference["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
    ref_uri = ctx.out("labels", "change-reference.geojson")
    ctx.store.put_bytes(ref_uri, json.dumps(reference, ensure_ascii=False).encode())
    extra["change_reference_uri"] = ref_uri
    asset, q = save_labelset(ctx, ard, masks, extra, stats)
    return ToolOutput(
        outputs=[asset],
        metrics={
            **{f"{c}_polygons": float(n) for c, n in counts.items()},
            "fixed_invalid": float(stats["fixed_invalid"]),
            **q,
        },
    )
