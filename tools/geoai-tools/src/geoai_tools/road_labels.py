"""도로 중심선 라벨 생성 (job:RoadCenterlineLabelJob, 진입점 road-labels).

입력: 분석 준비 데이터(gai:AnalysisReadyData)
      + 매개변수 label_source: 도로 중심선(LineString·MultiLineString) GeoJSON. 노면 다각형(Polygon)도 받는다.
선형 객체 검사·정리
  1) 좌표 참조 체계 판별·변환(벡터 라벨 래스터화와 같은 규칙), 빈 선·길이 0 선 제거, 중복 꼭짓점 제거
  2) 끝점 맞춤: 다른 선과 snap_m 이내인데 닿지 않은 끝점을 그 선 위 가장 가까운 점으로 옮긴다(덜 닿은 교차로)
  3) 교차점 분할(noding): 선이 만나는 곳에서 나눠 도로망 그래프의 노드로 만든다
  4) 폭: 속성 width_attribute(기본 width_m) 값, 없으면 default_width_m. 폭의 절반만큼 넓혀 노면을 만든다
래스터화: 노면을 타일마다 라벨 마스크(배경 0, 도로 1)로. 노면 다각형은 그대로 래스터화한다.
산출: 라벨 세트(gai:LabelSet) = manifest.json + masks.npz + reference_network.geojson(분할된 중심선, 평가용 참조 도로망)
매개변수: label_source(필수), width_attribute(width_m), default_width_m(6.0), snap_m(1.0), label_crs(선택)
"""

from __future__ import annotations

import json

from geoai_common.objectstore import get_json
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely import make_valid
from shapely.geometry import LineString, Point, mapping, shape
from shapely.ops import nearest_points, unary_union

from .data import load_npz
from .labels import class_value, find_ard, rasterize_tiles, read_geojson, save_labelset, source_crs
from .road_network import lines_graph, node_counts
from .runtime import ToolContext, ToolError, ToolOutput


def _width(props: dict, attr: str, default: float) -> tuple[float, bool]:
    try:
        w = float(props.get(attr))
        return (w, True) if w > 0 else (default, False)
    except (TypeError, ValueError):
        return default, False


def split_features(geojson: dict, src: CRS, dst: CRS, classes: list[dict], width_attr: str, default_w: float):
    """(중심선 [(선, 폭, 화소값)], 노면 [(면, 화소값)], 통계)."""
    lines, surfaces = [], []
    st = {"features": 0, "lines": 0, "surfaces": 0, "dropped_empty": 0, "dropped_other_type": 0,
          "unmatched_class": 0, "width_from_attribute": 0, "width_default": 0, "fixed_invalid": 0}  # fmt: skip
    for feat in geojson.get("features", []):
        st["features"] += 1
        geom = feat.get("geometry")
        props = feat.get("properties") or {}
        if not geom:
            st["dropped_empty"] += 1
            continue
        value = class_value(props, classes)
        if value is None:
            st["unmatched_class"] += 1
            continue
        if src != dst:
            geom = transform_geom(src, dst, geom)
        g = shape(geom)
        if g.geom_type in ("LineString", "MultiLineString"):
            w, from_attr = _width(props, width_attr, default_w)
            for part in getattr(g, "geoms", [g]):
                part = part.simplify(0)  # 중복 꼭짓점 제거
                if part.is_empty or part.length <= 0:
                    st["dropped_empty"] += 1
                    continue
                lines.append((part, w, value))
                st["lines"] += 1
                st["width_from_attribute" if from_attr else "width_default"] += 1
        elif g.geom_type in ("Polygon", "MultiPolygon"):
            if not g.is_valid:
                g = make_valid(g).buffer(0)
                st["fixed_invalid"] += 1
            if g.is_empty or g.area <= 0:
                st["dropped_empty"] += 1
                continue
            surfaces.append((g, value))
            st["surfaces"] += 1
        else:
            st["dropped_other_type"] += 1
    return lines, surfaces, st


def snap_endpoints(lines: list[tuple], snap_m: float) -> tuple[list[tuple], int]:
    """다른 선과 snap_m 이내인데 닿지 않은 끝점을 그 선 위로 옮긴다."""
    if snap_m <= 0 or len(lines) < 2:
        return lines, 0
    out, snapped = [], 0
    geoms = [ln for ln, _, _ in lines]
    for i, (ln, w, v) in enumerate(lines):
        coords = list(ln.coords)
        for end in (0, -1):
            p = Point(coords[end])
            others = [g for j, g in enumerate(geoms) if j != i and g.distance(p) <= snap_m]
            if not others or any(g.distance(p) < 1e-6 for g in others):
                continue
            target = min(others, key=lambda g: g.distance(p))
            q = nearest_points(target, p)[0]
            coords[end] = (q.x, q.y)
            snapped += 1
        out.append((LineString(coords), w, v))
    return out, snapped


def run(ctx: ToolContext) -> ToolOutput:
    ard = find_ard(ctx)
    source = ctx.spec.parameters.get("label_source")
    if not source:
        raise ToolError("missing_parameter", "label_source(도로 중심선 GeoJSON 주소)가 필요합니다.")
    width_attr = str(ctx.spec.parameters.get("width_attribute", "width_m"))
    default_w = ctx.param("default_width_m", 6.0)
    snap_m = ctx.param("snap_m", 1.0)

    ctx.progress(5, "분석 준비 데이터·도로 중심선 읽기")
    manifest = get_json(ctx.store, ard.storage_uri)
    valid = load_npz(ctx.store, manifest["tiles_uri"])["valid"]
    geojson = read_geojson(ctx, source)
    src_crs = source_crs(geojson, ctx.spec.parameters.get("label_crs"))
    dst_crs = CRS.from_user_input(manifest["crs"])
    lines, surfaces, stats = split_features(geojson, src_crs, dst_crs, ctx.spec.label_classes, width_attr, default_w)
    if not lines and not surfaces:
        raise ToolError(
            "no_road_geometry",
            f"도로 중심선·노면이 없습니다(지물 {stats['features']}개, 도로 아닌 분류 {stats['unmatched_class']}개, "
            f"선·면 아닌 기하 {stats['dropped_other_type']}개).",
        )

    ctx.progress(15, "끝점 맞춤·교차점 분할")
    lines, stats["snapped_endpoints"] = snap_endpoints(lines, snap_m)
    gsd = float(manifest["ground_sample_distance"])
    network = lines_graph([ln for ln, _, _ in lines], step=max(gsd, 1.0))
    topo = node_counts(network)
    noded = unary_union([ln for ln, _, _ in lines]) if lines else None
    segments = [s for s in getattr(noded, "geoms", [noded]) if s is not None and s.geom_type == "LineString"]

    ctx.progress(30, "노면 만들기(폭 확장)")
    shapes = [(ln.buffer(w / 2, cap_style="flat", join_style="round"), v) for ln, w, v in lines] + surfaces
    shapes = [(g, v) for g, v in shapes if not g.is_empty]

    ctx.progress(40, f"타일 래스터화 ({len(manifest['tiles'])}장)")
    masks = rasterize_tiles(ctx, shapes, manifest, valid)
    if not (masks > 0).any():
        raise ToolError(
            "label_no_overlap",
            f"도로 벡터({stats['features']}개 지물, {src_crs.to_string()})가 영상 범위와 겹치지 않습니다.",
        )

    ref_uri = None
    if segments:
        epsg = dst_crs.to_epsg()
        doc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"segment": i, "length_m": round(s.length, 2)},
                    "geometry": mapping(s),
                }
                for i, s in enumerate(segments)
            ],
        }
        if epsg:
            doc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
        ref_uri = ctx.out("labels", "reference_network.geojson")
        ctx.store.put_bytes(ref_uri, json.dumps(doc, ensure_ascii=False).encode())

    quality = {
        **stats,
        "segments": len(segments),
        "centerline_km": round(sum(s.length for s in segments) / 1000, 3),
        **topo,
    }
    asset, q = save_labelset(
        ctx,
        ard,
        masks,
        {
            "label_source": source,
            "source_crs": src_crs.to_string(),
            "label_geometry": "LineString",
            "reference_network_uri": ref_uri,
            "width": {"attribute": width_attr, "default_m": default_w},
        },
        quality,
    )
    return ToolOutput(
        outputs=[asset],
        metrics={
            "label_features": float(stats["features"]),
            "centerline_km": quality["centerline_km"],
            "junctions": float(topo["junctions"]),
            "dead_ends": float(topo["dead_ends"]),
            "snapped_endpoints": float(stats["snapped_endpoints"]),
            **q,
        },
    )
