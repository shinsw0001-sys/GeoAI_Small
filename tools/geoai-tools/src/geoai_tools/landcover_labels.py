"""토지피복 라벨 생성 (job:LandCoverLabelJob, 진입점 landcover-labels).

입력: 분석 준비 데이터(gai:AnalysisReadyData)
      + 매개변수 label_source: 토지피복 다각형 GeoJSON (예: 환경부 토지피복지도, 속성 L1_CODE·L1_NAME)
처리: 1) 좌표 참조 체계 판별·변환, 기하 오류 수정, 빈 기하 제거(벡터 라벨 래스터화와 같은 규칙)
      2) 항목 대응: 분류 코드 속성(code_attribute, 없으면 L1_CODE → L2_CODE → L3_CODE → code 순)을
         개념의 skos:notation 과 맞춘다. 중·세분류 코드(110, 111 …)는 첫 자리로 대분류(100)에 대응시킨다.
         코드가 없으면 이름 속성(class → L1_NAME)을 개념 이름·지역명과 맞춘다(가운뎃점·공백 무시).
      3) 최소 면적(min_area_m2) 미만 다각형 제거, 겹친 다각형은 작은 것이 위에 오게 래스터화
      4) 타일마다 분류 마스크로 래스터화. 다각형이 없는 화소는 미분류(0)로 두고 학습·평가에서 뺀다(ignore_value 0).
품질 검사: 라벨 피복 비율(미분류가 아닌 유효 화소 비율)이 min_coverage 미만이면 실패(landcover_coverage_low).
           대응되지 않는 항목 값, 다각형 겹침 면적, 시험 범위에 없는 항목을 품질 정보로 남긴다.
산출: 라벨 세트(gai:LabelSet) = manifest.json + masks.npz
매개변수: label_source(필수), code_attribute(선택), min_area_m2(0), min_coverage(0.5), label_crs(선택)
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from geoai_common.objectstore import get_json
from rasterio.crs import CRS
from shapely.ops import unary_union

from .data import load_npz
from .labels import find_ard, polygons, rasterize_tiles, read_geojson, save_labelset, source_crs
from .runtime import ToolContext, ToolError, ToolOutput

CODE_ATTRIBUTES = ("L1_CODE", "L2_CODE", "L3_CODE", "code")
NAME_ATTRIBUTES = ("class", "L1_NAME")
UNLABELLED = {"value": 0, "label": "미분류", "iri": None}


def _norm_name(v: object) -> str:
    return re.sub(r"[\s·ㆍ,\-_/]", "", str(v)).lower()


def major_code(raw: object) -> str | None:
    """분류 코드 → 대분류 코드. '110'·'111'·110.0 → '100'. 숫자가 아니면 None."""
    text = str(raw).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit() or len(text) != 3 or text[0] == "0":
        return None
    return text[0] + "00"


def class_lookup(classes: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    """(대분류 코드 → 화소값, 정규화 이름 → 화소값)."""
    by_code = {str(c["notation"]): int(c["value"]) for c in classes if c.get("notation")}
    by_name: dict[str, int] = {}
    for c in classes:
        for key in (c["label"], c["iri"].rsplit("#", 1)[-1], c["iri"]):
            by_name[_norm_name(key)] = int(c["value"])
    return by_code, by_name


def class_of(props: dict, by_code: dict, by_name: dict, code_attr: str | None) -> tuple[int | None, str]:
    """(화소값 | None, 대응 근거 'code'·'name'·원래 값)."""
    for attr in (code_attr,) if code_attr else CODE_ATTRIBUTES:
        if props.get(attr) is not None:
            code = major_code(props[attr])
            if code in by_code:
                return by_code[code], "code"
            return None, f"{attr}={props[attr]}"
    for attr in NAME_ATTRIBUTES:
        if props.get(attr) is not None:
            v = by_name.get(_norm_name(props[attr]))
            return (v, "name") if v is not None else (None, f"{attr}={props[attr]}")
    return None, "속성 없음"


def run(ctx: ToolContext) -> ToolOutput:
    ard = find_ard(ctx)
    source = ctx.spec.parameters.get("label_source")
    if not source:
        raise ToolError("missing_parameter", "label_source(토지피복 다각형 GeoJSON 주소)가 필요합니다.")
    classes = ctx.spec.label_classes
    if not classes:
        raise ToolError("no_label_classes", "분야 라벨 체계에 토지피복 개념이 없습니다.")
    code_attr = ctx.spec.parameters.get("code_attribute")
    min_area = ctx.param("min_area_m2", 0.0)
    min_cov = ctx.param("min_coverage", 0.5)

    ctx.progress(5, "분석 준비 데이터·토지피복 다각형 읽기")
    manifest = get_json(ctx.store, ard.storage_uri)
    valid = load_npz(ctx.store, manifest["tiles_uri"])["valid"]
    geojson = read_geojson(ctx, source)
    src_crs = source_crs(geojson, ctx.spec.parameters.get("label_crs"))
    dst_crs = CRS.from_user_input(manifest["crs"])
    gsd = float(manifest["ground_sample_distance"])

    ctx.progress(15, "좌표 변환·기하 오류 수정·항목 대응")
    stats = {"features": 0, "fixed_invalid": 0, "dropped_empty": 0, "unmatched_class": 0, "dropped_small": 0}
    matched_by = Counter()
    unmatched = Counter()
    shapes: list[tuple] = []
    by_code, by_name = class_lookup(classes)
    for g, props in polygons(geojson, src_crs, dst_crs, stats):
        value, how = class_of(props, by_code, by_name, code_attr)
        if value is None:
            stats["unmatched_class"] += 1
            unmatched[how] += 1
            continue
        if g.area < min_area:
            stats["dropped_small"] += 1
            continue
        matched_by[how] += 1
        shapes.append((g, value))
    if not shapes:
        hint = ", ".join(f"{k} ({n}개)" for k, n in unmatched.most_common(3)) or "다각형 없음"
        raise ToolError(
            "no_landcover_polygon",
            f"토지피복 대분류에 대응되는 다각형이 없습니다(지물 {stats['features']}개). 대응 안 된 값: {hint}",
        )
    # 겹침: 다각형 면적 합 − 합집합 면적. 래스터화는 큰 다각형부터 그려 작은 다각형이 위에 남게 한다.
    shapes.sort(key=lambda s: -s[0].area)
    overlap_m2 = max(0.0, sum(g.area for g, _ in shapes) - unary_union([g for g, _ in shapes]).area)

    ctx.progress(30, f"타일 래스터화 ({len(manifest['tiles'])}장)")
    masks = rasterize_tiles(ctx, shapes, manifest, valid, start=30)
    n_valid = int(valid.sum())
    coverage = float(((masks > 0) & valid).sum() / max(n_valid, 1))
    if not (masks > 0).any():
        raise ToolError(
            "label_no_overlap",
            f"토지피복 다각형({len(shapes)}개, {src_crs.to_string()})이 영상 범위와 겹치지 않습니다.",
        )
    if coverage < min_cov:
        raise ToolError(
            "landcover_coverage_low",
            f"토지피복 라벨이 유효 화소의 {coverage:.1%}만 덮습니다(기준 {min_cov:.0%}). "
            "범위·좌표 참조 체계를 확인하세요.",
        )

    counts = np.bincount(masks[valid].ravel(), minlength=len(classes) + 1)
    labelled = max(int(counts[1:].sum()), 1)
    polys = Counter(v for _, v in shapes)
    class_stats = [
        {
            "value": int(c["value"]),
            "label": c["label"],
            "notation": c.get("notation"),
            "polygons": int(polys.get(int(c["value"]), 0)),
            "area_m2": round(float(counts[int(c["value"])]) * gsd * gsd, 1),
            "share": round(float(counts[int(c["value"])]) / labelled, 4),
        }
        for c in classes
    ]
    quality = {
        **stats,
        "matched_by_code": int(matched_by["code"]),
        "matched_by_name": int(matched_by["name"]),
        "unmatched_values": [f"{k} ({n})" for k, n in unmatched.most_common(5)],
        "overlap_m2": round(overlap_m2, 1),
        "coverage": round(coverage, 4),
        "unlabelled_ratio": round(1 - coverage, 4),
        "missing_classes": [r["label"] for r in class_stats if r["area_m2"] == 0],
        "classes": class_stats,
    }
    extra = {
        "label_source": source,
        "source_crs": src_crs.to_string(),
        "label_geometry": "Polygon",
        "coverage": "exhaustive",
        "classes": [UNLABELLED, *classes],
        "ignore_value": 0,
    }
    asset, q = save_labelset(ctx, ard, masks, extra, quality)
    return ToolOutput(
        outputs=[asset],
        metrics={
            "label_features": float(len(shapes)),
            "coverage": round(coverage, 4),
            "fixed_invalid": float(stats["fixed_invalid"]),
            "unmatched_class": float(stats["unmatched_class"]),
            "classes_present": float(sum(r["area_m2"] > 0 for r in class_stats)),
            **q,
        },
    )
