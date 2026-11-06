"""벡터 라벨 래스터화 (job:VectorLabelRasterizeJob, 진입점 rasterize-labels).

입력: 분석 준비 데이터(gai:AnalysisReadyData)
      + 매개변수 label_source: 참조 벡터 GeoJSON 주소(s3://raw/...). 예: 건물 윤곽(수치지도·건물 통합정보)
처리: 1) 참조 벡터 좌표 참조 체계 판별(GeoJSON `crs` 멤버 → 매개변수 label_crs → EPSG:4326)
      2) 분석 준비 데이터 좌표 참조 체계로 변환, 기하 오류 수정(make_valid), 빈 기하 제거
      3) 지물 속성 `class` 를 분야 라벨 체계 개념(지역명 또는 한글 이름)과 맞춘다. 속성이 없으면 첫 개념.
      4) 타일마다 라벨 마스크로 래스터화(배경 0, 개념 화소값 1, 2, ...)
품질 검사: 라벨이 영상과 겹치지 않으면 실패(label_no_overlap)
산출: 라벨 세트(gai:LabelSet) = manifest.json + masks.npz
"""

from __future__ import annotations

import io
import json

import numpy as np
from geoai_common.messages import DataAssetRef
from geoai_common.objectstore import get_json, put_json
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely import make_valid
from shapely.geometry import box, mapping, shape

from .data import load_npz
from .runtime import ToolContext, ToolError, ToolOutput

GAI = "https://ontology.geoai.local/core#"


def source_crs(geojson: dict, fallback: str | None) -> CRS:
    name = (geojson.get("crs") or {}).get("properties", {}).get("name")
    return CRS.from_user_input(name or fallback or "EPSG:4326")


def class_value(props: dict, classes: list[dict]) -> int | None:
    if not classes:
        return 1
    wanted = props.get("class")
    if wanted is None:
        return int(classes[0]["value"])
    for c in classes:
        if str(wanted) in (c["iri"].rsplit("#", 1)[-1], c["label"], c["iri"]):
            return int(c["value"])
    return None


def read_geojson(ctx: ToolContext, uri: str) -> dict:
    try:
        return json.loads(ctx.store.get_bytes(uri))
    except (OSError, ValueError) as exc:
        raise ToolError("label_source_unreadable", f"참조 벡터를 읽지 못했습니다: {uri} ({exc})") from exc


def polygons(geojson: dict, src_crs: CRS, dst_crs: CRS, stats: dict) -> list[tuple]:
    """(면 기하, 속성) 목록. 좌표 변환, 기하 오류 수정, 빈 기하 제거를 한다. stats 에 건수를 더한다."""
    out = []
    for feat in geojson.get("features", []):
        stats["features"] = stats.get("features", 0) + 1
        geom = feat.get("geometry")
        if not geom:
            stats["dropped_empty"] = stats.get("dropped_empty", 0) + 1
            continue
        if src_crs != dst_crs:
            geom = transform_geom(src_crs, dst_crs, geom)
        g = shape(geom)
        if not g.is_valid:
            g = make_valid(g)
            stats["fixed_invalid"] = stats.get("fixed_invalid", 0) + 1
        g = g.buffer(0) if g.geom_type not in ("Polygon", "MultiPolygon") else g
        if g.is_empty or g.area <= 0:
            stats["dropped_empty"] = stats.get("dropped_empty", 0) + 1
            continue
        out.append((g, feat.get("properties") or {}))
    return out


def rasterize_tiles(
    ctx: ToolContext, shapes: list[tuple], manifest: dict, valid: np.ndarray, start: float = 40
) -> np.ndarray:
    """(기하, 화소값) 목록을 타일마다 마스크로. 뒤에 오는 기하가 앞을 덮는다."""
    tiles = manifest["tiles"]
    size = int(manifest["tile_size"])
    masks = np.zeros((len(tiles), size, size), dtype=np.uint8)
    for i, t in enumerate(tiles):
        transform = Affine(*t["transform"])
        x0, y0 = transform @ (0, 0)
        x1, y1 = transform @ (size, size)
        extent = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        hits = [(mapping(g), v) for g, v in shapes if g.intersects(extent)]
        if hits:
            masks[i] = rasterize(hits, out_shape=(size, size), transform=transform, dtype="uint8")
        masks[i][~valid[i]] = 0
        ctx.progress(start + (90 - start) * (i + 1) / len(tiles), f"타일 래스터화 {i + 1}/{len(tiles)}")
    return masks


def save_labelset(
    ctx: ToolContext, ard: DataAssetRef, masks: np.ndarray, extra: dict, quality: dict
) -> tuple[DataAssetRef, dict]:
    """마스크·라벨 세트 manifest 저장. (라벨 세트 자산, 품질 지표)"""
    n = len(masks)
    positive = float((masks > 0).mean())
    labelled_tiles = float((masks.reshape(n, -1).max(axis=1) > 0).mean())
    buf = io.BytesIO()
    np.savez_compressed(buf, masks=masks)
    masks_uri = ctx.out("labels", "masks.npz")
    ctx.store.put_bytes(masks_uri, buf.getvalue())
    pixel_counts = {int(v): int(c) for v, c in zip(*np.unique(masks, return_counts=True), strict=True)}
    labelset = {
        "kind": "raster-label-set",
        "ard_manifest": ard.storage_uri,
        "masks_uri": masks_uri,
        "classes": [{"value": 0, "label": "배경", "iri": None}, *ctx.spec.label_classes],
        **extra,
        "quality": {
            **quality,
            "positive_pixel_ratio": round(positive, 5),
            "labelled_tile_ratio": round(labelled_tiles, 4),
            "class_pixels": pixel_counts,
        },
    }
    uri = ctx.out("labels", "manifest.json")
    put_json(ctx.store, uri, labelset)
    asset = DataAssetRef(
        asset_id=f"{ctx.spec.task_id}.labels", asset_type=GAI + "LabelSet", storage_uri=uri, crs=ard.crs
    )
    return asset, {"positive_pixel_ratio": round(positive, 5), "labelled_tile_ratio": round(labelled_tiles, 4)}


def find_ard(ctx: ToolContext) -> DataAssetRef:
    ard = next((a for a in ctx.spec.inputs if str(a.asset_type).endswith("AnalysisReadyData")), None)
    if ard is None:
        raise ToolError("no_ard_input", "분석 준비 데이터 입력이 없습니다.")
    return ard


def run(ctx: ToolContext) -> ToolOutput:
    ard = find_ard(ctx)
    source = ctx.spec.parameters.get("label_source")
    if not source:
        raise ToolError("missing_parameter", "label_source(참조 벡터 GeoJSON 주소)가 필요합니다.")

    ctx.progress(5, "분석 준비 데이터·참조 벡터 읽기")
    manifest = get_json(ctx.store, ard.storage_uri)
    valid = load_npz(ctx.store, manifest["tiles_uri"])["valid"]
    geojson = read_geojson(ctx, source)
    src_crs = source_crs(geojson, ctx.spec.parameters.get("label_crs"))
    dst_crs = CRS.from_user_input(manifest["crs"])
    classes = ctx.spec.label_classes

    ctx.progress(20, "좌표 변환·기하 오류 수정")
    stats = {"features": 0, "fixed_invalid": 0, "dropped_empty": 0, "unmatched_class": 0}
    shapes: list[tuple] = []
    for g, props in polygons(geojson, src_crs, dst_crs, stats):
        value = class_value(props, classes)
        if value is None:
            stats["unmatched_class"] += 1
            continue
        shapes.append((g, value))

    ctx.progress(40, f"타일 래스터화 ({len(manifest['tiles'])}장)")
    masks = rasterize_tiles(ctx, shapes, manifest, valid)
    if not (masks > 0).any():
        raise ToolError(
            "label_no_overlap",
            f"참조 벡터({stats['features']}개 지물, {src_crs.to_string()})가 영상 범위와 겹치지 않습니다.",
        )
    asset, q = save_labelset(ctx, ard, masks, {"label_source": source, "source_crs": src_crs.to_string()}, stats)
    return ToolOutput(
        outputs=[asset],
        metrics={"label_features": float(len(shapes)), "fixed_invalid": float(stats["fixed_invalid"]), **q},
    )
