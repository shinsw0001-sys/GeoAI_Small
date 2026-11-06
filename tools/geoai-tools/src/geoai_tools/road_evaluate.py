"""도로망 성능 검증 (job:RoadNetworkEvaluateJob, 진입점 road-evaluate).

입력: 모델(gai:Model), 라벨 세트(gai:LabelSet, 도로 중심선 라벨 생성 산출물)
처리: 1) 시험 집합(없으면 검증 집합) 추론 — 범용 성능 검증과 같다
      2) 노면 화소 지표: IoU, F1, 정밀도, 재현율
      3) 시험 타일을 원래 격자 위치에 모자이크로 붙여 타일 경계를 넘는 도로를 하나로 잇는다
      4) 예측 노면 → 닫기·열기 → 세선화 → 골격 그래프 → 잔가지(spur_m)·작은 조각(min_component_m) 제거
      5) 참조 도로망: 라벨 세트의 분할된 중심선(reference_network.geojson)을 시험 범위로 자른 것.
         중심선이 없는 라벨(노면 다각형만)은 정답 마스크 골격으로 대신한다
      6) 중심선 지표(완전성·정확성·품질, 버퍼 buffer_m), APLS(제어 노드 간격 apls_spacing_m, 대응 반경
         apls_buffer_m), 경로 연결 비율, 교차점·끝점·연결 요소 수
산출: 성능 검증 보고(gai:EvaluationReport) report.json, 결과 산출물(gai:ResultProduct) road_network.geojson
      (예측 중심선, 길이 속성), 미리보기 preview.png(예측 노면 호박색, 예측 중심선 남색, 참조 중심선 청록색)
매개변수: buffer_m(3.0), apls_spacing_m(20.0), apls_buffer_m(4.0), spur_m(6.0), min_component_m(10.0),
          preview_tiles(6)
"""

from __future__ import annotations

import json

import numpy as np
from geoai_common.messages import DataAssetRef
from geoai_common.objectstore import put_json
from rasterio import Affine
from rasterio.features import rasterize
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from .evaluate import GT_RGB, PRED_RGB, infer_split, judge, save_png
from .model import confusion, pixel_metrics
from .road_network import (
    apls,
    buffer_metrics,
    clean_mask,
    dilate,
    lines_graph,
    node_counts,
    prune,
    skeleton_graph,
    thin,
    to_lines,
)
from .runtime import ToolContext, ToolError, ToolOutput

GAI = "https://ontology.geoai.local/core#"
CENTER_RGB = np.array([44, 72, 140], dtype=np.uint8)  # 남색: 예측 중심선


def mosaic(manifest: dict, idx: np.ndarray, layers: list[np.ndarray]) -> tuple[list[np.ndarray], float, float, dict]:
    """평가 타일을 원래 행·열 위치에 붙인다. (모자이크 목록, 좌상단 x, y, 타일 id → (행 오프셋, 열 오프셋))."""
    size = int(manifest["tile_size"])
    tiles = [manifest["tiles"][int(i)] for i in idx]
    r0 = min(t["row"] for t in tiles)
    c0 = min(t["col"] for t in tiles)
    nr = max(t["row"] for t in tiles) - r0 + 1
    nc = max(t["col"] for t in tiles) - c0 + 1
    out = [np.zeros((nr * size, nc * size), dtype=layer.dtype) for layer in layers]
    offsets = {}
    for k, t in enumerate(tiles):
        oy, ox = (t["row"] - r0) * size, (t["col"] - c0) * size
        offsets[int(idx[k])] = (oy, ox)
        for canvas, layer in zip(out, layers, strict=True):
            canvas[oy : oy + size, ox : ox + size] = layer[k]
    a = tiles[0]["transform"]
    x0 = a[2] - (tiles[0]["col"] - c0) * size * a[0]
    y0 = a[5] - (tiles[0]["row"] - r0) * size * a[4]
    return out, x0, y0, offsets


def network_from_mask(mask: np.ndarray, x0: float, y0: float, gsd: float, spur_m: float, min_m: float):
    """노면 마스크 → 도로망. 세선화는 선 끝을 폭의 절반만큼 깎으므로, 가장자리를 바깥으로 늘린(edge) 뒤 세선화하고
    잘라 낸다. 그러면 시험 범위 경계를 지나는 도로의 중심선이 경계까지 닿는다."""
    pad = 32
    skel = thin(clean_mask(np.pad(mask, pad, mode="edge"), 1))[pad:-pad, pad:-pad]
    g = skeleton_graph(skel, x0, y0, gsd)
    removed = prune(g, spur_m, min_m)
    return g, skel, removed


def run(ctx: ToolContext) -> ToolOutput:
    inf = infer_split(ctx)
    tiles, idx, pred, labels = inf.tiles, inf.idx, inf.pred, inf.labels
    manifest = tiles.manifest
    gsd = float(manifest["ground_sample_distance"])
    size = int(manifest["tile_size"])
    gt = tiles.masks[idx]
    valid = tiles.valid[idx]

    # 1) 노면 화소 지표 (도로 개념 하나: 배경 0, 도로 1)
    pm = pixel_metrics(confusion((pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8), valid, 2))
    metrics: dict[str, float] = {k: pm[k] for k in ("IoU", "F1", "precision", "recall", "mIoU", "pixel_accuracy")}

    # 2) 모자이크와 도로망
    ctx.progress(62, "시험 타일 모자이크·세선화")
    (pred_c, gt_c), x0, y0, offsets = mosaic(manifest, idx, [(pred > 0), (gt > 0)])
    spur_m = ctx.param("spur_m", 6.0)
    min_m = ctx.param("min_component_m", 10.0)
    pred_g, pred_skel, pruned = network_from_mask(pred_c, x0, y0, gsd, spur_m, min_m)

    area = unary_union([box(*_bounds(manifest["tiles"][int(i)]["transform"], size)) for i in idx])
    ref_uri = labels.get("reference_network_uri")
    if ref_uri:
        doc = json.loads(ctx.store.get_bytes(ref_uri))
        clipped = unary_union([shape(f["geometry"]) for f in doc.get("features", [])]).intersection(area)
        ref_lines = [g for g in getattr(clipped, "geoms", [clipped]) if g.geom_type == "LineString" and g.length > 0]
        ref_g = lines_graph(ref_lines, step=max(gsd, 1.0))
        ref_source = "centerline"
    else:
        ref_g, _, _ = network_from_mask(gt_c, x0, y0, gsd, spur_m, min_m)
        ref_lines = to_lines(ref_g, simplify=gsd)
        ref_source = "mask_skeleton"
    if ref_g.number_of_edges() == 0:
        where = "시험" if inf.split == "test" else "검증"
        raise ToolError("no_reference_network", f"{where} 집합 범위에 참조 도로가 없습니다.")

    ctx.progress(72, "중심선 대응·APLS 계산")
    pred_lines = to_lines(pred_g, simplify=gsd)
    bm = buffer_metrics(ref_lines, pred_lines, ctx.param("buffer_m", 3.0))
    ap = apls(
        ref_g,
        pred_g,
        spacing=ctx.param("apls_spacing_m", 20.0),
        buffer_m=ctx.param("apls_buffer_m", 4.0),
    )
    metrics.update(
        {
            "completeness": bm["completeness"],
            "correctness": bm["correctness"],
            "quality": bm["quality"],
            "APLS": ap["APLS"],
            "APLS_ref_to_pred": ap["APLS_ref_to_pred"],
            "APLS_pred_to_ref": ap["APLS_pred_to_ref"],
            "connectivity": ap["connectivity"] or 0.0,
            "reference_km": bm["reference_length_m"] / 1000,
            "predicted_km": bm["predicted_length_m"] / 1000,
        }
    )
    metrics = {k: round(float(v), 5) for k, v in metrics.items()}
    ref_nodes, pred_nodes = node_counts(ref_g), node_counts(pred_g)

    # 3) 예측 중심선 벡터
    ctx.progress(85, "보고서·미리보기 저장")
    crs = manifest["crs"]
    epsg = crs.split(":")[-1] if crs.upper().startswith("EPSG:") else None
    features = [
        {"type": "Feature", "properties": {"class": "도로", "concept": "Road", "length_m": round(ln.length, 2)},
         "geometry": mapping(ln)}
        for ln in pred_lines
    ]  # fmt: skip
    geojson = {"type": "FeatureCollection", "features": features}
    if epsg:
        geojson["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
    net_uri = ctx.out("evaluation", "road_network.geojson")
    ctx.store.put_bytes(net_uri, json.dumps(geojson, ensure_ascii=False).encode())

    # 4) 미리보기: 도로가 많은 타일 순
    k_prev = min(len(idx), ctx.param("preview_tiles", 6))
    order = sorted(range(len(idx)), key=lambda k: -int((gt[k] > 0).sum()))[:k_prev]
    disp = manifest.get("display_bands") or [0, 1, 2]
    panels = []
    for k in order:
        tid = int(idx[k])
        oy, ox = offsets[tid]
        tile = tiles.images[tid][disp].astype(np.float32)
        fg = pred[k] > 0
        tile[:, fg] = tile[:, fg] * 0.6 + PRED_RGB[:, None] * 0.4
        tile = tile.astype(np.uint8)
        transform = Affine(*manifest["tiles"][tid]["transform"])
        ext = box(*_bounds(manifest["tiles"][tid]["transform"], size))
        hits = [mapping(ln.buffer(gsd * 0.9)) for ln in ref_lines if ln.intersects(ext)]
        if hits:
            ref_px = rasterize(hits, out_shape=(size, size), transform=transform).astype(bool)
            tile[:, ref_px] = GT_RGB[:, None]
        skel = dilate(pred_skel[oy : oy + size, ox : ox + size], 1)
        tile[:, skel] = CENTER_RGB[:, None]
        panels.append(tile)
    preview_uri = save_png(ctx, _grid(panels))

    report = {
        "kind": "road-network-evaluation",
        "model": inf.model_asset.storage_uri,
        "labelset": inf.labels_asset.storage_uri,
        "split": inf.split,
        "tiles": int(len(idx)),
        "metrics": metrics,
        "network": {
            "reference_source": ref_source,
            "reference": {**ref_nodes, "length_km": round(bm["reference_length_m"] / 1000, 3)},
            "predicted": {**pred_nodes, "length_km": round(bm["predicted_length_m"] / 1000, 3)},
            "pruned": pruned,
            "apls_pairs": ap["pairs"],
            "parameters": {
                "buffer_m": ctx.param("buffer_m", 3.0),
                "apls_spacing_m": ctx.param("apls_spacing_m", 20.0),
                "apls_buffer_m": ctx.param("apls_buffer_m", 4.0),
                "spur_m": spur_m,
                "min_component_m": min_m,
            },
        },
        "class_summary": [],
        "acquisitions": manifest.get("acquisitions", []),
        "preview_layout": "network",
        "completion_criteria": judge(ctx, metrics),
        "preview_uri": preview_uri,
        "predictions_uri": net_uri,
        "model_card": {
            "val_IoU": inf.card.get("val_IoU"),
            "best_epoch": inf.card.get("best_epoch"),
            "device": inf.card.get("device"),
            "pretrained": inf.card.get("pretrained"),
        },
    }
    report_uri = ctx.out("evaluation", "report.json")
    put_json(ctx.store, report_uri, report)
    return ToolOutput(
        outputs=[
            DataAssetRef(
                asset_id=f"{ctx.spec.task_id}.report",
                asset_type=GAI + "EvaluationReport",
                storage_uri=report_uri,
                crs=crs,
            ),
            DataAssetRef(
                asset_id=f"{ctx.spec.task_id}.road-network",
                asset_type=GAI + "ResultProduct",
                storage_uri=net_uri,
                crs=crs,
            ),
        ],  # fmt: skip
        metrics=metrics,
    )


def _bounds(t: list[float], size: int) -> tuple[float, float, float, float]:
    a = Affine(*t)
    x0, y0 = a @ (0, 0)
    x1, y1 = a @ (size, size)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _grid(panels: list[np.ndarray], cols: int = 3, gap: int = 6) -> np.ndarray:
    rows = -(-len(panels) // cols)
    th, tw = panels[0].shape[1:]
    canvas = np.full((3, rows * th + (rows - 1) * gap, cols * tw + (cols - 1) * gap), 245, dtype=np.uint8)
    for k, tile in enumerate(panels):
        r, c = divmod(k, cols)
        canvas[:, r * (th + gap) : r * (th + gap) + th, c * (tw + gap) : c * (tw + gap) + tw] = tile
    return canvas
