"""토지피복 분류 성능 검증 (job:LandCoverEvaluateJob, 진입점 landcover-evaluate).

입력: 모델(gai:Model), 라벨 세트(gai:LabelSet, 전면 분류: 0 = 미분류)
지표: 시험 집합(없으면 검증 집합)의 라벨 화소(미분류 제외)로 만든 혼동 행렬(행: 참조, 열: 예측) 기준 [Congalton 1991]
  overall_accuracy  전체 정확도 = 대각합 ÷ 전체
  kappa             카파 계수 = (전체 정확도 − 우연 일치) ÷ (1 − 우연 일치) [Cohen 1960]
  mIoU (= IoU)      대분류별 IoU 평균(시험 범위에 참조·예측 모두 없는 항목은 뺀다)
  F1                대분류별 F1 평균
  fwIoU             참조 면적 비율로 가중한 IoU
  항목별            생산자 정확도(재현율, 누락 오차의 보수), 사용자 정확도(정밀도, 포함 오차의 보수), IoU, F1,
                    참조·예측 면적(m²)과 면적 차이
산출: 성능 검증 보고(gai:EvaluationReport) report.json(혼동 행렬 포함), 결과 산출물(gai:ResultProduct)
      landcover.geojson(예측 분류도를 항목별로 합친 다각형, 분류 코드 L1_CODE), 미리보기 preview.png
      (타일마다 [영상 | 예측 분류도 | 참조 분류도] 세 칸, 미분류는 회색. 빨간색은 쓰지 않는다)
매개변수: min_object_m2(분류도 다각형 최소 면적, 25), preview_tiles(4)
"""

from __future__ import annotations

import json

import numpy as np
from geoai_common.messages import DataAssetRef
from geoai_common.objectstore import put_json
from rasterio import Affine
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from .evaluate import PALETTE, infer_split, judge, save_png
from .runtime import ToolContext, ToolOutput

GAI = "https://ontology.geoai.local/core#"
# 대분류 코드별 표시 색. 채도를 낮춘 색만 쓰고 빨간 계열은 넣지 않는다(운영 콘솔의 위험 표시 전용).
LANDCOVER_RGB = {
    "100": (150, 140, 160),  # 시가화·건조지역: 회보라
    "200": (222, 196, 104),  # 농업지역: 황토
    "300": (52, 120, 80),  # 산림지역: 짙은 초록
    "400": (156, 196, 112),  # 초지: 연두
    "500": (110, 170, 165),  # 습지: 청록 회색
    "600": (196, 168, 132),  # 나지: 모래
    "700": (72, 118, 186),  # 수역: 파랑
}
UNLABELLED_RGB = (214, 214, 214)


def class_colors(classes: list[dict]) -> np.ndarray:
    """화소값 → 색 표(0 = 미분류 회색)."""
    lut = np.zeros((max(int(c["value"]) for c in classes) + 1, 3), dtype=np.uint8)
    lut[0] = UNLABELLED_RGB
    for c in classes:
        v = int(c["value"])
        if v == 0:
            continue
        lut[v] = LANDCOVER_RGB.get(str(c.get("notation")), PALETTE[(v - 1) % len(PALETTE)])
    return lut


def accuracy_metrics(cm: np.ndarray) -> tuple[dict, list[dict]]:
    """혼동 행렬(K×K, 행: 참조, 열: 예측) → (전체 지표, 항목별 지표)."""
    cm = cm.astype(np.float64)
    n = cm.sum()
    tp = np.diag(cm)
    ref, pred = cm.sum(axis=1), cm.sum(axis=0)
    oa = tp.sum() / max(n, 1)
    pe = float((ref * pred).sum() / max(n * n, 1))
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 1.0
    present = (ref + pred) > 0
    iou = tp / np.maximum(ref + pred - tp, 1)
    pa = tp / np.maximum(ref, 1)
    ua = tp / np.maximum(pred, 1)
    f1 = 2 * pa * ua / np.maximum(pa + ua, 1e-12)
    rows = [
        {
            "producer_accuracy": round(float(pa[i]), 5) if ref[i] else None,
            "user_accuracy": round(float(ua[i]), 5) if pred[i] else None,
            "IoU": round(float(iou[i]), 5) if present[i] else None,
            "F1": round(float(f1[i]), 5) if present[i] else None,
        }
        for i in range(len(cm))
    ]
    miou = float(iou[present].mean()) if present.any() else 0.0
    metrics = {
        "overall_accuracy": float(oa),
        "kappa": float(kappa),
        "mIoU": miou,
        "IoU": miou,
        "F1": float(f1[present].mean()) if present.any() else 0.0,
        "fwIoU": float((ref / max(n, 1) * iou).sum()),
        "pixel_accuracy": float(oa),
    }
    return metrics, rows


def render_maps(images: np.ndarray, preds: np.ndarray, gts: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """타일마다 [영상 | 예측 분류도 | 참조 분류도] 한 줄. images: (N, 3, T, T)."""
    gap = 6
    n, _, t, _ = images.shape
    canvas = np.full((3, n * t + (n - 1) * gap, 3 * t + 2 * gap), 245, dtype=np.uint8)
    for k in range(n):
        y = k * (t + gap)
        panels = [images[k].astype(np.uint8), lut[preds[k]].transpose(2, 0, 1), lut[gts[k]].transpose(2, 0, 1)]
        for c, p in enumerate(panels):
            canvas[:, y : y + t, c * (t + gap) : c * (t + gap) + t] = p
    return canvas


def run(ctx: ToolContext) -> ToolOutput:
    inf = infer_split(ctx)
    labels, tiles, idx, pred = inf.labels, inf.tiles, inf.idx, inf.pred
    classes = [c for c in labels["classes"] if int(c["value"]) > 0]
    k = len(classes)
    ignore = labels.get("ignore_value", 0)
    gt = tiles.masks[idx]
    valid = tiles.valid[idx]
    scored = valid & (gt != ignore)
    gsd = float(tiles.manifest["ground_sample_distance"])
    px_m2 = gsd * gsd

    ctx.progress(65, "혼동 행렬·정확도 지표")
    p, g = pred[scored].astype(np.int64) - 1, gt[scored].astype(np.int64) - 1
    cm = np.bincount(g * k + p, minlength=k * k).reshape(k, k)
    metrics, rows = accuracy_metrics(cm)
    unlabelled_pred = np.bincount(pred[valid & (gt == ignore)].ravel(), minlength=k + 1)[1:]
    lut = class_colors(labels["classes"])
    class_summary = []
    for i, (c, r) in enumerate(zip(classes, rows, strict=True)):
        ref_m2, pred_m2 = float(cm[i].sum()) * px_m2, float(cm[:, i].sum()) * px_m2
        class_summary.append(
            {
                "value": int(c["value"]),
                "label": c["label"],
                "iri": c.get("iri"),
                "notation": c.get("notation"),
                "color": lut[int(c["value"])].tolist(),
                **r,
                "reference_m2": round(ref_m2, 1),
                "predicted_m2": round(pred_m2, 1),
                "area_difference_ratio": round((pred_m2 - ref_m2) / ref_m2, 4) if ref_m2 else None,
                "unlabelled_predicted_m2": round(float(unlabelled_pred[i]) * px_m2, 1),
            }
        )
    metrics = {key: round(float(v), 5) for key, v in metrics.items()}

    ctx.progress(75, "예측 분류도 다각형(항목별 합치기)")
    min_m2 = ctx.param("min_object_m2", 25.0)
    per_class: dict[int, list] = {int(c["value"]): [] for c in classes}
    for j, tid in enumerate(idx):
        transform = Affine(*tiles.manifest["tiles"][int(tid)]["transform"])
        for geom, v in shapes(pred[j], mask=valid[j] & (pred[j] > 0), transform=transform):
            per_class.setdefault(int(v), []).append(shape(geom))
    features, dropped = [], 0
    for c in classes:
        v = int(c["value"])
        merged = unary_union(per_class.get(v, []))
        for part in getattr(merged, "geoms", [merged]):
            if part.is_empty or part.geom_type != "Polygon":
                continue
            if part.area < min_m2:
                dropped += 1
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "L1_CODE": c.get("notation"),
                        "L1_NAME": c["label"],
                        "concept": (c.get("iri") or "").rsplit("#", 1)[-1],
                        "area_m2": round(part.area, 1),
                    },
                    "geometry": mapping(part.simplify(gsd, preserve_topology=True)),
                }
            )
    for row in class_summary:
        row["polygons_predicted"] = sum(f["properties"]["L1_NAME"] == row["label"] for f in features)

    ctx.progress(85, "보고서·미리보기 저장")
    # 참조 항목이 다양한 타일을 먼저 보여 준다
    n_prev = min(len(idx), ctx.param("preview_tiles", 4))
    order = sorted(range(len(idx)), key=lambda j: (-len(np.unique(gt[j][scored[j]])), j))[:n_prev]
    disp = tiles.manifest.get("display_bands") or [0, 1, 2]
    shown_pred = np.where(valid[order], pred[order], 0)
    preview_uri = save_png(ctx, render_maps(tiles.images[idx[order]][:, disp], shown_pred, gt[order], lut))

    crs = tiles.manifest["crs"]
    epsg = crs.split(":")[-1] if crs.upper().startswith("EPSG:") else None
    geojson = {"type": "FeatureCollection", "features": features}
    if epsg:
        geojson["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
    map_uri = ctx.out("evaluation", "landcover.geojson")
    ctx.store.put_bytes(map_uri, json.dumps(geojson, ensure_ascii=False).encode())

    report = {
        "kind": "landcover-evaluation",
        "model": inf.model_asset.storage_uri,
        "labelset": inf.labels_asset.storage_uri,
        "split": inf.split,
        "tiles": int(len(idx)),
        "metrics": metrics,
        "class_summary": class_summary,
        "confusion_matrix": {
            "axis": "행: 참조, 열: 예측 (화소 수)",
            "labels": [c["label"] for c in classes],
            "notations": [c.get("notation") for c in classes],
            "matrix": cm.astype(int).tolist(),
        },
        "scored_pixels": int(scored.sum()),
        "unlabelled_pixels": int((valid & (gt == ignore)).sum()),
        "map": {"polygons": len(features), "dropped_small": dropped, "min_object_m2": min_m2},
        "preview_layout": "landcover-triple",
        "completion_criteria": judge(ctx, metrics),
        "preview_uri": preview_uri,
        "predictions_uri": map_uri,
        "model_card": {
            "val_IoU": inf.card.get("val_IoU"),
            "best_epoch": inf.card.get("best_epoch"),
            "device": inf.card.get("device"),
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
                asset_id=f"{ctx.spec.task_id}.landcover",
                asset_type=GAI + "ResultProduct",
                storage_uri=map_uri,
                crs=crs,
            ),
        ],
        metrics=metrics,
    )
