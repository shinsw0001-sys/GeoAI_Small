"""분할 모델 성능 검증 (job:SegmentationEvaluateJob, 진입점 segmentation-evaluate).

입력: 모델(gai:Model), 라벨 세트(gai:LabelSet). 영상 타일은 라벨 세트가 가리키는 분석 준비 데이터에서 읽는다.
지표: 시험 집합(없으면 검증 집합) 기준
  화소 단위  IoU(전경 교집합 대 합집합 비율), mIoU(배경 포함 평균), F1, precision(정밀도), recall(재현율)
  객체 단위  예측·정답 연결 요소를 IoU 0.5 이상으로 1:1 대응시켜 object_precision, object_recall, object_F1
산출: 성능 검증 보고(gai:EvaluationReport) report.json, 결과 산출물(gai:ResultProduct) predictions.geojson,
      미리보기 preview.png(예측: 호박색 채움, 정답: 청록색 윤곽)
매개변수: min_object_m2(객체 최소 면적, 8), preview_tiles(6)
"""

from __future__ import annotations

import io
import json
import operator

import numpy as np
import rasterio
import torch
from geoai_common.messages import DataAssetRef
from geoai_common.objectstore import get_json, put_json
from rasterio import Affine
from rasterio.features import shapes
from shapely.geometry import mapping, shape

from .data import load_labelled
from .model import UNet, confusion, normalize, pick_device, pixel_metrics
from .runtime import ToolContext, ToolError, ToolOutput

GAI = "https://ontology.geoai.local/core#"
OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}
PRED_RGB = np.array([224, 160, 48], dtype=np.float32)  # 호박색
GT_RGB = np.array([32, 178, 170], dtype=np.uint8)  # 청록색


def objects(mask: np.ndarray, transform: Affine, min_px: int) -> list:
    return [
        shape(g)
        for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=transform)
        if v == 1 and shape(g).area >= min_px * abs(transform.a * transform.e)
    ]


def match_objects(pred: list, gt: list, thr: float = 0.5) -> int:
    """IoU 내림차순 탐욕 1:1 대응. 대응된 쌍의 수(참 양성)."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if p.intersects(g):
                inter = p.intersection(g).area
                iou = inter / (p.area + g.area - inter)
                if iou >= thr:
                    pairs.append((iou, i, j))
    used_p, used_g = set(), set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    return len(used_p)


def boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    inner = m.copy()
    inner[1:] &= m[:-1]
    inner[:-1] &= m[1:]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    return m & ~inner


def render_preview(images: np.ndarray, preds: np.ndarray, gts: np.ndarray, cols: int = 3) -> np.ndarray:
    n = len(images)
    rows = -(-n // cols)
    t = images.shape[-1]
    gap = 6
    canvas = np.full((3, rows * t + (rows - 1) * gap, cols * t + (cols - 1) * gap), 245, dtype=np.uint8)
    for k in range(n):
        tile = images[k].astype(np.float32)
        fg = preds[k] > 0
        tile[:, fg] = tile[:, fg] * 0.55 + PRED_RGB[:, None] * 0.45
        tile = tile.astype(np.uint8)
        edge = boundary(gts[k] > 0)
        tile[:, edge] = GT_RGB[:, None]
        r, c = divmod(k, cols)
        canvas[:, r * (t + gap) : r * (t + gap) + t, c * (t + gap) : c * (t + gap) + t] = tile
    return canvas


def run(ctx: ToolContext) -> ToolOutput:
    model_asset = next((a for a in ctx.spec.inputs if str(a.asset_type).endswith("#Model")), None)
    labels_asset = next((a for a in ctx.spec.inputs if str(a.asset_type).endswith("LabelSet")), None)
    if model_asset is None or labels_asset is None:
        raise ToolError("missing_input", "모델과 라벨 세트가 모두 필요합니다.")

    ctx.progress(5, "모델·시험 집합 읽기")
    card = get_json(ctx.store, model_asset.storage_uri)
    labels, tiles = load_labelled(ctx.store, labels_asset.storage_uri)
    arch = card["architecture"]
    n_cls = int(arch["num_classes"])
    if n_cls != len(labels["classes"]):
        raise ToolError("class_mismatch", f"모델 개념 수 {n_cls}, 라벨 세트 개념 수 {len(labels['classes'])}")
    split = "test" if len(tiles.split("test")) else "val"
    idx = tiles.split(split)
    if len(idx) == 0:
        raise ToolError("empty_split", "시험·검증 집합이 비었습니다.")

    device = pick_device()
    model = UNet(in_channels=arch["in_channels"], num_classes=n_cls, base=arch["base_channels"], depth=arch["depth"])
    state = torch.load(io.BytesIO(ctx.store.get_bytes(card["weights_uri"])), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    norm = card["normalization"]
    preds = []
    with torch.no_grad():
        for i in range(0, len(idx), 8):
            x = normalize(tiles.images[idx[i : i + 8]], norm["mean"], norm["std"]).to(device)
            preds.append(model(x).argmax(1).cpu().numpy().astype(np.uint8))
            ctx.progress(10 + 50 * min(i + 8, len(idx)) / len(idx), f"추론 {min(i + 8, len(idx))}/{len(idx)}")
    pred = np.concatenate(preds)
    pred[~tiles.valid[idx]] = 0
    gt = tiles.masks[idx]
    metrics = pixel_metrics(confusion(pred, gt, tiles.valid[idx], n_cls))

    ctx.progress(65, "객체 단위 대응·다각형 추출")
    gsd = float(tiles.manifest["ground_sample_distance"])
    min_px = max(1, int(ctx.param("min_object_m2", 8.0) / (gsd * gsd)))
    tp = n_pred = n_gt = 0
    features = []
    for k, tid in enumerate(idx):
        transform = Affine(*tiles.manifest["tiles"][int(tid)]["transform"])
        for cls in labels["classes"][1:]:
            v = int(cls["value"])
            p_obj = objects(pred[k] == v, transform, min_px)
            g_obj = objects(gt[k] == v, transform, min_px)
            tp += match_objects(p_obj, g_obj)
            n_pred += len(p_obj)
            n_gt += len(g_obj)
            for p in p_obj:
                features.append(
                    {
                        "type": "Feature",
                        "properties": {"class": cls["label"], "tile": int(tid), "area_m2": round(p.area, 1)},
                        "geometry": mapping(p.simplify(gsd, preserve_topology=True)),
                    }
                )
    op = tp / max(n_pred, 1)
    orc = tp / max(n_gt, 1)
    metrics.update(
        {
            "object_precision": op,
            "object_recall": orc,
            "object_F1": 2 * op * orc / max(op + orc, 1e-12),
        }
    )
    metrics = {k: round(float(v), 5) for k, v in metrics.items()}

    ctx.progress(85, "보고서·미리보기 저장")
    k_prev = min(len(idx), ctx.param("preview_tiles", 6))
    canvas = render_preview(tiles.images[idx[:k_prev]], pred[:k_prev], gt[:k_prev])
    png = ctx.workdir / "preview.png"
    with rasterio.open(
        png, "w", driver="PNG", width=canvas.shape[2], height=canvas.shape[1], count=3, dtype="uint8"
    ) as dst:
        dst.write(canvas)
    preview_uri = ctx.out("evaluation", "preview.png")
    ctx.store.put_file(preview_uri, png)

    crs = tiles.manifest["crs"]
    epsg = crs.split(":")[-1] if crs.upper().startswith("EPSG:") else None
    geojson = {"type": "FeatureCollection", "features": features}
    if epsg:
        geojson["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
    pred_uri = ctx.out("evaluation", "predictions.geojson")
    ctx.store.put_bytes(pred_uri, json.dumps(geojson, ensure_ascii=False).encode())

    criteria = [
        {
            **c,
            "value": metrics.get(c["metric"]),
            "met": c["metric"] in metrics and OPS.get(c["operator"], operator.ge)(metrics[c["metric"]], c["threshold"]),
        }
        for c in ctx.spec.completion_criteria
    ]
    report = {
        "kind": "segmentation-evaluation",
        "model": model_asset.storage_uri,
        "labelset": labels_asset.storage_uri,
        "split": split,
        "tiles": int(len(idx)),
        "objects": {
            "predicted": n_pred,
            "reference": n_gt,
            "matched": tp,
            "iou_threshold": 0.5,
            "min_object_m2": min_px * gsd * gsd,
        },
        "metrics": metrics,
        "completion_criteria": criteria,
        "preview_uri": preview_uri,
        "predictions_uri": pred_uri,
        "model_card": {
            "val_IoU": card.get("val_IoU"),
            "best_epoch": card.get("best_epoch"),
            "device": card.get("device"),
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
                asset_id=f"{ctx.spec.task_id}.predictions",
                asset_type=GAI + "ResultProduct",
                storage_uri=pred_uri,
                crs=crs,
            ),
        ],
        metrics=metrics,
    )
