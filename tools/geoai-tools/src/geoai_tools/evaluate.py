"""분할 모델 성능 검증 (job:SegmentationEvaluateJob, 진입점 segmentation-evaluate).

입력: 모델(gai:Model), 라벨 세트(gai:LabelSet). 영상 타일은 라벨 세트가 가리키는 분석 준비 데이터에서 읽는다.
지표: 시험 집합(없으면 검증 집합) 기준
  화소 단위  IoU(전경 교집합 대 합집합 비율), mIoU(배경 포함 평균), F1, precision(정밀도), recall(재현율)
  객체 단위  예측·정답 연결 요소를 IoU 0.5 이상으로 1:1 대응시켜 object_precision, object_recall, object_F1
  전경 개념이 둘 이상이면(변화탐지 등) 추가로
  전경 여부  foreground_IoU, foreground_F1, foreground_precision, foreground_recall (개념 무관, 전경 대 배경)
  개념별    class_summary: 개념마다 IoU, F1, 예측·정답 면적(m²), 객체 수
산출: 성능 검증 보고(gai:EvaluationReport) report.json, 결과 산출물(gai:ResultProduct) predictions.geojson,
      미리보기 preview.png
      - 단일 개념: 예측 호박색 채움, 정답 청록색 윤곽
      - 다개념: 개념 색 채움(예측)과 같은 색 윤곽(정답). 빨간색은 쓰지 않는다(운영 콘솔의 위험 표시 전용).
      - 다시기: 타일마다 [첫 시점 | 마지막 시점 + 예측·정답] 두 칸
매개변수: min_object_m2(객체 최소 면적, 8), preview_tiles(6)
"""

from __future__ import annotations

import io
import json
import operator
from dataclasses import dataclass

import numpy as np
import rasterio
import torch
from geoai_common.messages import DataAssetRef
from geoai_common.objectstore import get_json, put_json
from rasterio import Affine
from rasterio.features import shapes
from shapely.geometry import mapping, shape

from .data import TileSet, load_labelled
from .model import UNet, confusion, normalize, pick_device, pixel_metrics
from .runtime import ToolContext, ToolError, ToolOutput

GAI = "https://ontology.geoai.local/core#"
OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}
PRED_RGB = np.array([224, 160, 48], dtype=np.float32)  # 호박색
GT_RGB = np.array([32, 178, 170], dtype=np.uint8)  # 청록색
# 개념 화소값 1, 2, ... 순 색. 채도를 낮춘 호박·하늘·보라·초록·갈색·청회. 빨간 계열은 넣지 않는다.
PALETTE = np.array(
    [[224, 160, 48], [91, 143, 199], [140, 120, 196], [64, 168, 120], [176, 128, 88], [96, 160, 168]], dtype=np.uint8
)


def class_color(value: int) -> list[int]:
    return PALETTE[(value - 1) % len(PALETTE)].tolist()


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


def _overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, n_cls: int) -> np.ndarray:
    tile = image.astype(np.float32)
    if n_cls <= 2:
        fg = pred > 0
        tile[:, fg] = tile[:, fg] * 0.55 + PRED_RGB[:, None] * 0.45
        tile = tile.astype(np.uint8)
        tile[:, boundary(gt > 0)] = GT_RGB[:, None]
        return tile
    for v in range(1, n_cls):
        m = pred == v
        tile[:, m] = tile[:, m] * 0.5 + PALETTE[(v - 1) % len(PALETTE)].astype(np.float32)[:, None] * 0.5
    tile = tile.astype(np.uint8)
    for v in range(1, n_cls):
        tile[:, boundary(gt == v)] = PALETTE[(v - 1) % len(PALETTE)][:, None]
    return tile


def render_preview(
    images: np.ndarray,
    preds: np.ndarray,
    gts: np.ndarray,
    cols: int = 3,
    n_cls: int = 2,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """images: (N, 3, T, T) 표시 영상. reference 가 있으면(다시기) 타일마다 [비교 시점 | 표시 시점+겹침] 두 칸."""
    panels = []
    for k in range(len(images)):
        over = _overlay(images[k], preds[k], gts[k], n_cls)
        if reference is not None:
            gap = np.full((3, over.shape[1], 3), 245, dtype=np.uint8)
            over = np.concatenate([reference[k].astype(np.uint8), gap, over], axis=2)
        panels.append(over)
    if reference is not None:
        cols = 2
    n = len(panels)
    rows = -(-n // cols)
    th, tw = panels[0].shape[1:]
    gap = 6
    canvas = np.full((3, rows * th + (rows - 1) * gap, cols * tw + (cols - 1) * gap), 245, dtype=np.uint8)
    for k, tile in enumerate(panels):
        r, c = divmod(k, cols)
        canvas[:, r * (th + gap) : r * (th + gap) + th, c * (tw + gap) : c * (tw + gap) + tw] = tile
    return canvas


@dataclass
class Inference:
    """시험 집합 추론 결과. 분야 전용 성능 검증(도로망 등)도 이 결과에서 시작한다."""

    model_asset: DataAssetRef
    labels_asset: DataAssetRef
    card: dict
    labels: dict
    tiles: TileSet
    split: str
    idx: np.ndarray
    pred: np.ndarray
    n_cls: int


def infer_split(ctx: ToolContext) -> Inference:
    """입력 확인 → 모델·라벨 세트 읽기 → 시험(없으면 검증) 집합 추론."""
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
    if int(arch["in_channels"]) != int(tiles.images.shape[1]):
        raise ToolError(
            "band_mismatch", f"모델 입력 밴드 {arch['in_channels']}, 분석 준비 데이터 밴드 {tiles.images.shape[1]}"
        )
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
    return Inference(model_asset, labels_asset, card, labels, tiles, split, idx, pred, n_cls)


def judge(ctx: ToolContext, metrics: dict) -> list[dict]:
    """요청 완료 기준을 지표와 대조한다(보고서 기록용. 최종 판정은 Studio Agent가 한다)."""
    return [
        {
            **c,
            "value": metrics.get(c["metric"]),
            "met": metrics.get(c["metric"]) is not None
            and OPS.get(c["operator"], operator.ge)(metrics[c["metric"]], c["threshold"]),
        }
        for c in ctx.spec.completion_criteria
    ]


def save_png(ctx: ToolContext, canvas: np.ndarray, name: str = "preview.png") -> str:
    png = ctx.workdir / name
    with rasterio.open(
        png, "w", driver="PNG", width=canvas.shape[2], height=canvas.shape[1], count=3, dtype="uint8"
    ) as dst:
        dst.write(canvas)
    uri = ctx.out("evaluation", name)
    ctx.store.put_file(uri, png)
    return uri


def run(ctx: ToolContext) -> ToolOutput:
    inf = infer_split(ctx)
    model_asset, labels_asset, card, labels, tiles = inf.model_asset, inf.labels_asset, inf.card, inf.labels, inf.tiles
    split, idx, pred, n_cls = inf.split, inf.idx, inf.pred, inf.n_cls
    gt = tiles.masks[idx]
    cm = confusion(pred, gt, tiles.valid[idx], n_cls)
    metrics = pixel_metrics(cm)
    gsd = float(tiles.manifest["ground_sample_distance"])
    class_summary = []
    if n_cls > 2:
        fg = pixel_metrics(confusion((pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8), tiles.valid[idx], 2))
        metrics.update({f"foreground_{k}": fg[k] for k in ("IoU", "F1", "precision", "recall")})
    tp_c = np.diag(cm).astype(np.float64)
    for cls in labels["classes"][1:]:
        v = int(cls["value"])
        fp, fn = cm[:, v].sum() - tp_c[v], cm[v, :].sum() - tp_c[v]
        prec, rec = tp_c[v] / max(tp_c[v] + fp, 1), tp_c[v] / max(tp_c[v] + fn, 1)
        class_summary.append(
            {
                "value": v,
                "label": cls["label"],
                "iri": cls.get("iri"),
                "color": class_color(v),
                # 시험 집합에 정답·예측 모두 없는 유형은 판정 불가(None)
                "IoU": round(float(tp_c[v] / (tp_c[v] + fp + fn)), 5) if tp_c[v] + fp + fn else None,
                "F1": round(float(2 * prec * rec / max(prec + rec, 1e-12)), 5) if tp_c[v] + fp + fn else None,
                "predicted_m2": round(float(cm[:, v].sum()) * gsd * gsd, 1),
                "reference_m2": round(float(cm[v, :].sum()) * gsd * gsd, 1),
            }
        )

    ctx.progress(65, "객체 단위 대응·다각형 추출")
    min_px = max(1, int(ctx.param("min_object_m2", 8.0) / (gsd * gsd)))
    tp = n_pred = n_gt = 0
    per_class = {int(c["value"]): [0, 0, 0] for c in labels["classes"][1:]}  # 대응, 예측, 정답
    features = []
    for k, tid in enumerate(idx):
        transform = Affine(*tiles.manifest["tiles"][int(tid)]["transform"])
        for cls in labels["classes"][1:]:
            v = int(cls["value"])
            p_obj = objects(pred[k] == v, transform, min_px)
            g_obj = objects(gt[k] == v, transform, min_px)
            matched = match_objects(p_obj, g_obj)
            tp += matched
            n_pred += len(p_obj)
            n_gt += len(g_obj)
            per_class[v][0] += matched
            per_class[v][1] += len(p_obj)
            per_class[v][2] += len(g_obj)
            for p in p_obj:
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "class": cls["label"],
                            "concept": (cls.get("iri") or "").rsplit("#", 1)[-1],
                            "tile": int(tid),
                            "area_m2": round(p.area, 1),
                        },
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
    for row in class_summary:
        row["objects_matched"], row["objects_predicted"], row["objects_reference"] = per_class[row["value"]]
    k_prev = min(len(idx), ctx.param("preview_tiles", 6))
    # 정답이 있는 타일을 먼저 보여 준다(변화는 드물다).
    order = sorted(range(len(idx)), key=lambda k: -int((gt[k] > 0).sum()))[:k_prev]
    disp = tiles.manifest.get("display_bands") or [0, 1, 2]
    ref_bands = tiles.manifest.get("reference_display_bands")
    sel = idx[order]
    canvas = render_preview(
        tiles.images[sel][:, disp],
        pred[order],
        gt[order],
        n_cls=n_cls,
        reference=tiles.images[sel][:, ref_bands] if ref_bands else None,
    )
    preview_uri = save_png(ctx, canvas)

    crs = tiles.manifest["crs"]
    epsg = crs.split(":")[-1] if crs.upper().startswith("EPSG:") else None
    geojson = {"type": "FeatureCollection", "features": features}
    if epsg:
        geojson["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
    pred_uri = ctx.out("evaluation", "predictions.geojson")
    ctx.store.put_bytes(pred_uri, json.dumps(geojson, ensure_ascii=False).encode())

    criteria = judge(ctx, metrics)
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
        "class_summary": class_summary,
        "acquisitions": tiles.manifest.get("acquisitions", []),
        "preview_layout": "pair" if ref_bands else "single",
        "completion_criteria": criteria,
        "preview_uri": preview_uri,
        "predictions_uri": pred_uri,
        "model_card": {
            "val_IoU": card.get("val_IoU"),
            "best_epoch": card.get("best_epoch"),
            "device": card.get("device"),
            "pretrained": card.get("pretrained"),
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
