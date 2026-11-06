"""분할 모델 학습 (job:SegmentationTrainJob, 진입점 segmentation-train).

입력: 분석 준비 데이터(gai:AnalysisReadyData), 라벨 세트(gai:LabelSet)
처리: U-Net을 학습 집합으로 학습하고, 세대(epoch)마다 검증 집합 IoU를 재서 가장 좋은 가중치를 남긴다.
      손실은 교차 엔트로피 + 연성 주사위 손실. 뒤집기·90도 회전으로 증강한다.
      사전 학습 가중치(pretrained_uri)는 데이터·모델 저장소에서만 읽는다(실행 중 인터넷 의존 금지).
산출: 모델(gai:Model) = model.json(구조·정규화·학습 이력) + model.pt(가중치)
매개변수: epochs(20), batch_size(8), learning_rate(0.001), base_channels(16), seed(42), pretrained_uri(선택)
"""

from __future__ import annotations

import io
import time

import numpy as np
import torch
from geoai_common.messages import DataAssetRef
from geoai_common.objectstore import put_json
from torch import nn

from .data import load_labelled
from .model import UNet, confusion, dice_loss, normalize, pick_device, pixel_metrics
from .runtime import ToolContext, ToolError, ToolOutput

GAI = "https://ontology.geoai.local/core#"


def augment(x: torch.Tensor, y: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    k = int(torch.randint(0, 4, (1,), generator=gen))
    x, y = torch.rot90(x, k, dims=(2, 3)), torch.rot90(y, k, dims=(1, 2))
    if torch.rand(1, generator=gen) < 0.5:
        x, y = x.flip(3), y.flip(2)
    return x, y


def evaluate_split(model: nn.Module, x: torch.Tensor, y: np.ndarray, valid: np.ndarray, n_cls: int, device) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 8):
            preds.append(model(x[i : i + 8].to(device)).argmax(1).cpu().numpy())
    return pixel_metrics(confusion(np.concatenate(preds), y, valid, n_cls))


def run(ctx: ToolContext) -> ToolOutput:
    labels_asset = next((a for a in ctx.spec.inputs if str(a.asset_type).endswith("LabelSet")), None)
    if labels_asset is None:
        raise ToolError("no_labelset_input", "라벨 세트 입력이 없습니다.")
    epochs = ctx.param("epochs", 20)
    batch = ctx.param("batch_size", 8)
    lr = ctx.param("learning_rate", 0.001)
    base = ctx.param("base_channels", 16)
    seed = ctx.param("seed", 42)

    ctx.progress(2, "라벨 세트·타일 읽기")
    labels, tiles = load_labelled(ctx.store, labels_asset.storage_uri)
    n_cls = len(labels["classes"])
    norm = tiles.manifest["normalization"]
    train_idx, val_idx = tiles.split("train"), tiles.split("val")
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ToolError("empty_split", f"학습 {len(train_idx)}장, 검증 {len(val_idx)}장. 둘 다 1장 이상이어야 합니다.")

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    device = pick_device()
    x_train = normalize(tiles.images[train_idx], norm["mean"], norm["std"])
    y_train = torch.from_numpy(tiles.masks[train_idx].astype(np.int64))
    x_val = normalize(tiles.images[val_idx], norm["mean"], norm["std"])

    model = UNet(num_classes=n_cls, base=base).to(device)
    pretrained = ctx.spec.parameters.get("pretrained_uri")
    if pretrained:
        state = torch.load(io.BytesIO(ctx.store.get_bytes(pretrained)), map_location=device, weights_only=True)
        model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 3, total_steps=epochs * -(-len(train_idx) // batch))
    ce = nn.CrossEntropyLoss()

    history, best, best_state, best_epoch = [], -1.0, None, 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_idx), generator=gen)
        losses = []
        for i in range(0, len(order), batch):
            sel = order[i : i + batch]
            xb, yb = augment(x_train[sel], y_train[sel], gen)
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb) + dice_loss(logits, yb, n_cls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        val = evaluate_split(model, x_val, tiles.masks[val_idx], tiles.valid[val_idx], n_cls, device)
        history.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 5), "val_IoU": round(val["IoU"], 5)})
        if val["IoU"] > best:
            best, best_epoch = val["IoU"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ctx.progress(
            5 + 90 * epoch / epochs,
            f"세대 {epoch}/{epochs} 손실 {np.mean(losses):.3f} 검증 IoU {val['IoU']:.3f}",
            force=True,
        )

    buf = io.BytesIO()
    torch.save(best_state, buf)
    weights_uri = ctx.out("model", "model.pt")
    ctx.store.put_bytes(weights_uri, buf.getvalue())
    card = {
        "kind": "segmentation-model",
        "architecture": {"name": "unet", "in_channels": 3, "num_classes": n_cls, "base_channels": base, "depth": 4},
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "classes": labels["classes"],
        "normalization": norm,
        "tile_size": tiles.manifest["tile_size"],
        "crs": tiles.manifest["crs"],
        "weights_uri": weights_uri,
        "trained_on": {"labelset": labels_asset.storage_uri, "train_tiles": len(train_idx), "val_tiles": len(val_idx)},
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch,
            "learning_rate": lr,
            "seed": seed,
            "pretrained_uri": pretrained,
        },
        "device": str(device),
        "train_seconds": round(time.monotonic() - started, 1),
        "best_epoch": best_epoch,
        "val_IoU": round(best, 5),
        "history": history,
    }
    card_uri = ctx.out("model", "model.json")
    put_json(ctx.store, card_uri, card)
    return ToolOutput(
        outputs=[
            DataAssetRef(
                asset_id=f"{ctx.spec.task_id}.model",
                asset_type=GAI + "Model",
                storage_uri=card_uri,
                crs=labels_asset.crs,
            )
        ],
        metrics={
            "val_IoU": round(best, 5),
            "best_epoch": float(best_epoch),
            "epochs": float(epochs),
            "train_seconds": card["train_seconds"],
        },
    )
