"""U-Net 분할 모델과 지표 계산.

U-Net은 부호기·복호기 사이를 건너뛰기 연결로 이어 경계를 정밀하게 복원하는 분할 구조다 [Ronneberger 2015].
기본 채널 수(base)로 크기를 조절한다. base=16 이면 약 190만 매개변수로, 중앙 처리 장치에서도 학습된다.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base: int = 16, depth: int = 4) -> None:
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([_block(in_channels, chans[0])])
        self.down.extend(_block(chans[i], chans[i + 1]) for i in range(depth))
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth)))
        self.dec = nn.ModuleList(_block(chans[i] * 2, chans[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chans[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec, strict=True):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


def normalize(images: np.ndarray, mean: list[float], std: list[float]) -> torch.Tensor:
    x = torch.from_numpy(images.astype(np.float32) / 255.0)
    m = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
    return (x - m) / s


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """전경 개념들의 연성 주사위 손실(soft Dice) 평균. 전경이 작은 영상의 불균형을 보완한다."""
    prob = logits.softmax(dim=1)[:, 1:]
    onehot = nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()[:, 1:]
    inter = (prob * onehot).sum(dim=(0, 2, 3))
    total = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    return 1 - ((2 * inter + 1) / (total + 1)).mean()


def confusion(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, num_classes: int) -> np.ndarray:
    p, t = pred[valid].astype(np.int64), target[valid].astype(np.int64)
    return np.bincount(t * num_classes + p, minlength=num_classes**2).reshape(num_classes, num_classes)


def pixel_metrics(cm: np.ndarray) -> dict[str, float]:
    """혼동 행렬(행: 정답, 열: 예측)에서 화소 지표. 전경 개념 평균(IoU, F1 ...)과 배경 포함 평균(mIoU)."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    fg = slice(1, None)
    return {
        "IoU": float(iou[fg].mean()),
        "mIoU": float(iou.mean()),
        "F1": float(f1[fg].mean()),
        "precision": float(precision[fg].mean()),
        "recall": float(recall[fg].mean()),
        "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)),
    }


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
