"""라이다 점군 읽기·쓰기와 격자화.

형식: LAS 1.2~1.4 (laspy [Brown 2026]). 압축 형식(LAZ)은 1년차에 다루지 않는다.
좌표 참조 체계는 LAS 의 OGC WKT 가변 길이 레코드(LASF_Projection, 2112)에서 읽고,
없으면 입력 자산의 좌표 참조 체계(gai:crs)를 쓴다. 분류 코드는 ASPRS LAS 규격을 따른다(2 지면, 5 높은 식생, 6 건물).

격자화 결과
- 수치 표면 모델(DSM): 칸마다 가장 높은 점. 빈 칸은 이웃 최댓값으로 몇 번 채운다.
- 수치 지형 모델(DTM): 지면 점(분류 2)의 칸별 최솟값을 굵은 격자(ground_cell_m)로 모으고
  빈 칸을 채운 뒤 원래 격자로 보간한다.
  지면 분류가 없으면 굵은 칸의 최솟값을 지면으로 본다.
- 정규화 수치 표면 모델(nDSM) = DSM − DTM (0 미만은 0). 지물(건물·수목) 높이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from rasterio import Affine

GROUND = 2
HIGH_VEGETATION = 5
BUILDING = 6


@dataclass
class PointCloud:
    xyz: np.ndarray  # (N, 3) float64
    classification: np.ndarray  # (N,) uint8
    crs_wkt: str | None


def read_points(path: str | Path) -> PointCloud:
    las = laspy.read(str(path))
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
    wkt = None
    for vlr in las.header.vlrs:
        if vlr.user_id == "LASF_Projection" and vlr.record_id == 2112:
            data = vlr.record_data if hasattr(vlr, "record_data") else vlr.record_data_bytes()
            wkt = bytes(data).rstrip(b"\0").decode("utf-8", errors="replace") or None
    return PointCloud(xyz=xyz, classification=np.asarray(las.classification, dtype=np.uint8), crs_wkt=wkt)


def write_points(path: str | Path, xyz: np.ndarray, classification: np.ndarray, crs_wkt: str | None) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.floor(xyz.min(axis=0))
    if crs_wkt:
        header.global_encoding.wkt = True
        header.vlrs.append(
            laspy.VLR(
                user_id="LASF_Projection", record_id=2112, description="OGC WKT", record_data=crs_wkt.encode() + b"\0"
            )
        )
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = classification.astype(np.uint8)
    las.write(str(path))


def _cells(xyz: np.ndarray, transform: Affine, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """점마다 격자 칸 번호(행 우선)와 격자 안에 드는지 여부."""
    inv = ~transform
    col, row = inv @ (xyz[:, 0], xyz[:, 1])
    col, row = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
    inside = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    return row * shape[1] + col, inside


def _fill_max(grid: np.ndarray, iterations: int) -> np.ndarray:
    """빈 칸(NaN)을 이웃 8칸의 최댓값으로 채운다. 점 밀도가 칸 크기보다 낮을 때의 구멍을 메운다."""
    g = grid.copy()
    for _ in range(iterations):
        empty = np.isnan(g)
        if not empty.any():
            break
        padded = np.pad(np.where(empty, -np.inf, g), 1, constant_values=-np.inf)
        h, w = g.shape
        neigh = np.max(
            [padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0
        )
        fill = empty & np.isfinite(neigh)
        g[fill] = neigh[fill]
    return g


def _fill_mean(grid: np.ndarray) -> np.ndarray:
    """빈 칸을 이웃 평균으로 다 찰 때까지 채운다(굵은 지면 격자용). 전부 비면 0."""
    g = grid.copy()
    if np.isnan(g).all():
        return np.zeros_like(g)
    while np.isnan(g).any():
        padded = np.pad(g, 1, constant_values=np.nan)
        h, w = g.shape
        stack = np.stack([padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        count = np.sum(~np.isnan(stack), axis=0)
        mean = np.nansum(stack, axis=0) / np.maximum(count, 1)
        fill = np.isnan(g) & (count > 0)
        g[fill] = mean[fill]
    return g


def _upsample(coarse: np.ndarray, factor: int, shape: tuple[int, int]) -> np.ndarray:
    """굵은 격자를 쌍선형 보간으로 원래 격자 크기로 늘린다."""
    ch, cw = coarse.shape
    ys = (np.arange(shape[0]) + 0.5) / factor - 0.5
    xs = (np.arange(shape[1]) + 0.5) / factor - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, ch - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, cw - 1)
    y1, x1 = np.clip(y0 + 1, 0, ch - 1), np.clip(x0 + 1, 0, cw - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]
    wx = np.clip(xs - x0, 0, 1)[None, :]
    top = coarse[y0][:, x0] * (1 - wx) + coarse[y0][:, x1] * wx
    bottom = coarse[y1][:, x0] * (1 - wx) + coarse[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def normalized_surface(
    cloud: PointCloud, transform: Affine, shape: tuple[int, int], ground_cell_m: float = 8.0, fill_iterations: int = 3
) -> tuple[np.ndarray, dict[str, float]]:
    """정규화 수치 표면 모델(float32, m)과 품질 지표. 점이 없는 곳은 0."""
    idx, inside = _cells(cloud.xyz, transform, shape)
    z = cloud.xyz[inside, 2]
    idx = idx[inside]
    cls = cloud.classification[inside]
    n_cells = shape[0] * shape[1]

    dsm = np.full(n_cells, -np.inf)
    np.maximum.at(dsm, idx, z)
    covered = np.isfinite(dsm)
    dsm = np.where(covered, dsm, np.nan).reshape(shape)
    dsm = _fill_max(dsm, fill_iterations)

    gsd = abs(transform.a)
    factor = max(1, int(round(ground_cell_m / gsd)))
    ch, cw = -(-shape[0] // factor), -(-shape[1] // factor)
    rows, cols = np.divmod(idx, shape[1])
    cidx = (rows // factor) * cw + cols // factor
    ground = cls == GROUND
    use = ground if ground.sum() >= 0.01 * max(len(z), 1) else np.ones_like(ground)
    dtm_c = np.full(ch * cw, np.inf)
    np.minimum.at(dtm_c, cidx[use], z[use])
    dtm_c = np.where(np.isfinite(dtm_c), dtm_c, np.nan).reshape(ch, cw)
    dtm = _upsample(_fill_mean(dtm_c), factor, shape)

    ndsm = np.where(np.isnan(dsm), 0.0, np.clip(dsm - dtm, 0.0, None)).astype(np.float32)
    quality = {
        "points": float(len(z)),
        "points_outside": float((~inside).sum()),
        "density_per_m2": round(len(z) / max(n_cells * gsd * gsd, 1e-9), 3),
        "cell_coverage": round(float(covered.mean()), 4),
        "filled_coverage": round(float((~np.isnan(dsm)).mean()), 4),
        "ground_points_ratio": round(float(ground.mean()) if len(z) else 0.0, 4),
    }
    return ndsm, quality
