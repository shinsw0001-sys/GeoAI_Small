"""개발·시험용 합성 항공 영상과 건물 윤곽 참조 벡터.

실행 중 인터넷 의존 금지 규칙 때문에 공개 데이터셋을 내려받지 않고, 같은 형식의 자료를
만들어 전 과정을 점검한다. 산출물은 실제 자료와 형식이 같다.

- 항공 영상: 3밴드(RGB) 8비트 GeoTIFF, EPSG:5186, 지상 표본 거리 0.5 m
- 참조 벡터: 건물 윤곽 다각형 GeoJSON (좌표는 EPSG:5186, `crs` 멤버로 표기)

장면 구성: 식생·나지 배경, 격자 도로, 블록 안의 회전된 직사각형 건물(지붕 색 5종, 그림자),
건물 가장자리를 가리는 수목. 회색 지붕과 도로가 비슷한 색이라 색만으로는 구분되지 않는다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import affinity
from shapely.geometry import Point, box, mapping

CRS = "EPSG:5186"
GSD = 0.5
ORIGIN = (198_000.0, 552_000.0)  # 좌상단 (동향, 북향), 중부원점 범위 안

ROOF_COLORS = np.array([[150, 150, 150], [92, 94, 98], [150, 78, 60], [70, 105, 150], [80, 130, 115]], dtype=np.float32)


def _smooth(a: np.ndarray, k: int) -> np.ndarray:
    """상자 필터 두 번(가우스 근사)."""
    for axis in (0, 1):
        for _ in range(2):
            c = np.cumsum(np.pad(a, [(k, k) if i == axis else (0, 0) for i in range(a.ndim)], mode="edge"), axis=axis)
            hi = np.take(c, np.arange(2 * k, c.shape[axis]), axis=axis)
            lo = np.take(c, np.arange(0, c.shape[axis] - 2 * k), axis=axis)
            a = (hi - lo) / (2 * k)
    return a


def generate(out_dir: str | Path, size: int = 2048, seed: int = 7, name: str = "sample-site") -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    transform = from_origin(ORIGIN[0], ORIGIN[1], GSD, GSD)
    h = w = size

    # 1) 배경: 식생과 나지가 섞인 저주파 장
    field = _smooth(rng.normal(size=(h, w)).astype(np.float32), 24)
    field = (field - field.min()) / (np.ptp(field) + 1e-6)
    veg = np.stack([60 + 30 * field, 105 + 40 * field, 50 + 20 * field])
    soil = np.stack([150 + 20 * field, 130 + 15 * field, 100 + 10 * field])
    mix = (_smooth(rng.normal(size=(h, w)).astype(np.float32), 40) > 0.02)[None]
    img = np.where(mix, veg, soil).astype(np.float32)

    # 2) 격자 도로
    road = np.zeros((h, w), dtype=bool)
    step = int(rng.integers(150, 190))
    lines = list(range(int(rng.integers(40, 90)), size, step))
    for c in lines:
        width = int(rng.integers(10, 18))
        road[:, c : c + width] = True
        road[c : c + width, :] = True
    img = np.where(road[None], np.array([118, 118, 120], dtype=np.float32)[:, None, None], img)

    # 3) 블록 안의 건물
    buildings = []
    blocks = [*zip(lines, lines[1:], strict=False)]
    for r0, r1 in blocks:
        for c0, c1 in blocks:
            for _ in range(int(rng.integers(3, 8))):
                bw, bh = rng.uniform(14, 44), rng.uniform(14, 40)  # 화소 (7~22 m)
                cx = rng.uniform(c0 + 24 + bw / 2, c1 - 6 - bw / 2) if c1 - c0 > bw + 32 else None
                cy = rng.uniform(r0 + 24 + bh / 2, r1 - 6 - bh / 2) if r1 - r0 > bh + 32 else None
                if cx is None or cy is None:
                    continue
                angle = float(rng.choice([0, 0, 0, rng.uniform(-35, 35)]))
                poly = affinity.rotate(box(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), angle)
                if any(poly.buffer(4).intersects(b) for b, _ in buildings):
                    continue
                buildings.append((poly, ROOF_COLORS[rng.integers(len(ROOF_COLORS))]))

    def to_world(poly):
        return affinity.affine_transform(poly, [GSD, 0, 0, -GSD, ORIGIN[0], ORIGIN[1]])

    ident = rasterio.Affine.identity()
    if buildings:
        # 그림자: 남동쪽으로 밀린 같은 윤곽
        shadow = rasterize([affinity.translate(p, 5, 5) for p, _ in buildings], out_shape=(h, w), transform=ident)
        img[:, shadow.astype(bool)] *= 0.45
        index = rasterize(
            [(p, i + 1) for i, (p, _) in enumerate(buildings)], out_shape=(h, w), transform=ident, dtype="int32"
        )
        xx = np.broadcast_to(np.arange(w)[None, :], (h, w))
        centers = np.array([0.0] + [p.centroid.x for p, _ in buildings])
        colors = np.vstack([np.zeros((1, 3), np.float32), np.stack([c for _, c in buildings])])
        roof = index > 0
        # 박공지붕 느낌의 밝기 차
        shade = np.where(xx < centers[index], 1.08, 0.92)[roof]
        for b in range(3):
            img[b][roof] = colors[index[roof], b] * shade

    # 4) 수목(건물 가장자리 일부를 가린다)
    n_trees = size * size // 9000
    trees = [(Point(rng.uniform(0, w), rng.uniform(0, h)).buffer(rng.uniform(3, 9)), i + 1) for i in range(n_trees)]
    tindex = rasterize(trees, out_shape=(h, w), transform=ident, dtype="int32")
    tone = np.concatenate([[1.0], rng.uniform(0.8, 1.1, n_trees)])[tindex]
    tmask = tindex > 0
    for b, base in enumerate((45, 85, 40)):
        img[b][tmask] = base * tone[tmask]

    # 5) 센서 잡음·약한 흐림
    img = np.stack([_smooth(b, 1) for b in img]) + rng.normal(0, 4, size=img.shape)
    data = np.clip(img, 0, 255).astype(np.uint8)

    tif = out / f"{name}.tif"
    with rasterio.open(
        tif, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=CRS, transform=transform,
        tiled=True, blockxsize=256, blockysize=256, compress="deflate", photometric="RGB",
    ) as dst:  # fmt: skip
        dst.write(data)

    features = [
        {
            "type": "Feature",
            "properties": {"id": i, "class": "Building", "area_m2": round(to_world(p).area, 1)},
            "geometry": mapping(to_world(p)),
        }
        for i, (p, _) in enumerate(buildings)
    ]
    geojson = out / f"{name}-buildings.geojson"
    geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::5186"}},
                "features": features,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return tif, geojson


def area_km2(size: int) -> float:
    return math.pow(size * GSD / 1000.0, 2)
