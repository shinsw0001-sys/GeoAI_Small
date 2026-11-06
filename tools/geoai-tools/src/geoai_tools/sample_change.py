"""개발·시험용 합성 변화탐지 자료: 두 시점 항공 영상, 두 시점 라이다 점군, 시점별 건물 윤곽·토지피복 벡터.

실행 중 인터넷 의존 금지 규칙 때문에 공개 변화탐지 자료를 내려받지 않고, 같은 형식의 자료를 만든다.

- 항공 영상: 3밴드 8비트 GeoTIFF, EPSG:5186, 0.5 m. 이후 시점은 조도·색조가 조금 다르다(촬영 조건 차이).
- 라이다 점군: LAS 1.4(점 형식 6), 기본 3점/m², 분류 2 지면·5 높은 식생·6 건물, 좌표 참조 체계 WKT 기록.
- 건물 윤곽: 시점별 GeoJSON(class=Building).
- 토지피복: 시점별 GeoJSON(class=환경부 토지피복지도 대분류 이름). 변화 구역과 변하지 않은 구역을 함께 둔다.

두 시점 사이의 변화: 건물 멸실(약 15%), 건물 변경(약 15%, 한쪽으로 증축), 건물 신축(빈 터),
토지 변화(초지 ↔ 나지). 수목·도로는 같다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import affinity
from shapely.geometry import Point, box, mapping

from .lidar import BUILDING, GROUND, HIGH_VEGETATION, write_points
from .sample import CRS, GSD, ORIGIN, ROOF_COLORS, _smooth

DATES = ("2023-05-12", "2025-05-20")
SOIL = np.array([152, 134, 104], dtype=np.float32)
GRASS = np.array([88, 132, 64], dtype=np.float32)


@dataclass
class Building:
    poly: object  # 화소 좌표 다각형
    color: np.ndarray
    height: float


@dataclass
class Scene:
    size: int
    field: np.ndarray
    base: np.ndarray  # (3, H, W) 배경(식생·나지·도로)
    road: np.ndarray
    trees: list = field(default_factory=list)  # (원, 높이)
    buildings: list[list[Building]] = field(default_factory=list)  # 시점별
    land: list[list[tuple]] = field(default_factory=list)  # 시점별 (다각형, 대분류 이름)


def _place(rng, region, existing, wmin=14, wmax=40) -> object | None:
    r0, r1, c0, c1 = region
    bw, bh = rng.uniform(wmin, wmax), rng.uniform(wmin, wmax)
    if c1 - c0 <= bw + 32 or r1 - r0 <= bh + 32:
        return None
    cx = rng.uniform(c0 + 24 + bw / 2, c1 - 6 - bw / 2)
    cy = rng.uniform(r0 + 24 + bh / 2, r1 - 6 - bh / 2)
    poly = affinity.rotate(
        box(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), float(rng.choice([0, 0, rng.uniform(-30, 30)]))
    )
    if any(poly.buffer(5).intersects(p) for p in existing):
        return None
    return poly


def build_scene(size: int, seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    h = w = size
    fld = _smooth(rng.normal(size=(h, w)).astype(np.float32), 24)
    fld = (fld - fld.min()) / (np.ptp(fld) + 1e-6)
    veg = np.stack([60 + 30 * fld, 105 + 40 * fld, 50 + 20 * fld])
    soil = np.stack([150 + 20 * fld, 130 + 15 * fld, 100 + 10 * fld])
    mix = (_smooth(rng.normal(size=(h, w)).astype(np.float32), 40) > 0.02)[None]
    base = np.where(mix, veg, soil).astype(np.float32)

    road = np.zeros((h, w), dtype=bool)
    step = int(rng.integers(150, 190))
    lines = list(range(int(rng.integers(40, 90)), size, step))
    for c in lines:
        width = int(rng.integers(10, 18))
        road[:, c : c + width] = True
        road[c : c + width, :] = True
    base = np.where(road[None], np.array([118, 118, 120], dtype=np.float32)[:, None, None], base)
    scene = Scene(size=size, field=fld, base=base, road=road)

    # 시점 1 건물
    regions = [
        (r0, r1, c0, c1)
        for r0, r1 in zip(lines, lines[1:], strict=False)
        for c0, c1 in zip(lines, lines[1:], strict=False)
    ]
    t1: list[Building] = []
    for region in regions:
        for _ in range(int(rng.integers(2, 6))):
            poly = _place(rng, region, [b.poly for b in t1])
            if poly is not None:
                t1.append(Building(poly, ROOF_COLORS[rng.integers(len(ROOF_COLORS))], float(rng.uniform(4, 28))))

    # 시점 2: 멸실·변경·신축
    t2: list[Building] = []
    fate = rng.choice(["keep", "demolish", "modify"], size=len(t1), p=[0.6, 0.2, 0.2])
    # 시험 자료에 모든 변화 유형이 들어가도록 최소 1동씩 둔다.
    for k, kind in enumerate(("demolish", "modify")):
        if len(t1) > 2 and kind not in fate:
            fate[k] = kind
    for b, f in zip(t1, fate, strict=True):
        if f == "demolish":
            continue
        if f == "modify":
            minx, miny, maxx, maxy = b.poly.bounds
            sx, sy = (rng.uniform(1.35, 1.7), 1.0) if rng.random() < 0.5 else (1.0, rng.uniform(1.35, 1.7))
            grown = affinity.scale(b.poly, sx, sy, origin=(minx, miny))
            others = [o.poly for o in t1 if o is not b]
            if (
                grown.bounds[2] < size - 4
                and grown.bounds[3] < size - 4
                and not any(grown.buffer(3).intersects(o) for o in others)
            ):
                t2.append(Building(grown, b.color, b.height))
                continue
        t2.append(b)
    occupied = [b.poly for b in t1] + [b.poly for b in t2]
    for region in regions:
        for _ in range(4):  # 빈 터에 신축 (자리가 나면 블록당 최대 4동)
            poly = _place(rng, region, occupied) if rng.random() < 0.6 else None
            if poly is not None and not road[int(poly.centroid.y), int(poly.centroid.x)]:
                t2.append(Building(poly, ROOF_COLORS[rng.integers(len(ROOF_COLORS))], float(rng.uniform(4, 28))))
                occupied.append(poly)
    scene.buildings = [t1, t2]

    # 토지피복 구역: 변화(초지↔나지)와 무변화
    land1, land2 = [], []
    for _ in range(max(4, size * size // 30000)):
        c = Point(rng.uniform(30, w - 30), rng.uniform(30, h - 30))
        patch = affinity.scale(c.buffer(rng.uniform(20, 45)), rng.uniform(0.7, 1.4), 1.0)
        if any(patch.buffer(4).intersects(p) for p in occupied) or any(patch.intersects(p) for p, _ in land1):
            continue
        cls1 = "초지" if rng.random() < 0.5 else "나지"
        changed = rng.random() < 0.65
        cls2 = ("나지" if cls1 == "초지" else "초지") if changed else cls1
        land1.append((patch, cls1))
        land2.append((patch, cls2))
    scene.land = [land1, land2]

    n_trees = size * size // 9000
    scene.trees = [
        (Point(rng.uniform(0, w), rng.uniform(0, h)).buffer(rng.uniform(3, 9)), float(rng.uniform(6, 12)))
        for _ in range(n_trees)
    ]
    # 변화 구역·새 건물 자리 위의 수목은 없앤다(라벨과 겹치지 않게).
    keep_out = [p for p, _ in land1] + [b.poly for b in t2] + [b.poly for b in t1]
    scene.trees = [(t, ht) for t, ht in scene.trees if not any(t.intersects(k) for k in keep_out)]
    return scene


def _land_color(cls: str, fld: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = GRASS if cls == "초지" else SOIL
    return base[:, None] + (fld[mask] - 0.5)[None, :] * 30


def render(scene: Scene, epoch: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """시점 영상(3, H, W uint8)과 지물 높이(H, W, m)."""
    rng = np.random.default_rng(seed + 101 * (epoch + 1))
    h = w = scene.size
    ident = rasterio.Affine.identity()
    img = scene.base.copy()
    height = np.zeros((h, w), dtype=np.float32)

    for poly, cls in scene.land[epoch]:
        m = rasterize([poly], out_shape=(h, w), transform=ident).astype(bool) & ~scene.road
        img[:, m] = _land_color(cls, scene.field, m)

    buildings = scene.buildings[epoch]
    if epoch > 0:  # 멸실 자리: 정지된 나지
        gone = [b.poly for b in scene.buildings[0] if all(b.poly is not x.poly for x in buildings)]
        if gone:
            m = rasterize(gone, out_shape=(h, w), transform=ident).astype(bool)
            img[:, m] = _land_color("나지", scene.field, m)
    if buildings:
        shadow = rasterize([affinity.translate(b.poly, 5, 5) for b in buildings], out_shape=(h, w), transform=ident)
        img[:, shadow.astype(bool)] *= 0.45
        index = rasterize(
            [(b.poly, i + 1) for i, b in enumerate(buildings)], out_shape=(h, w), transform=ident, dtype="int32"
        )
        xx = np.broadcast_to(np.arange(w)[None, :], (h, w))
        centers = np.array([0.0] + [b.poly.centroid.x for b in buildings])
        colors = np.vstack([np.zeros((1, 3), np.float32), np.stack([b.color for b in buildings])])
        heights = np.array([0.0] + [b.height for b in buildings], dtype=np.float32)
        roof = index > 0
        shade = np.where(xx < centers[index], 1.08, 0.92)[roof]
        for b in range(3):
            img[b][roof] = colors[index[roof], b] * shade
        height = heights[index]

    if scene.trees:
        tindex = rasterize(
            [(t, i + 1) for i, (t, _) in enumerate(scene.trees)], out_shape=(h, w), transform=ident, dtype="int32"
        )
        tmask = tindex > 0
        tone = np.concatenate([[1.0], np.random.default_rng(seed).uniform(0.8, 1.1, len(scene.trees))])[tindex]
        for b, base in enumerate((45, 85, 40)):
            img[b][tmask] = base * tone[tmask]
        theights = np.array([0.0] + [ht for _, ht in scene.trees], dtype=np.float32)
        height = np.where(tmask & (height == 0), theights[tindex], height)

    # 촬영 조건 차이: 이후 시점은 조금 밝고 따뜻한 색조
    if epoch > 0:
        img = img * np.array([1.06, 1.03, 0.97], dtype=np.float32)[:, None, None] + 6
    img = np.stack([_smooth(b, 1) for b in img]) + rng.normal(0, 4, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), height


def lidar_points(
    scene: Scene, height: np.ndarray, epoch: int, seed: int, density: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 977 * (epoch + 1))
    size = scene.size
    n = int(density * (size * GSD) ** 2)
    px = rng.uniform(0, size, n)
    py = rng.uniform(0, size, n)
    r, c = np.minimum(py.astype(int), size - 1), np.minimum(px.astype(int), size - 1)
    x = ORIGIN[0] + px * GSD
    y = ORIGIN[1] - py * GSD
    ground = 32.0 + 0.012 * (px * GSD) - 0.008 * (py * GSD) + 2.0 * scene.field[r, c]
    above = height[r, c]
    z = ground + above + rng.normal(0, 0.05, n)
    cls = np.full(n, GROUND, dtype=np.uint8)
    roofs = (
        rasterize(
            [b.poly for b in scene.buildings[epoch]], out_shape=(size, size), transform=rasterio.Affine.identity()
        )
        if scene.buildings[epoch]
        else np.zeros((size, size), np.uint8)
    )
    cls[(above > 0) & (roofs[r, c] > 0)] = BUILDING
    cls[(above > 0) & (roofs[r, c] == 0)] = HIGH_VEGETATION
    return np.column_stack([x, y, z]), cls


def _geojson(features: list[dict]) -> dict:
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::5186"}},
        "features": features,
    }


def generate_change(out_dir: str | Path, size: int = 1024, seed: int = 11, density: float = 3.0) -> list[dict]:
    """시점별 파일 경로 목록 [{acquired_at, image, lidar, buildings, landcover}]."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene(size, seed)
    transform = from_origin(ORIGIN[0], ORIGIN[1], GSD, GSD)
    wkt = RioCRS.from_user_input(CRS).to_wkt()

    def to_world(poly):
        return affinity.affine_transform(poly, [GSD, 0, 0, -GSD, ORIGIN[0], ORIGIN[1]])

    files = []
    for epoch, date in enumerate(DATES):
        tag = date.replace("-", "")
        img, height = render(scene, epoch, seed)
        tif = out / f"site-{tag}.tif"
        with rasterio.open(
            tif, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8", crs=CRS, transform=transform,
            tiled=True, blockxsize=256, blockysize=256, compress="deflate", photometric="RGB",
        ) as dst:  # fmt: skip
            dst.write(img)
        las = out / f"site-{tag}.las"
        xyz, cls = lidar_points(scene, height, epoch, seed, density)
        write_points(las, xyz, cls, wkt)
        bld = out / f"site-{tag}-buildings.geojson"
        bld.write_text(
            json.dumps(
                _geojson(
                    [
                        {
                            "type": "Feature",
                            "properties": {"id": i, "class": "Building", "height_m": round(b.height, 1)},
                            "geometry": mapping(to_world(b.poly)),
                        }
                        for i, b in enumerate(scene.buildings[epoch])
                    ]
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        land = out / f"site-{tag}-landcover.geojson"
        land.write_text(
            json.dumps(
                _geojson(
                    [
                        {"type": "Feature", "properties": {"id": i, "class": c}, "geometry": mapping(to_world(p))}
                        for i, (p, c) in enumerate(scene.land[epoch])
                    ]
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        files.append({"acquired_at": date, "image": tif, "lidar": las, "buildings": bld, "landcover": land})
    return files
