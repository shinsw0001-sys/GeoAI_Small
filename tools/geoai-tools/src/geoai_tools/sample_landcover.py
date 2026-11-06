"""개발·시험용 합성 토지피복 자료: 항공 영상과 토지피복 다각형(환경부 토지피복지도 형식).

실행 중 인터넷 의존 금지 규칙 때문에 공개 토지피복지도를 내려받지 않고, 같은 형식의 자료를 만든다.

- 항공 영상: 3밴드(RGB) 8비트 GeoTIFF, EPSG:5186, 0.5 m
- 참조 벡터: 토지피복 다각형 GeoJSON. 속성 L1_CODE(대분류 코드 100~700), L1_NAME(대분류 이름),
  농업지역은 L2_CODE(210 논, 220 밭)를 더한다. 장면 전체를 빈틈없이 덮는다(전면 분류).

장면 구성: 불규칙한 필지(보로노이 분할)를 저주파 지형 장으로 산림·농업·시가화·초지·나지에 배정하고,
장면을 가로지르는 하천(수역)과 그 둔치 일부·작은 못 둘레의 습지, 시가화 지역을 잇는 도로(시가화·건조지역의
교통 시설)를 얹는다. 논과 초지, 나지와 밝은 지붕처럼 색이 비슷한 항목을 섞어 질감까지 봐야 구분되게 했다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from shapely import affinity
from shapely.geometry import LineString, MultiPoint, Point, box, mapping, shape
from shapely.ops import unary_union, voronoi_diagram

from .sample import CRS, GSD, ORIGIN, ROOF_COLORS, _smooth

# (화소값, L1_CODE, L1_NAME). 화소값은 장면 내부 번호이고 라벨 화소값과 무관하다.
CLASSES = [
    (1, "100", "시가화건조지역"),
    (2, "200", "농업지역"),
    (3, "300", "산림지역"),
    (4, "400", "초지"),
    (5, "500", "습지"),
    (6, "600", "나지"),
    (7, "700", "수역"),
]
URBAN, AGRI, FOREST, GRASS, WETLAND, BARREN, WATER = range(1, 8)


def _bezier(points: list[tuple[float, float]], n: int = 80) -> LineString:
    p0, p1, p2 = (np.array(p) for p in points)
    t = np.linspace(0, 1, n)[:, None]
    return LineString((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2)


def _field(rng, h: int, w: int, k: int) -> np.ndarray:
    f = _smooth(rng.normal(size=(h, w)).astype(np.float32), k)
    return (f - f.min()) / (np.ptp(f) + 1e-6)


def scene(size: int, rng) -> tuple[np.ndarray, list[tuple], list[tuple]]:
    """(장면 화소값 지도, 필지 [(다각형, 화소값, 논 여부)], 시가화 건물 [(다각형)])."""
    h = w = size
    ident = rasterio.Affine.identity()
    frame = box(0, 0, w, h)
    terrain = _field(rng, h, w, max(8, size // 10))

    # 1) 필지: 보로노이 분할. 지형 장 값으로 항목을 고른다(높은 곳 산림, 낮고 평탄한 곳 농업, 일부 시가화)
    n_parcels = max(24, size * size // 9000)
    seeds = MultiPoint([(rng.uniform(0, w), rng.uniform(0, h)) for _ in range(n_parcels)])
    cells = [c.intersection(frame) for c in voronoi_diagram(seeds, envelope=frame).geoms]
    cells = [c for c in cells if not c.is_empty]
    at = np.array(
        [terrain[int(min(h - 1, max(0, c.centroid.y))), int(min(w - 1, max(0, c.centroid.x)))] for c in cells]
    )
    rank = at.argsort().argsort() / max(len(cells) - 1, 1)  # 지형 장 값의 순위(0~1)
    parcels = []
    for cell, r in zip(cells, rank, strict=True):
        u = rng.uniform()
        if r > 0.7:
            cls = FOREST if u < 0.85 else GRASS
        elif r < 0.2:
            cls = URBAN if u < 0.8 else BARREN
        else:
            cls = AGRI if u < 0.6 else GRASS if u < 0.85 else BARREN
        parcels.append((cell, cls, bool(rng.uniform() < 0.5)))
    cls_map = rasterize([(p, c) for p, c, _ in parcels], out_shape=(h, w), transform=ident, dtype="uint8")

    # 2) 하천(수역)과 둔치 습지, 작은 못과 둘레 습지
    a = (-10.0, rng.uniform(0.2, 0.8) * h)
    b = (w + 10.0, rng.uniform(0.2, 0.8) * h)
    mid = (w / 2 + rng.normal(0, w * 0.1), (a[1] + b[1]) / 2 + rng.normal(0, h * 0.2))
    river_line = _bezier([a, mid, b])
    river = river_line.buffer(rng.uniform(18, 30) / GSD / 2 * 1.0)
    banks = []
    for s in np.linspace(0.1, 0.9, 5):
        if rng.uniform() < 0.7:
            c = river_line.interpolate(s, normalized=True)
            banks.append(c.buffer(rng.uniform(40, 80) / GSD / 2))
    ponds = []
    for _ in range(max(2, size // 400)):
        c = Point(rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h)
        if c.distance(river) < 80:
            continue
        ponds.append(affinity.scale(c.buffer(rng.uniform(15, 30)), rng.uniform(0.7, 1.4), 1.0))
    wetland = unary_union([bk.intersection(river.buffer(30)) for bk in banks] + [p.buffer(16) for p in ponds])
    if not wetland.is_empty:
        cls_map[rasterize([wetland], out_shape=(h, w), transform=ident).astype(bool)] = WETLAND
    water = unary_union([river, *ponds])
    cls_map[rasterize([water], out_shape=(h, w), transform=ident).astype(bool)] = WATER

    # 3) 도로(시가화·건조지역의 교통 시설): 시가화 필지를 잇는 두 길. 하천 위는 다리라 도로가 덮는다
    urban_cells = [p for p, c, _ in parcels if c == URBAN] or [parcels[0][0]]
    roads = []
    for _ in range(2):
        p, q = rng.choice(len(urban_cells), 2, replace=len(urban_cells) < 2)
        a2, b2 = urban_cells[p].centroid, urban_cells[q].centroid
        direction = np.array([b2.x - a2.x, b2.y - a2.y])
        if np.hypot(*direction) < 1:
            direction = np.array([np.cos(rng.uniform(0, np.pi)), np.sin(rng.uniform(0, np.pi))])
        direction = direction / np.hypot(*direction)
        start = np.array([a2.x, a2.y]) - direction * size * 1.5
        end = np.array([a2.x, a2.y]) + direction * size * 1.5
        roads.append(LineString([tuple(start), tuple(end)]).intersection(frame).buffer(rng.uniform(6, 9) / GSD / 2))
    road = unary_union(roads)
    cls_map[rasterize([road], out_shape=(h, w), transform=ident).astype(bool)] = URBAN

    # 4) 시가화 필지 안 건물
    buildings = []
    for cell, cls, _ in parcels:
        if cls != URBAN:
            continue
        inner = cell.buffer(-6)
        if inner.is_empty:
            continue
        minx, miny, maxx, maxy = inner.bounds
        for _ in range(int(inner.area // 250) + 1):
            bw, bh = rng.uniform(14, 36), rng.uniform(12, 30)
            cx, cy = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            poly = affinity.rotate(box(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), float(rng.uniform(-20, 20)))
            if (
                inner.contains(poly)
                and not poly.intersects(road)
                and not any(poly.buffer(4).intersects(o) for o in buildings)
            ):
                buildings.append(poly)
    return cls_map, parcels, buildings


def render(cls_map: np.ndarray, parcels: list[tuple], buildings: list, rng) -> np.ndarray:
    h, w = cls_map.shape
    ident = rasterio.Affine.identity()
    lo = _field(rng, h, w, 24)
    hi = _smooth(rng.normal(size=(h, w)).astype(np.float32), 2)
    hi = (hi - hi.min()) / (np.ptp(hi) + 1e-6)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((3, h, w), dtype=np.float32)

    def paint(mask: np.ndarray, base, tex: np.ndarray, amp: float) -> None:
        for b in range(3):
            img[b][mask] = base[b] + (tex[mask] - 0.5) * amp

    # 농업: 필지마다 방향이 다른 이랑 줄무늬. 논(초록·황록)과 밭(갈색)
    agri = cls_map == AGRI
    idx = rasterize(
        [(p, i + 1) for i, (p, c, _) in enumerate(parcels) if c == AGRI],
        out_shape=(h, w),
        transform=ident,
        dtype="int32",
    )
    angle = np.concatenate([[0.0], rng.uniform(0, np.pi, len(parcels))])
    period = np.concatenate([[8.0], rng.uniform(6, 12, len(parcels))])
    paddy = np.concatenate([[False], [pd for _, _, pd in parcels]])
    stripe = 0.5 + 0.5 * np.sin((xx * np.cos(angle[idx]) + yy * np.sin(angle[idx])) * 2 * np.pi / period[idx])
    m = agri & paddy[idx]
    paint(m, (128, 158, 72), 0.6 * stripe + 0.4 * lo, 46)
    m = agri & ~paddy[idx]
    paint(m, (150, 118, 82), 0.6 * stripe + 0.4 * lo, 50)
    # 산림: 짙은 초록 수관 질감(고주파 반점)
    paint(cls_map == FOREST, (46, 86, 44), 0.7 * hi + 0.3 * lo, 60)
    # 초지: 밝은 초록, 매끈함
    paint(cls_map == GRASS, (118, 160, 82), lo, 40)
    # 습지: 초록·갈색에 물 반점
    m = cls_map == WETLAND
    paint(m, (86, 110, 74), 0.5 * hi + 0.5 * lo, 50)
    speck = m & (hi > 0.72)
    paint(speck, (58, 82, 96), lo, 20)
    # 나지: 밝은 모래·흙
    paint(cls_map == BARREN, (182, 162, 130), 0.5 * hi + 0.5 * lo, 44)
    # 수역: 어두운 청록, 잔물결
    paint(cls_map == WATER, (48, 78, 98), 0.3 * hi + 0.7 * lo, 24)
    # 시가화: 회색 포장 + 건물 지붕(그림자)
    paint(cls_map == URBAN, (128, 128, 126), 0.4 * hi + 0.6 * lo, 30)
    if buildings:
        shadow = rasterize([affinity.translate(p, 4, 4) for p in buildings], out_shape=(h, w), transform=ident)
        img[:, shadow.astype(bool) & (cls_map == URBAN)] *= 0.6
        index = rasterize(
            [(p, i + 1) for i, p in enumerate(buildings)], out_shape=(h, w), transform=ident, dtype="int32"
        )
        colors = np.vstack(
            [np.zeros((1, 3), np.float32), ROOF_COLORS[rng.integers(len(ROOF_COLORS), size=len(buildings))]]
        )
        roof = index > 0
        for b in range(3):
            img[b][roof] = colors[index[roof], b]
    # 필지마다 밝기를 조금씩 달리한다(촬영·작황 차이). 같은 항목이라도 색이 한 가지가 아니다
    pidx = rasterize(
        [(p, i + 1) for i, (p, _, _) in enumerate(parcels)], out_shape=(h, w), transform=ident, dtype="int32"
    )
    pcls = np.concatenate([[0], [c for _, c, _ in parcels]])[pidx]
    tone = np.where(pcls == cls_map, np.concatenate([[1.0], rng.uniform(0.86, 1.14, len(parcels))])[pidx], 1.0)
    img = img * tone[None]
    img = np.stack([_smooth(b, 1) for b in img]) + rng.normal(0, 4, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_landcover(
    out_dir: str | Path, size: int = 1024, seed: int = 31, name: str = "sample-landcover"
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cls_map, parcels, buildings = scene(size, rng)
    data = render(cls_map, parcels, buildings, rng)

    transform = from_origin(ORIGIN[0], ORIGIN[1], GSD, GSD)
    tif = out / f"{name}.tif"
    with rasterio.open(
        tif, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8", crs=CRS, transform=transform,
        tiled=True, blockxsize=256, blockysize=256, compress="deflate", photometric="RGB",
    ) as dst:  # fmt: skip
        dst.write(data)

    # 참조 벡터: 장면 화소값 지도를 다각형으로. 농업지역은 필지의 논·밭 구분을 L2_CODE 로 붙인다
    ident = rasterio.Affine.identity()
    paddy_map = rasterize(
        [(p, 1) for p, c, pd in parcels if c == AGRI and pd], out_shape=cls_map.shape, transform=ident, dtype="uint8"
    )
    key = cls_map.astype(np.int32) * 2 + ((cls_map == AGRI) & (paddy_map == 1))
    names = {v: (code, nm) for v, code, nm in CLASSES}
    features = []
    for geom, v in shapes(key, transform=transform):
        cls, is_paddy = int(v) // 2, int(v) % 2
        code, nm = names[cls]
        props = {"id": len(features), "L1_CODE": code, "L1_NAME": nm}
        if cls == AGRI:
            props["L2_CODE"] = "210" if is_paddy else "220"
        g = shape(geom)
        props["area_m2"] = round(g.area, 1)
        features.append({"type": "Feature", "properties": props, "geometry": mapping(g)})
    geojson = out / f"{name}-landcover.geojson"
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
