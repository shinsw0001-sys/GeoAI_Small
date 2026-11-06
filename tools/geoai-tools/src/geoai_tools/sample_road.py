"""개발·시험용 합성 도로 자료: 항공 영상과 도로 중심선 참조 벡터(폭 속성 포함).

실행 중 인터넷 의존 금지 규칙 때문에 공개 도로 자료를 내려받지 않고, 같은 형식의 자료를 만든다.

- 항공 영상: 3밴드(RGB) 8비트 GeoTIFF, EPSG:5186, 0.5 m
- 참조 벡터: 도로 중심선 LineString GeoJSON. 속성 class=Road, width_m(도로 폭, m), road_type(간선·국지·막다른 길)

장면 구성: 장면을 가로지르는 간선도로(폭 10~14 m, 중앙선 표시), 간선 사이를 잇는 굽은 국지도로(5~8 m),
국지도로에서 갈라지는 막다른 길(4~5 m). 교차점에서 중심선이 정확히 만난다(도로망 위상).
도로변 건물(회색 지붕 포함), 도로와 색이 같은 포장 주차장(도로 아님), 도로를 일부 가리는 가로수.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import affinity
from shapely.geometry import LineString, Point, box, mapping
from shapely.ops import unary_union

from .sample import CRS, GSD, ORIGIN, ROOF_COLORS, _smooth

ASPHALT = np.array([104, 106, 110], dtype=np.float32)
MARKING = np.array([226, 222, 200], dtype=np.float32)


def _bezier(p0, p1, p2, n: int = 40) -> LineString:
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t**2 * np.array(p2)
    return LineString(pts)


def road_lines(size: int, rng) -> list[tuple[LineString, float, str]]:
    """(화소 좌표 중심선, 폭 m, 도로 유형). 국지도로·막다른 길은 기존 도로 위 점에서 시작해 교차점이 정확히 맞는다."""
    roads: list[tuple[LineString, float, str]] = []
    # 1) 간선: 가로·세로 방향으로 고르게 퍼진 직선(조금씩 기울어짐). 1024 화소면 2 + 2 개, 교차점 4곳
    n_dir = max(1, size // 512)
    for k in range(n_dir):
        for vertical in (False, True):
            t = (k + 0.5) / n_dir * size + rng.uniform(-0.12, 0.12) * size / n_dir
            drift = rng.uniform(-0.08, 0.08) * size
            a, b = (t, -5.0), (t + drift, size + 5.0)
            if not vertical:
                a, b = (a[1], a[0]), (b[1], b[0])
            roads.append((LineString([a, b]), float(rng.uniform(10, 14)), "간선"))
    arterials = [r[0] for r in roads]
    # 2) 국지도로: 서로 다른 간선 위 두 점을 굽은 선으로 잇는다
    n_local = max(3, size // 160)
    for _ in range(n_local * 3):
        if sum(r[2] == "국지" for r in roads) >= n_local:
            break
        i, j = rng.choice(len(arterials), 2, replace=False)
        p0 = arterials[i].interpolate(rng.uniform(0.15, 0.85), normalized=True)
        p2 = arterials[j].interpolate(rng.uniform(0.15, 0.85), normalized=True)
        if not size * 0.2 < p0.distance(p2) < size * 0.7:
            continue
        mid = np.array([(p0.x + p2.x) / 2, (p0.y + p2.y) / 2]) + rng.normal(0, size * 0.1, 2)
        line = _bezier((p0.x, p0.y), tuple(mid), (p2.x, p2.y))
        # 다른 국지도로와 너무 가깝게 나란하면 버린다
        if any(line.buffer(30).intersection(r[0]).length > 120 for r in roads):
            continue
        roads.append((line, float(rng.uniform(5, 8)), "국지"))
    # 3) 막다른 길: 국지도로 위 점에서 수직 방향으로 40~90 m
    locals_ = [r[0] for r in roads if r[2] == "국지"]
    for _ in range(max(2, size // 250)):
        base = locals_[int(rng.integers(len(locals_)))]
        s = rng.uniform(0.25, 0.75)
        p = base.interpolate(s, normalized=True)
        q = base.interpolate(min(1.0, s + 0.02), normalized=True)
        dx, dy = q.x - p.x, q.y - p.y
        norm = np.hypot(dx, dy) or 1.0
        sign = rng.choice([-1, 1])
        length = rng.uniform(80, 180)  # 화소
        end = (p.x - sign * dy / norm * length, p.y + sign * dx / norm * length)
        spur = LineString([(p.x, p.y), end])
        if any(spur.buffer(12).intersection(r[0]).length > 30 for r in roads if r[0] is not base):
            continue
        roads.append((spur, float(rng.uniform(4, 5)), "막다른 길"))
    frame = box(0, 0, size, size)
    out = []
    for line, width, kind in roads:
        clipped = line.intersection(frame)
        if clipped.is_empty or clipped.length < 10:
            continue
        for part in getattr(clipped, "geoms", [clipped]):
            if part.geom_type == "LineString" and part.length >= 10:
                out.append((part, width, kind))
    return out


def generate_road(
    out_dir: str | Path, size: int = 1024, seed: int = 21, name: str = "sample-road"
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h = w = size
    ident = rasterio.Affine.identity()

    # 1) 배경: 식생·나지
    fld = _smooth(rng.normal(size=(h, w)).astype(np.float32), 24)
    fld = (fld - fld.min()) / (np.ptp(fld) + 1e-6)
    veg = np.stack([60 + 30 * fld, 105 + 40 * fld, 50 + 20 * fld])
    soil = np.stack([150 + 20 * fld, 130 + 15 * fld, 100 + 10 * fld])
    mix = (_smooth(rng.normal(size=(h, w)).astype(np.float32), 40) > 0.02)[None]
    img = np.where(mix, veg, soil).astype(np.float32)

    roads = road_lines(size, rng)
    surface = unary_union([ln.buffer(wd / GSD / 2, cap_style="flat") for ln, wd, _ in roads])
    road_mask = rasterize([surface], out_shape=(h, w), transform=ident).astype(bool)

    # 2) 포장 주차장(도로와 같은 색, 도로 아님): 도로에서 떨어진 직사각형
    lots = []
    for _ in range(max(2, size // 300)):
        lw, lh = rng.uniform(40, 80), rng.uniform(30, 60)
        cx, cy = rng.uniform(lw, w - lw), rng.uniform(lh, h - lh)
        lot = box(cx - lw / 2, cy - lh / 2, cx + lw / 2, cy + lh / 2)
        if lot.buffer(8).intersects(surface) or any(lot.buffer(8).intersects(o) for o in lots):
            continue
        lots.append(lot)
    if lots:
        lot_mask = rasterize(lots, out_shape=(h, w), transform=ident).astype(bool)
        img[:, lot_mask] = (ASPHALT + 6)[:, None]

    # 3) 노면: 아스팔트 + 간선 중앙선(파선)
    tone = 1.0 + 0.04 * (fld - 0.5)
    img[:, road_mask] = (ASPHALT[:, None, None] * tone[None])[:, road_mask]
    dashes = []
    for ln, _, kind in roads:
        if kind != "간선":
            continue
        d = 0.0
        while d < ln.length:
            seg = [ln.interpolate(d), ln.interpolate(min(ln.length, d + 6))]
            dashes.append(LineString([(p.x, p.y) for p in seg]).buffer(0.6))
            d += 14
    if dashes:
        mark = rasterize(dashes, out_shape=(h, w), transform=ident).astype(bool) & road_mask
        img[:, mark] = MARKING[:, None]

    # 4) 도로변 건물(그림자 포함). 도로·주차장과 겹치지 않는다
    buildings = []
    for _ in range(size * size // 9000):
        bw, bh = rng.uniform(14, 40), rng.uniform(14, 36)
        cx, cy = rng.uniform(bw, w - bw), rng.uniform(bh, h - bh)
        poly = affinity.rotate(box(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), float(rng.uniform(-30, 30)))
        if poly.buffer(6).intersects(surface) or any(poly.buffer(6).intersects(o) for o in lots + buildings):
            continue
        buildings.append(poly)
    if buildings:
        shadow = rasterize([affinity.translate(p, 5, 5) for p in buildings], out_shape=(h, w), transform=ident)
        img[:, shadow.astype(bool)] *= 0.5
        shapes = [(p, i + 1) for i, p in enumerate(buildings)]
        index = rasterize(shapes, out_shape=(h, w), transform=ident, dtype="int32")
        roof_colors = ROOF_COLORS[rng.integers(len(ROOF_COLORS), size=len(buildings))]
        colors = np.vstack([np.zeros((1, 3), np.float32), roof_colors])
        roof = index > 0
        for b in range(3):
            img[b][roof] = colors[index[roof], b]

    # 5) 가로수: 도로 가장자리에 줄지어 심어 노면 일부를 가린다(중심선은 이어진다)
    trees = []
    for ln, wd, _ in roads:
        half = wd / GSD / 2
        d = rng.uniform(0, 30)
        while d < ln.length:
            if rng.uniform() < 0.35:
                p = ln.interpolate(d)
                q = ln.interpolate(min(ln.length, d + 1))
                dx, dy = q.x - p.x, q.y - p.y
                n = np.hypot(dx, dy) or 1.0
                side = rng.choice([-1, 1])
                c = Point(p.x - side * dy / n * half, p.y + side * dx / n * half)
                trees.append(c.buffer(rng.uniform(4, 8)))
            d += rng.uniform(18, 40)
    for _ in range(size * size // 20000):
        trees.append(Point(rng.uniform(0, w), rng.uniform(0, h)).buffer(rng.uniform(3, 9)))
    if trees:
        tidx = rasterize([(t, i + 1) for i, t in enumerate(trees)], out_shape=(h, w), transform=ident, dtype="int32")
        tone_t = np.concatenate([[1.0], rng.uniform(0.8, 1.1, len(trees))])[tidx]
        tmask = tidx > 0
        for b, base in enumerate((45, 85, 40)):
            img[b][tmask] = base * tone_t[tmask]

    img = np.stack([_smooth(b, 1) for b in img]) + rng.normal(0, 4, size=img.shape)
    data = np.clip(img, 0, 255).astype(np.uint8)
    transform = from_origin(ORIGIN[0], ORIGIN[1], GSD, GSD)
    tif = out / f"{name}.tif"
    with rasterio.open(
        tif, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=CRS, transform=transform,
        tiled=True, blockxsize=256, blockysize=256, compress="deflate", photometric="RGB",
    ) as dst:  # fmt: skip
        dst.write(data)

    def to_world(geom):
        return affinity.affine_transform(geom, [GSD, 0, 0, -GSD, ORIGIN[0], ORIGIN[1]])

    features = [
        {
            "type": "Feature",
            "properties": {"id": i, "class": "Road", "width_m": round(wd, 1), "road_type": kind},
            "geometry": mapping(to_world(ln)),
        }
        for i, (ln, wd, kind) in enumerate(roads)
    ]
    geojson = out / f"{name}-centerlines.geojson"
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
