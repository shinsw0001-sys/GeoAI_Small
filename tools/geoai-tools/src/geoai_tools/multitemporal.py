"""다시기 영상·라이다 정합 (job:MultiTemporalIngestJob, 진입점 multitemporal-ingest).

입력: 시점별 원본 영상(항공·위성·드론) GeoTIFF 1개 이상 + 라이다 점군 LAS(선택). 모든 입력에 취득 일자
      (acquired_at)가 있어야 하고, 서로 다른 일자가 2개 이상이어야 한다.
검사: 좌표 참조 체계 존재, 3밴드 이상, 모든 시점 영상이 겹치는 범위가 있는지
처리: 1) 공통 격자: 목표 좌표 참조 체계·해상도에서 모든 시점 영상 범위의 교집합을 해상도 배수로 맞춘다.
      2) 시점마다 영상을 공통 격자로 재투영·병합(쌍선형)해 정사영상 1개를 만든다.
      3) 라이다가 있는 시점은 점군을 공통 격자로 격자화해 정규화 수치 표면 모델(지물 높이, m)을 만든다.
      모든 산출물이 같은 격자이므로 전처리 단계에서 화소 단위로 바로 쌓을 수 있다.
산출: 시점별 정사영상(gai:OrthoImage), 시점별 정규화 수치 표면 모델(gai:DigitalSurfaceModel).
      모두 acquired_at 을 갖는다.
매개변수: target_crs(기본: 첫 영상), target_gsd(기본: 가장 고운 영상 해상도), ground_cell_m(지면 추정 격자, 8),
          min_overlap_ratio(교집합 ÷ 합집합 하한, 0.5)
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import rasterio
from geoai_common.messages import DataAssetRef
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.warp import transform as warp_points

from .ingest import _to_uint8
from .lidar import normalized_surface, read_points
from .runtime import ToolContext, ToolError, ToolOutput

GAI = "https://ontology.geoai.local/core#"
RASTER_EXT = (".tif", ".tiff")
POINT_EXT = (".las",)


def _grid(bounds: list[tuple[float, float, float, float]], gsd: float) -> tuple[Affine, int, int, float]:
    left = max(b[0] for b in bounds)
    bottom = max(b[1] for b in bounds)
    right = min(b[2] for b in bounds)
    top = min(b[3] for b in bounds)
    if right <= left or top <= bottom:
        raise ToolError("no_temporal_overlap", "시점별 영상이 서로 겹치지 않습니다.")
    left, top = math.ceil(left / gsd) * gsd, math.floor(top / gsd) * gsd
    width, height = int((right - left) // gsd), int((top - bottom) // gsd)
    union = (max(b[2] for b in bounds) - min(b[0] for b in bounds)) * (
        max(b[3] for b in bounds) - min(b[1] for b in bounds)
    )
    return Affine(gsd, 0, left, 0, -gsd, top), width, height, (width * height * gsd * gsd) / max(union, 1e-9)


def run(ctx: ToolContext) -> ToolOutput:
    rasters = [a for a in ctx.spec.inputs if a.storage_uri.lower().endswith(RASTER_EXT)]
    clouds = [a for a in ctx.spec.inputs if a.storage_uri.lower().endswith(POINT_EXT)]
    if not rasters:
        raise ToolError("no_raster_input", "GeoTIFF 영상 입력이 없습니다.")
    undated = [a.storage_uri for a in rasters + clouds if a.acquired_at is None]
    if undated:
        raise ToolError("missing_acquisition_date", f"취득 일자(acquired_at)가 없는 입력: {', '.join(undated)}")
    by_date: dict[str, list] = defaultdict(list)
    for a in rasters:
        by_date[a.acquired_at.isoformat()].append(a)
    if len(by_date) < 2:
        raise ToolError("insufficient_acquisitions", f"영상 취득 일자가 {len(by_date)}개입니다. 2시점 이상 필요합니다.")
    dates = sorted(by_date)
    lidar_by_date: dict[str, list] = defaultdict(list)
    for a in clouds:
        lidar_by_date[a.acquired_at.isoformat()].append(a)
    orphan = sorted(set(lidar_by_date) - set(dates))
    if orphan:
        raise ToolError("lidar_without_image", f"같은 일자의 영상이 없는 라이다 시점: {', '.join(orphan)}")

    ctx.progress(3, f"원본 내려받기 ({len(rasters)}개 영상, {len(clouds)}개 점군)")
    paths = {a.storage_uri: ctx.fetch(a.storage_uri, f"raw-{i}{a.storage_uri[-4:]}") for i, a in enumerate(rasters)}
    datasets = {uri: rasterio.open(p) for uri, p in paths.items()}
    try:
        for a in rasters:
            ds = datasets[a.storage_uri]
            if ds.crs is None:
                raise ToolError("missing_crs", f"좌표 참조 체계가 없는 영상: {a.storage_uri}")
            if ds.count < 3:
                raise ToolError("band_mismatch", f"3밴드(RGB) 이상이어야 합니다: {a.storage_uri} ({ds.count}밴드)")
        first = datasets[by_date[dates[0]][0].storage_uri]
        target_crs = CRS.from_user_input(ctx.spec.parameters.get("target_crs") or first.crs)
        gsd = float(ctx.spec.parameters.get("target_gsd") or 0) or min(
            min(abs(ds.res[0]), abs(ds.res[1])) for ds in datasets.values()
        )
        # 시점마다 그 시점 영상들의 합집합 범위, 시점 간에는 교집합
        date_bounds = []
        for d in dates:
            bs = [
                transform_bounds(datasets[a.storage_uri].crs, target_crs, *datasets[a.storage_uri].bounds)
                for a in by_date[d]
            ]
            date_bounds.append(
                (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
            )
        transform, width, height, overlap = _grid(date_bounds, gsd)
        min_overlap = ctx.param("min_overlap_ratio", 0.5)
        if overlap < min_overlap:
            raise ToolError(
                "low_temporal_overlap", f"시점 간 겹침 비율 {overlap:.0%} 이 기준 {min_overlap:.0%} 보다 낮습니다."
            )
        if width * height == 0:
            raise ToolError("no_temporal_overlap", "공통 격자가 비었습니다.")

        epsg = target_crs.to_epsg()
        crs_text = f"EPSG:{epsg}" if epsg else target_crs.to_string()
        outputs: list[DataAssetRef] = []
        metrics: dict[str, float] = {
            "acquisitions": float(len(dates)),
            "width_px": float(width),
            "height_px": float(height),
            "area_km2": round(width * height * gsd * gsd / 1e6, 4),
            "temporal_overlap_ratio": round(overlap, 4),
        }
        profile = dict(
            driver="GTiff", width=width, height=height, crs=target_crs, transform=transform,
            tiled=True, blockxsize=256, blockysize=256, compress="deflate",
        )  # fmt: skip
        n_steps = len(dates) * 2
        for k, d in enumerate(dates):
            ctx.progress(10 + 80 * (2 * k) / n_steps, f"{d} 영상 공통 격자 정합")
            mosaic = np.zeros((3, height, width), dtype=np.float32)
            filled = np.zeros((height, width), dtype=bool)
            for a in by_date[d]:
                ds = datasets[a.storage_uri]
                dst = np.zeros((3, height, width), dtype=np.float32)
                reproject(
                    source=ds.read([1, 2, 3]).astype(np.float32), destination=dst, src_transform=ds.transform,
                    src_crs=ds.crs, src_nodata=0, dst_transform=transform, dst_crs=target_crs, dst_nodata=0,
                    resampling=Resampling.bilinear,
                )  # fmt: skip
                new = np.any(dst > 0, axis=0) & ~filled
                mosaic[:, new] = dst[:, new]
                filled |= new
            eight_bit = all(datasets[a.storage_uri].dtypes[0] == "uint8" for a in by_date[d])
            image = np.clip(mosaic, 0, 255).astype(np.uint8) if eight_bit else _to_uint8(mosaic, filled)
            nodata = float(1.0 - filled.mean())
            if nodata > 0.9:
                raise ToolError("mostly_nodata", f"{d} 영상의 유효 화소가 10% 미만입니다.")
            tag = d.replace("-", "")
            out = ctx.workdir / f"ortho-{tag}.tif"
            with rasterio.open(out, "w", count=3, dtype="uint8", nodata=0, photometric="RGB", **profile) as dst_ds:
                dst_ds.write(image)
            uri = ctx.out(f"ortho-{tag}.tif")
            ctx.store.put_file(uri, out)
            outputs.append(
                DataAssetRef(
                    asset_id=f"{ctx.spec.task_id}.ortho-{tag}", asset_type=GAI + "OrthoImage", storage_uri=uri,
                    crs=crs_text, ground_sample_distance=round(gsd, 4), acquired_at=d,
                )
            )  # fmt: skip
            metrics[f"nodata_ratio_{tag}"] = round(nodata, 4)

            if not lidar_by_date.get(d):
                continue
            ctx.progress(10 + 80 * (2 * k + 1) / n_steps, f"{d} 라이다 격자화")
            ndsm = np.zeros((height, width), dtype=np.float32)
            points = 0.0
            coverage = []
            for i, a in enumerate(lidar_by_date[d]):
                cloud = read_points(ctx.fetch(a.storage_uri, f"lidar-{tag}-{i}.las"))
                src = CRS.from_wkt(cloud.crs_wkt) if cloud.crs_wkt else (CRS.from_user_input(a.crs) if a.crs else None)
                if src is None:
                    raise ToolError("missing_crs", f"좌표 참조 체계가 없는 점군: {a.storage_uri}")
                if src != target_crs:
                    xs, ys = warp_points(src, target_crs, cloud.xyz[:, 0], cloud.xyz[:, 1])
                    cloud.xyz[:, 0], cloud.xyz[:, 1] = xs, ys
                part, q = normalized_surface(cloud, transform, (height, width), ctx.param("ground_cell_m", 8.0))
                ndsm = np.maximum(ndsm, part)
                points += q["points"]
                coverage.append(q["cell_coverage"])
            if points == 0:
                raise ToolError("lidar_no_overlap", f"{d} 라이다 점군이 공통 격자와 겹치지 않습니다.")
            out = ctx.workdir / f"ndsm-{tag}.tif"
            with rasterio.open(out, "w", count=1, dtype="float32", **profile) as dst_ds:
                dst_ds.write(ndsm[None])
            uri = ctx.out(f"ndsm-{tag}.tif")
            ctx.store.put_file(uri, out)
            outputs.append(
                DataAssetRef(
                    asset_id=f"{ctx.spec.task_id}.ndsm-{tag}", asset_type=GAI + "DigitalSurfaceModel", storage_uri=uri,
                    crs=crs_text, ground_sample_distance=round(gsd, 4), acquired_at=d,
                )
            )  # fmt: skip
            metrics[f"lidar_points_{tag}"] = points
            metrics[f"lidar_cell_coverage_{tag}"] = round(float(np.mean(coverage)), 4)
            metrics[f"max_height_m_{tag}"] = round(float(ndsm.max()), 2)
    finally:
        for ds in datasets.values():
            ds.close()
    ctx.progress(95, "산출물 등록")
    return ToolOutput(outputs=outputs, metrics=metrics)
