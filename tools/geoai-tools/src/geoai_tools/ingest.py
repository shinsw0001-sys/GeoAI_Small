"""정사영상 수집·정합 (job:OrthoIngestJob, 진입점 ortho-ingest).

입력: 원본 영상(항공·위성·드론) GeoTIFF 1개 이상
검사: 좌표 참조 체계 존재, 3밴드 이상, 8비트 또는 변환 가능한 형식
처리: 목표 좌표 참조 체계로 재투영하고 여러 장이면 병합한다. 결과는 타일 구조·압축 GeoTIFF.
산출: 정사영상(gai:OrthoImage) 1개
매개변수: target_crs(기본: 첫 입력의 좌표 참조 체계), target_gsd(기본: 가장 고운 입력 해상도)
"""

from __future__ import annotations

import numpy as np
import rasterio
from geoai_common.messages import DataAssetRef
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, calculate_default_transform

from .runtime import ToolContext, ToolError, ToolOutput

GAI = "https://ontology.geoai.local/core#"


def _to_uint8(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        return data
    out = np.zeros(data.shape, dtype=np.uint8)
    for b in range(data.shape[0]):
        band = data[b].astype(np.float32)
        lo, hi = np.percentile(band[valid], [2, 98]) if valid.any() else (0.0, 1.0)
        out[b] = np.clip((band - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return out


def run(ctx: ToolContext) -> ToolOutput:
    rasters = [a for a in ctx.spec.inputs if a.storage_uri.lower().endswith((".tif", ".tiff"))]
    if not rasters:
        raise ToolError("no_raster_input", "GeoTIFF 입력이 없습니다.")
    paths = []
    for i, asset in enumerate(rasters):
        ctx.progress(5 + 30 * i / len(rasters), f"원본 내려받기 {i + 1}/{len(rasters)}")
        paths.append(ctx.fetch(asset.storage_uri, f"raw-{i}.tif"))

    datasets = [rasterio.open(p) for p in paths]
    try:
        for asset, ds in zip(rasters, datasets, strict=True):
            if ds.crs is None:
                raise ToolError("missing_crs", f"좌표 참조 체계가 없는 영상: {asset.storage_uri}")
            if ds.count < 3:
                raise ToolError("band_mismatch", f"3밴드(RGB) 이상이어야 합니다: {asset.storage_uri} ({ds.count}밴드)")
        target_crs = CRS.from_user_input(ctx.spec.parameters.get("target_crs") or datasets[0].crs)
        target_gsd = float(ctx.spec.parameters.get("target_gsd") or 0) or None

        ctx.progress(40, "좌표 참조 체계 정합·재투영")
        vrts = []
        for ds in datasets:
            if ds.crs == target_crs and target_gsd is None:
                vrts.append(ds)
                continue
            transform, width, height = calculate_default_transform(
                ds.crs, target_crs, ds.width, ds.height, *ds.bounds, resolution=target_gsd
            )
            vrts.append(
                WarpedVRT(
                    ds, crs=target_crs, transform=transform, width=width, height=height, resampling=Resampling.bilinear
                )
            )
        res = target_gsd or min(min(abs(v.res[0]), abs(v.res[1])) for v in vrts)
        ctx.progress(60, f"병합 ({len(vrts)}장)")
        mosaic, transform = merge(vrts, res=res, indexes=[1, 2, 3], nodata=0)
        valid = np.any(mosaic > 0, axis=0)
        mosaic = _to_uint8(mosaic, valid)
    finally:
        for ds in datasets:
            ds.close()

    nodata_ratio = float(1.0 - valid.mean())
    if nodata_ratio > 0.9:
        raise ToolError("mostly_nodata", f"유효 화소가 10% 미만입니다 (무자료 {nodata_ratio:.0%}).")
    out_path = ctx.workdir / "ortho.tif"
    _, height, width = mosaic.shape
    with rasterio.open(
        out_path, "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8", crs=target_crs,
        transform=transform, nodata=0, tiled=True, blockxsize=256, blockysize=256, compress="deflate",
        photometric="RGB",
    ) as dst:  # fmt: skip
        dst.write(mosaic)
    ctx.progress(90, "정사영상 등록")
    uri = ctx.out("ortho.tif")
    ctx.store.put_file(uri, out_path)
    epsg = target_crs.to_epsg()
    crs_text = f"EPSG:{epsg}" if epsg else target_crs.to_string()
    area = width * height * res * res / 1e6
    return ToolOutput(
        outputs=[
            DataAssetRef(
                asset_id=f"{ctx.spec.task_id}.ortho",
                asset_type=GAI + "OrthoImage",
                storage_uri=uri,
                crs=crs_text,
                ground_sample_distance=round(res, 4),
            )
        ],
        metrics={
            "input_count": float(len(rasters)),
            "width_px": float(width),
            "height_px": float(height),
            "area_km2": round(area, 4),
            "nodata_ratio": round(nodata_ratio, 4),
        },
    )
