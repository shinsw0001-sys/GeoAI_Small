"""타일 분할·품질 검사 (job:TilingJob, 진입점 tiling).

입력: 정사영상(gai:OrthoImage)
처리: 고정 크기 타일로 자르고, 무자료 비율이 기준을 넘는 타일을 버린다. 인접 타일이 학습과
      시험에 함께 들어가 성능이 부풀지 않도록 공간 블록(block_tiles × block_tiles 타일) 단위로
      학습·검증·시험 집합을 나눈다. 밴드별 평균·표준편차(유효 화소)를 정규화 통계로 남긴다.
산출: 분석 준비 데이터(gai:AnalysisReadyData) = manifest.json + tiles.npz
매개변수: tile_size(256), max_nodata_ratio(0.2), block_tiles(2), split(학습·검증·시험 비율, 0.7/0.15/0.15), seed(42)
"""

from __future__ import annotations

import io

import numpy as np
import rasterio
from geoai_common.messages import DataAssetRef
from geoai_common.objectstore import put_json
from rasterio.windows import Window

from .runtime import ToolContext, ToolError, ToolOutput

GAI = "https://ontology.geoai.local/core#"
SPLITS = ("train", "val", "test")


def assign_splits(blocks: list[tuple[int, int]], ratios: list[float], seed: int) -> dict[tuple[int, int], str]:
    order = np.random.default_rng(seed).permutation(len(blocks))
    n = len(blocks)
    n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
    n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
    out = {}
    for rank, idx in enumerate(order):
        out[blocks[idx]] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    return out


def run(ctx: ToolContext) -> ToolOutput:
    ortho = next((a for a in ctx.spec.inputs if a.storage_uri.lower().endswith((".tif", ".tiff"))), None)
    if ortho is None:
        raise ToolError("no_raster_input", "정사영상 입력이 없습니다.")
    size = ctx.param("tile_size", 256)
    max_nodata = ctx.param("max_nodata_ratio", 0.2)
    block = ctx.param("block_tiles", 2)
    ratios = [float(x) for x in ctx.spec.parameters.get("split", [0.7, 0.15, 0.15])]
    seed = ctx.param("seed", 42)

    ctx.progress(5, "정사영상 내려받기")
    path = ctx.fetch(ortho.storage_uri, "ortho.tif")
    with rasterio.open(path) as ds:
        rows, cols = ds.height // size, ds.width // size
        if rows * cols == 0:
            raise ToolError("image_too_small", f"영상({ds.width}×{ds.height})이 타일 크기 {size}보다 작습니다.")
        images, valids, index = [], [], []
        dropped = 0
        total = rows * cols
        for r in range(rows):
            for c in range(cols):
                window = Window(c * size, r * size, size, size)
                tile = ds.read([1, 2, 3], window=window)
                valid = np.any(tile > 0, axis=0)
                if 1.0 - valid.mean() > max_nodata:
                    dropped += 1
                    continue
                t = ds.window_transform(window)
                images.append(tile)
                valids.append(valid)
                index.append({"row": r, "col": c, "transform": [t.a, t.b, t.c, t.d, t.e, t.f]})
                ctx.progress(10 + 70 * (r * cols + c + 1) / total, f"타일 분할 {r * cols + c + 1}/{total}")
        crs = ds.crs.to_string()
        gsd = abs(ds.res[0])
    if len(images) < 3:
        raise ToolError("too_few_tiles", f"유효 타일이 {len(images)}개뿐입니다. 학습·검증·시험에 3개 이상 필요합니다.")

    blocks = sorted({(t["row"] // block, t["col"] // block) for t in index})
    split_of = assign_splits(blocks, ratios, seed)
    for i, t in enumerate(index):
        t["id"] = i
        t["split"] = split_of[(t["row"] // block, t["col"] // block)]

    img = np.stack(images)
    val = np.stack(valids)
    ctx.progress(85, "정규화 통계 계산")
    pixels = img.transpose(1, 0, 2, 3)[:, val]  # (밴드, 유효 화소)
    mean = (pixels.mean(axis=1) / 255.0).round(5).tolist()
    std = (pixels.std(axis=1) / 255.0 + 1e-6).round(5).tolist()

    buf = io.BytesIO()
    np.savez_compressed(buf, images=img, valid=val)
    tiles_uri = ctx.out("ard", "tiles.npz")
    ctx.store.put_bytes(tiles_uri, buf.getvalue())
    counts = {s: sum(t["split"] == s for t in index) for s in SPLITS}
    manifest = {
        "kind": "analysis-ready-tiles",
        "source": ortho.storage_uri,
        "crs": crs,
        "ground_sample_distance": gsd,
        "tile_size": size,
        "bands": ["red", "green", "blue"],
        "normalization": {"mean": mean, "std": std, "scale": 255.0},
        "tiles_uri": tiles_uri,
        "tiles": index,
        "split_counts": counts,
        "quality": {"total_tiles": total, "dropped_nodata": dropped, "max_nodata_ratio": max_nodata},
    }
    manifest_uri = ctx.out("ard", "manifest.json")
    put_json(ctx.store, manifest_uri, manifest)
    return ToolOutput(
        outputs=[
            DataAssetRef(
                asset_id=f"{ctx.spec.task_id}.ard",
                asset_type=GAI + "AnalysisReadyData",
                storage_uri=manifest_uri,
                crs=ortho.crs,
                ground_sample_distance=ortho.ground_sample_distance,
            )
        ],
        metrics={
            "tile_count": float(len(index)),
            "dropped_tiles": float(dropped),
            "train_tiles": float(counts["train"]),
            "val_tiles": float(counts["val"]),
            "test_tiles": float(counts["test"]),
        },
    )
