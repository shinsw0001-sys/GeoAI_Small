"""타일 분할·품질 검사 (job:TilingJob, 진입점 tiling).

입력: 정사영상(gai:OrthoImage) 1개 이상, 정규화 수치 표면 모델(gai:DigitalSurfaceModel, 선택)
      - 정사영상 1개: 단시기 분야(건물 추출 등). 3밴드(red, green, blue).
      - 정사영상 여럿(시점별 acquired_at): 다시기 분야(변화탐지). 같은 격자여야 하며, 시점 순으로
        영상 3밴드씩, 이어서 시점별 지물 높이 1밴드씩 쌓는다(조기 융합 입력).
처리: 1) 다시기면 상대 방사 정규화: 이후 시점 영상의 밴드별 평균·표준편차를 첫 시점에 맞춘다(촬영 조건 차이 보정).
      2) 지물 높이는 0.5 m 단위 8비트(0~127.5 m)로 바꾼다.
      3) 고정 크기 타일로 자르고, 무자료 비율(모든 시점 영상 기준)이 기준을 넘는 타일을 버린다.
      4) 인접 타일이 학습과 시험에 함께 들어가 성능이 부풀지 않도록 공간 블록(block_tiles × block_tiles 타일)
         단위로 학습·검증·시험 집합을 나눈다. 밴드별 평균·표준편차(유효 화소)를 정규화 통계로 남긴다.
산출: 분석 준비 데이터(gai:AnalysisReadyData) = manifest.json + tiles.npz
매개변수: tile_size(256), max_nodata_ratio(0.2), block_tiles(2), split(0.7/0.15/0.15), seed(42),
          radiometric_normalization(true), height_step_m(0.5)
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
RGB = ("red", "green", "blue")


def assign_splits(blocks: list[tuple[int, int]], ratios: list[float], seed: int) -> dict[tuple[int, int], str]:
    order = np.random.default_rng(seed).permutation(len(blocks))
    n = len(blocks)
    n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
    n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
    out = {}
    for rank, idx in enumerate(order):
        out[blocks[idx]] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    return out


def _date(asset: DataAssetRef) -> str:
    return asset.acquired_at.isoformat() if asset.acquired_at else ""


def _match_stats(src: np.ndarray, valid: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray) -> np.ndarray:
    """밴드별 평균·표준편차 맞춤(상대 방사 정규화). src (3, H, W) uint8."""
    out = src.astype(np.float32)
    for b in range(src.shape[0]):
        v = out[b][valid]
        m, s = v.mean(), v.std() + 1e-6
        out[b] = (out[b] - m) / s * ref_std[b] + ref_mean[b]
    out[:, ~valid] = 0
    return np.clip(out, 1, 255).astype(np.uint8) * valid[None]


def run(ctx: ToolContext) -> ToolOutput:
    rasters = [a for a in ctx.spec.inputs if a.storage_uri.lower().endswith((".tif", ".tiff"))]
    orthos = sorted([a for a in rasters if not str(a.asset_type).endswith("DigitalSurfaceModel")], key=_date)
    heights = sorted([a for a in rasters if str(a.asset_type).endswith("DigitalSurfaceModel")], key=_date)
    if not orthos:
        raise ToolError("no_raster_input", "정사영상 입력이 없습니다.")
    multi = len(orthos) > 1 or bool(heights)
    size = ctx.param("tile_size", 256)
    max_nodata = ctx.param("max_nodata_ratio", 0.2)
    block = ctx.param("block_tiles", 2)
    ratios = [float(x) for x in ctx.spec.parameters.get("split", [0.7, 0.15, 0.15])]
    seed = ctx.param("seed", 42)
    step = ctx.param("height_step_m", 0.5)

    ctx.progress(5, f"입력 내려받기 (영상 {len(orthos)}, 높이 {len(heights)})")
    paths = [ctx.fetch(a.storage_uri, f"in-{i}.tif") for i, a in enumerate(orthos + heights)]
    datasets = [rasterio.open(p) for p in paths]
    try:
        ref = datasets[0]
        for a, ds in zip(orthos + heights, datasets, strict=True):
            if (ds.width, ds.height) != (ref.width, ref.height) or not ds.transform.almost_equals(ref.transform):
                raise ToolError("grid_mismatch", f"입력 격자가 첫 정사영상과 다릅니다: {a.storage_uri}")
        rows, cols = ref.height // size, ref.width // size
        if rows * cols == 0:
            raise ToolError("image_too_small", f"영상({ref.width}×{ref.height})이 타일 크기 {size}보다 작습니다.")

        # 밴드 정의와 전체 배열(다시기는 방사 정규화를 위해 전체를 읽는다)
        band_info: list[dict] = []
        stacks: list[np.ndarray] = []
        normalization_log = []
        if multi:
            images = [ds.read([1, 2, 3]) for ds in datasets[: len(orthos)]]
            valid_all = np.all([np.any(im > 0, axis=0) for im in images], axis=0)
            ref_px = images[0][:, valid_all].astype(np.float32)
            ref_mean, ref_std = ref_px.mean(axis=1), ref_px.std(axis=1) + 1e-6
            do_norm = bool(ctx.spec.parameters.get("radiometric_normalization", True))
            for k, (a, im) in enumerate(zip(orthos, images, strict=True)):
                if k > 0 and do_norm:
                    before = im[:, valid_all].astype(np.float32).mean(axis=1)
                    im = _match_stats(im, valid_all, ref_mean, ref_std)
                    normalization_log.append({"acquired_at": _date(a), "mean_before": before.round(2).tolist()})
                stacks.append(im)
                band_info += [
                    {
                        "name": f"{_date(a)}:{c}",
                        "kind": c,
                        "acquired_at": _date(a),
                        "source": a.storage_uri,
                        "scale": 1.0,
                    }
                    for c in RGB
                ]
            for a, ds in zip(heights, datasets[len(orthos) :], strict=True):
                h = ds.read(1).astype(np.float32)
                stacks.append(np.clip(np.round(h / step), 0, 255).astype(np.uint8)[None])
                band_info.append(
                    {"name": f"{_date(a)}:height", "kind": "height", "acquired_at": _date(a), "source": a.storage_uri,
                     "scale": step, "unit": "m"}
                )  # fmt: skip
            full = np.concatenate(stacks)
        else:
            band_info = [
                {"name": c, "kind": c, "acquired_at": _date(orthos[0]), "source": orthos[0].storage_uri, "scale": 1.0}
                for c in RGB
            ]
            full = None
            valid_all = None
        n_img = len(orthos)

        tiles, valids, index = [], [], []
        dropped = 0
        total = rows * cols
        for r in range(rows):
            for c in range(cols):
                window = Window(c * size, r * size, size, size)
                if full is None:
                    tile = ref.read([1, 2, 3], window=window)
                    valid = np.any(tile > 0, axis=0)
                else:
                    sl = (slice(r * size, (r + 1) * size), slice(c * size, (c + 1) * size))
                    tile = full[:, sl[0], sl[1]]
                    valid = valid_all[sl]
                if 1.0 - valid.mean() > max_nodata:
                    dropped += 1
                    continue
                t = ref.window_transform(window)
                tiles.append(tile)
                valids.append(valid)
                index.append({"row": r, "col": c, "transform": [t.a, t.b, t.c, t.d, t.e, t.f]})
                ctx.progress(10 + 70 * (r * cols + c + 1) / total, f"타일 분할 {r * cols + c + 1}/{total}")
        crs = ref.crs.to_string()
        gsd = abs(ref.res[0])
    finally:
        for ds in datasets:
            ds.close()
    if len(tiles) < 3:
        raise ToolError("too_few_tiles", f"유효 타일이 {len(tiles)}개뿐입니다. 학습·검증·시험에 3개 이상 필요합니다.")

    blocks = sorted({(t["row"] // block, t["col"] // block) for t in index})
    split_of = assign_splits(blocks, ratios, seed)
    for i, t in enumerate(index):
        t["id"] = i
        t["split"] = split_of[(t["row"] // block, t["col"] // block)]

    img = np.stack(tiles)
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
        "source": orthos[0].storage_uri,
        "sources": [a.storage_uri for a in orthos + heights],
        "acquisitions": sorted({_date(a) for a in orthos if a.acquired_at}),
        "crs": crs,
        "ground_sample_distance": gsd,
        "tile_size": size,
        "bands": [b["name"] for b in band_info],
        "band_info": band_info,
        # 미리보기용 밴드: 마지막 시점 영상, 비교용으로 첫 시점 영상
        "display_bands": [3 * (n_img - 1) + i for i in range(3)],
        "reference_display_bands": [0, 1, 2] if n_img > 1 else None,
        "normalization": {"mean": mean, "std": std, "scale": 255.0},
        "radiometric_normalization": normalization_log,
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
                crs=orthos[0].crs,
                ground_sample_distance=orthos[0].ground_sample_distance,
            )
        ],
        metrics={
            "tile_count": float(len(index)),
            "dropped_tiles": float(dropped),
            "train_tiles": float(counts["train"]),
            "val_tiles": float(counts["val"]),
            "test_tiles": float(counts["test"]),
            "bands": float(len(band_info)),
        },
    )
