"""변화탐지 처리 도구 연쇄: 다시기 영상·라이다 정합 → 다채널 타일 분할 → 변화 라벨 도출 → 학습 → 성능 검증."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from geoai_common.jobs import JobSpec  # noqa: E402
from geoai_common.objectstore import LocalObjectStore, get_json  # noqa: E402
from geoai_tools import ENTRYPOINTS  # noqa: E402
from geoai_tools.__main__ import seed_change_sample  # noqa: E402
from geoai_tools.change_labels import derive_changes  # noqa: E402
from geoai_tools.model import UNet, transfer_weights  # noqa: E402
from geoai_tools.runtime import run_job  # noqa: E402
from shapely.geometry import box  # noqa: E402

GAI = "https://ontology.geoai.local/core#"
CD = "https://ontology.geoai.local/domain/change-detection#"
CLASSES = [
    {"iri": CD + n, "label": lbl, "value": v}
    for v, (n, lbl) in enumerate(
        [
            ("DemolishedBuilding", "건물 멸실"),
            ("LandChange", "토지 변화"),
            ("ModifiedBuilding", "건물 변경"),
            ("NewBuilding", "건물 신축"),
        ],
        start=1,
    )
]


def spec(name, entry, inputs, **params):
    return JobSpec(
        job_id=f"c-{name}",
        job_type=f"https://ontology.geoai.local/jobs#{name}",
        entrypoint=entry,
        task_id=f"task.{name}",
        task_type=GAI + "X",
        project_id="p",
        domain=CD + "Domain",
        label_classes=CLASSES,
        inputs=inputs,
        parameters=params,
        completion_criteria=[{"metric": "foreground_F1", "operator": ">=", "threshold": 0.1}],
        output_prefix=f"s3://work/p/{name}",
        progress_uri=f"s3://jobs/c-{name}/progress.json",
        result_uri=f"s3://jobs/c-{name}/result.json",
    )


def raw_inputs(meta):
    out = []
    for a in meta["acquisitions"]:
        out.append(
            {"asset_id": a["image"], "asset_type": GAI + "AerialImage", "storage_uri": a["image"], "crs": "EPSG:5186",
             "acquired_at": a["acquired_at"]}
        )  # fmt: skip
        out.append(
            {"asset_id": a["lidar"], "asset_type": GAI + "LidarPointCloud", "storage_uri": a["lidar"],
             "crs": "EPSG:5186", "acquired_at": a["acquired_at"]}
        )  # fmt: skip
    return out


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = LocalObjectStore(tmp_path_factory.mktemp("store"))
    s.meta = seed_change_sample(s, size=512)
    return s


@pytest.fixture(scope="module")
def chain(store):
    out = {}
    r = run_job(spec("ingest", "multitemporal-ingest", raw_inputs(store.meta)), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["ingest"] = r
    r = run_job(spec("tiling", "tiling", r.outputs, tile_size=128, block_tiles=1), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["ard"] = r
    sources = [
        {"acquired_at": a["acquired_at"], "buildings": a["buildings"], "landcover": a["landcover"]}
        for a in store.meta["acquisitions"]
    ]
    r = run_job(spec("labels", "change-labels", r.outputs, label_sources=sources), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["labels"] = r
    r = run_job(spec("train", "segmentation-train", out["ard"].outputs + r.outputs, epochs=3), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["model"] = r
    r = run_job(spec("eval", "segmentation-evaluate", r.outputs + out["labels"].outputs), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["eval"] = r
    return out


def test_ingest_common_grid(chain):
    outs = chain["ingest"].outputs
    kinds = sorted((str(o.asset_type).rsplit("#", 1)[-1], str(o.acquired_at)) for o in outs)
    assert kinds == [
        ("DigitalSurfaceModel", "2023-05-12"),
        ("DigitalSurfaceModel", "2025-05-20"),
        ("OrthoImage", "2023-05-12"),
        ("OrthoImage", "2025-05-20"),
    ]
    m = chain["ingest"].metrics
    assert m["acquisitions"] == 2 and m["temporal_overlap_ratio"] == pytest.approx(1.0, abs=0.01)
    assert m["lidar_cell_coverage_20230512"] > 0.4
    # 건물 높이(4~28 m)가 지물 높이로 나온다
    assert 4 < m["max_height_m_20250520"] < 40


def test_tiling_stacks_bands(store, chain):
    man = get_json(store, chain["ard"].outputs[0].storage_uri)
    assert man["bands"] == [
        "2023-05-12:red", "2023-05-12:green", "2023-05-12:blue",
        "2025-05-20:red", "2025-05-20:green", "2025-05-20:blue",
        "2023-05-12:height", "2025-05-20:height",
    ]  # fmt: skip
    assert man["display_bands"] == [3, 4, 5] and man["reference_display_bands"] == [0, 1, 2]
    assert man["radiometric_normalization"][0]["acquired_at"] == "2025-05-20"


def test_change_labels(store, chain):
    ls = get_json(store, chain["labels"].outputs[0].storage_uri)
    counts = ls["quality"]["change_polygons"]
    assert all(counts[c] > 0 for c in ("NewBuilding", "DemolishedBuilding", "ModifiedBuilding", "LandChange")), counts
    assert set(ls["quality"]["class_pixels"]) >= {"0", "1", "2", "3", "4"}
    ref = json.loads(store.get_bytes(ls["change_reference_uri"]))
    assert {f["properties"]["concept"] for f in ref["features"]} == {
        "NewBuilding",
        "DemolishedBuilding",
        "ModifiedBuilding",
        "LandChange",
    }


def test_multichannel_model_and_report(store, chain):
    card = get_json(store, chain["model"].outputs[0].storage_uri)
    assert card["architecture"]["in_channels"] == 8 and card["architecture"]["num_classes"] == 5
    assert card["class_weights"] is not None and len(card["class_weights"]) == 5
    report = get_json(store, chain["eval"].outputs[0].storage_uri)
    assert {"foreground_F1", "foreground_IoU", "IoU", "F1"} <= set(report["metrics"])
    assert [c["label"] for c in report["class_summary"]] == ["건물 멸실", "토지 변화", "건물 변경", "건물 신축"]
    assert all(c["color"] != [255, 0, 0] for c in report["class_summary"])
    assert report["preview_layout"] == "pair" and store.exists(report["preview_uri"])


def test_single_date_is_rejected(store):
    one = [i for i in raw_inputs(store.meta) if i["acquired_at"] == "2023-05-12"]
    r = run_job(spec("one", "multitemporal-ingest", one), store, ENTRYPOINTS)
    assert (r.status, r.reason) == ("failed", "insufficient_acquisitions")


def test_derive_rules():
    before = [box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)]
    after = [box(0, 0, 10, 10), box(20, 0, 30, 18), box(60, 0, 70, 10)]  # 유지, 증축, 신축 (40 멸실)
    land1 = [(box(0, 30, 20, 50), "초지"), (box(30, 30, 50, 50), "초지")]
    land2 = [(box(0, 30, 20, 50), "나지"), (box(30, 30, 50, 50), "초지")]
    ch = derive_changes(before, after, land1, land2)
    assert [g.bounds for g in ch["NewBuilding"]] == [(60, 0, 70, 10)]
    assert [g.bounds for g in ch["DemolishedBuilding"]] == [(40, 0, 50, 10)]
    assert ch["ModifiedBuilding"][0].area == pytest.approx(80)
    assert [g.bounds for g in ch["LandChange"]] == [(0, 30, 20, 50)]


def test_transfer_from_rgb_model():
    src = UNet(in_channels=3, num_classes=2, base=8)
    dst = UNet(in_channels=8, num_classes=5, base=8)
    info = transfer_weights(dst, src.state_dict(), [[0, 1, 2], [3, 4, 5]])
    assert info["adapted"] == 1 and "head.weight" in info["skipped"]
    w = dst.state_dict()["down.0.0.weight"]
    assert torch.allclose(w[:, 0:3], src.state_dict()["down.0.0.weight"] / 2)
    assert torch.count_nonzero(w[:, 6:]) == 0
    assert np.isclose(info["copied"] + info["adapted"] + len(info["skipped"]), len(src.state_dict()))
