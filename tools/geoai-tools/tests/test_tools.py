"""처리 도구 5종을 작업 실행기 없이 차례로 실행해 산출물 형식과 실패 처리를 확인한다."""

import json

import pytest

pytest.importorskip("torch")
from geoai_common.jobs import JobSpec  # noqa: E402
from geoai_common.objectstore import LocalObjectStore, get_json, put_json  # noqa: E402
from geoai_tools import ENTRYPOINTS  # noqa: E402
from geoai_tools.__main__ import seed_sample  # noqa: E402
from geoai_tools.runtime import run_job  # noqa: E402

GAI = "https://ontology.geoai.local/core#"
DOMAIN = "https://ontology.geoai.local/domain/building-extraction#Domain"
CLASSES = [{"iri": "https://ontology.geoai.local/domain/building-extraction#Building", "label": "건물", "value": 1}]


def spec(name, entry, inputs, **params):
    return JobSpec(
        job_id=f"t-{name}",
        job_type=f"https://ontology.geoai.local/jobs#{name}",
        entrypoint=entry,
        task_id=f"task.{name}",
        task_type=GAI + "X",
        project_id="p",
        domain=DOMAIN,
        label_classes=CLASSES,
        inputs=inputs,
        parameters=params,
        completion_criteria=[{"metric": "IoU", "operator": ">=", "threshold": 0.3}],
        output_prefix=f"s3://work/p/{name}",
        progress_uri=f"s3://jobs/t-{name}/progress.json",
        result_uri=f"s3://jobs/t-{name}/result.json",
    )


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = LocalObjectStore(tmp_path_factory.mktemp("store"))
    s.meta = seed_sample(s, size=1024)
    return s


@pytest.fixture(scope="module")
def chain(store):
    raw = {"asset_id": "a1", "asset_type": GAI + "AerialImage", "storage_uri": store.meta["image"], "crs": "EPSG:5186"}
    out = {}
    r = run_job(spec("ingest", "ortho-ingest", [raw]), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["ortho"] = r
    r = run_job(spec("tiling", "tiling", r.outputs, tile_size=128), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["ard"] = r
    r = run_job(
        spec("labels", "rasterize-labels", r.outputs, label_source=store.meta["label_source"]), store, ENTRYPOINTS
    )
    assert r.status == "succeeded", r.detail
    out["labels"] = r
    r = run_job(spec("train", "segmentation-train", out["ard"].outputs + r.outputs, epochs=4), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["model"] = r
    r = run_job(spec("eval", "segmentation-evaluate", r.outputs + out["labels"].outputs), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["eval"] = r
    return out


def test_ingest_ortho(chain):
    o = chain["ortho"].outputs[0]
    assert str(o.asset_type) == GAI + "OrthoImage"
    assert o.crs == "EPSG:5186" and o.ground_sample_distance == 0.5
    assert chain["ortho"].metrics["area_km2"] == pytest.approx(0.262144, abs=1e-3)


def test_tiling_splits(store, chain):
    m = get_json(store, chain["ard"].outputs[0].storage_uri)
    assert m["tile_size"] == 128 and len(m["tiles"]) == 64
    assert all(m["split_counts"][s] > 0 for s in ("train", "val", "test"))
    # 같은 공간 블록의 타일은 같은 집합에 들어간다
    by_block = {}
    for t in m["tiles"]:
        by_block.setdefault((t["row"] // 2, t["col"] // 2), set()).add(t["split"])
    assert all(len(v) == 1 for v in by_block.values())


def test_labels_quality(store, chain):
    m = get_json(store, chain["labels"].outputs[0].storage_uri)
    assert m["classes"][1]["label"] == "건물"
    assert 0.01 < m["quality"]["positive_pixel_ratio"] < 0.5


def test_model_and_report(store, chain):
    card = get_json(store, chain["model"].outputs[0].storage_uri)
    assert len(card["history"]) == 4 and card["weights_uri"].endswith("model.pt")
    report = get_json(store, chain["eval"].outputs[0].storage_uri)
    assert set(report["metrics"]) >= {"IoU", "mIoU", "F1", "precision", "recall", "object_F1"}
    assert report["completion_criteria"][0]["metric"] == "IoU"
    assert store.exists(report["preview_uri"])
    geo = json.loads(store.get_bytes(report["predictions_uri"]))
    assert geo["crs"]["properties"]["name"].endswith("5186")
    # 4세대 학습으로도 무작위보다는 확실히 낫다
    assert report["metrics"]["IoU"] > 0.2


def test_label_source_outside_image_fails(store, chain):
    far = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [0, 0]}}],
    }
    store.put_bytes("s3://raw/far.geojson", json.dumps(far).encode())
    r = run_job(
        spec("labels-far", "rasterize-labels", chain["ard"].outputs, label_source="s3://raw/far.geojson"),
        store,
        ENTRYPOINTS,
    )
    assert r.status == "failed"
    assert r.reason in {"label_no_overlap", "tool_error"}
    assert get_json(store, "s3://jobs/t-labels-far/result.json")["status"] == "failed"


def test_missing_crs_is_reported(store, tmp_path):
    import numpy as np
    import rasterio

    p = tmp_path / "nocrs.tif"
    with rasterio.open(p, "w", driver="GTiff", width=64, height=64, count=3, dtype="uint8") as d:
        d.write(np.ones((3, 64, 64), dtype=np.uint8))
    store.put_file("s3://raw/nocrs.tif", p)
    raw = {"asset_id": "n", "asset_type": GAI + "AerialImage", "storage_uri": "s3://raw/nocrs.tif", "crs": "EPSG:5186"}
    r = run_job(spec("nocrs", "ortho-ingest", [raw]), store, ENTRYPOINTS)
    assert (r.status, r.reason) == ("failed", "missing_crs")


def test_unknown_entrypoint(store):
    r = run_job(
        spec("x", "nope", [{"asset_id": "a", "asset_type": GAI + "Model", "storage_uri": "s3://raw/x"}]),
        store,
        ENTRYPOINTS,
    )
    assert r.reason == "unknown_entrypoint"
    put_json(store, "s3://jobs/ok.json", {"a": 1})
