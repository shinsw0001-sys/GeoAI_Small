"""토지피복 분류 처리 도구 연쇄: 정사영상 정합 → 타일 분할 → 토지피복 라벨 → 학습(미분류 제외) → 분류 성능 검증.
항목 대응(분류 코드·이름), 미분류 처리, 혼동 행렬 지표는 정답을 아는 작은 예로 따로 확인한다."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from geoai_common.jobs import JobSpec  # noqa: E402
from geoai_common.objectstore import LocalObjectStore, get_json, put_json  # noqa: E402
from geoai_tools import ENTRYPOINTS  # noqa: E402
from geoai_tools.__main__ import seed_landcover_sample  # noqa: E402
from geoai_tools.landcover_evaluate import LANDCOVER_RGB, accuracy_metrics  # noqa: E402
from geoai_tools.landcover_labels import class_lookup, class_of, major_code  # noqa: E402
from geoai_tools.model import class_weights, decode, dice_loss  # noqa: E402
from geoai_tools.runtime import run_job  # noqa: E402

GAI = "https://ontology.geoai.local/core#"
LC = "https://ontology.geoai.local/domain/land-cover-classification#"
# 온톨로지 순서(IRI 순 화소값)와 같게 둔다
NAMES = [
    ("Agriculture", "농업지역", "200"),
    ("Barren", "나지", "600"),
    ("Forest", "산림지역", "300"),
    ("Grassland", "초지", "400"),
    ("UrbanArea", "시가화·건조지역", "100"),
    ("Water", "수역", "700"),
    ("Wetland", "습지", "500"),
]
CLASSES = [
    {"iri": LC + n, "label": label, "value": i + 1, "geometry": "Polygon", "notation": code}
    for i, (n, label, code) in enumerate(NAMES)
]


def spec(name, entry, inputs, **params):
    return JobSpec(
        job_id=f"lc-{name}",
        job_type=f"https://ontology.geoai.local/jobs#{name}",
        entrypoint=entry,
        task_id=f"task.{name}",
        task_type=GAI + "X",
        project_id="p",
        domain=LC + "Domain",
        label_classes=CLASSES,
        inputs=inputs,
        parameters=params,
        completion_criteria=[{"metric": "overall_accuracy", "operator": ">=", "threshold": 0.1}],
        output_prefix=f"s3://work/p/{name}",
        progress_uri=f"s3://jobs/lc-{name}/progress.json",
        result_uri=f"s3://jobs/lc-{name}/result.json",
    )


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = LocalObjectStore(tmp_path_factory.mktemp("store"))
    s.meta = seed_landcover_sample(s, size=512)
    return s


@pytest.fixture(scope="module")
def chain(store):
    raw = {"asset_id": "a1", "asset_type": GAI + "AerialImage", "storage_uri": store.meta["image"], "crs": "EPSG:5186"}
    out = {}
    r = run_job(spec("ingest", "ortho-ingest", [raw]), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    r = run_job(spec("tiling", "tiling", r.outputs, tile_size=128, block_tiles=1), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["ard"] = r
    r = run_job(
        spec("labels", "landcover-labels", r.outputs, label_source=store.meta["label_source"]), store, ENTRYPOINTS
    )
    assert r.status == "succeeded", r.detail
    out["labels"] = r
    r = run_job(spec("train", "segmentation-train", out["ard"].outputs + r.outputs, epochs=3), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["model"] = r
    r = run_job(spec("eval", "landcover-evaluate", r.outputs + out["labels"].outputs), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["eval"] = r
    return out


def test_landcover_labels(store, chain):
    ls = get_json(store, chain["labels"].outputs[0].storage_uri)
    q = ls["quality"]
    # 전면 분류: 0 = 미분류(학습·평가 제외), 표본은 장면 전체를 덮는다
    assert ls["ignore_value"] == 0 and ls["coverage"] == "exhaustive"
    assert ls["classes"][0] == {"value": 0, "label": "미분류", "iri": None}
    assert q["coverage"] > 0.99 and q["unmatched_class"] == 0
    # 표본은 L1_CODE 로 대응된다(이름 대응 없음)
    assert q["matched_by_code"] == q["features"] and q["matched_by_name"] == 0
    shares = {c["label"]: c["share"] for c in q["classes"]}
    assert abs(sum(shares.values()) - 1) < 1e-3 and shares["수역"] > 0
    assert chain["labels"].metrics["classes_present"] >= 5


def test_landcover_model_ignores_unlabelled(store, chain):
    card = get_json(store, chain["model"].outputs[0].storage_uri)
    assert card["ignore_value"] == 0 and card["architecture"]["num_classes"] == 8
    assert card["class_weights"][0] == 0.0


def test_landcover_report(store, chain):
    report = get_json(store, chain["eval"].outputs[0].storage_uri)
    m = report["metrics"]
    assert {"overall_accuracy", "kappa", "mIoU", "IoU", "F1", "fwIoU"} <= set(m)
    assert all(0.0 <= m[k] <= 1.0 for k in ("overall_accuracy", "mIoU", "F1", "fwIoU")) and -1 <= m["kappa"] <= 1
    assert report["kind"] == "landcover-evaluation" and report["preview_layout"] == "landcover-triple"
    cm = np.array(report["confusion_matrix"]["matrix"])
    assert cm.shape == (7, 7) and cm.sum() == report["scored_pixels"]
    assert m["overall_accuracy"] == pytest.approx(np.trace(cm) / cm.sum(), abs=1e-4)
    assert [c["notation"] for c in report["class_summary"]] == [c["notation"] for c in CLASSES]
    assert store.exists(report["preview_uri"])
    lc = json.loads(store.get_bytes(report["predictions_uri"]))
    assert lc["features"] and {f["properties"]["L1_CODE"] for f in lc["features"]} <= set(LANDCOVER_RGB)
    assert all(f["geometry"]["type"] == "Polygon" for f in lc["features"])
    assert str(chain["eval"].outputs[1].asset_type).endswith("ResultProduct")


def test_landcover_coverage_and_unmatched(store, chain):
    # 장면 귀퉁이 작은 다각형 하나: 피복 비율 미달
    corner = [[198000, 552000], [198020, 552000], [198020, 551980], [198000, 551980], [198000, 552000]]
    crs = {"type": "name", "properties": {"name": "EPSG:5186"}}
    feat = {
        "type": "Feature",
        "properties": {"L1_CODE": "300"},
        "geometry": {"type": "Polygon", "coordinates": [corner]},
    }
    put_json(store, "s3://raw/lc/corner.geojson", {"type": "FeatureCollection", "crs": crs, "features": [feat]})
    r = run_job(
        spec("corner", "landcover-labels", chain["ard"].outputs, label_source="s3://raw/lc/corner.geojson"),
        store,
        ENTRYPOINTS,
    )
    assert (r.status, r.reason) == ("failed", "landcover_coverage_low")
    # 피복 비율 기준을 0으로 낮추면 통과하고 나머지는 미분류
    r = run_job(
        spec(
            "corner2",
            "landcover-labels",
            chain["ard"].outputs,
            label_source="s3://raw/lc/corner.geojson",
            min_coverage=0,
        ),
        store,
        ENTRYPOINTS,
    )
    assert r.status == "succeeded" and r.metrics["coverage"] < 0.01
    # 대분류에 없는 코드만 있으면 실패
    feat["properties"] = {"L1_CODE": "900"}
    put_json(store, "s3://raw/lc/bad.geojson", {"type": "FeatureCollection", "crs": crs, "features": [feat]})
    r = run_job(
        spec("bad", "landcover-labels", chain["ard"].outputs, label_source="s3://raw/lc/bad.geojson"),
        store,
        ENTRYPOINTS,
    )
    assert (r.status, r.reason) == ("failed", "no_landcover_polygon") and "L1_CODE=900" in r.detail


# ---------------------------------------------------------------------------
# 정답을 아는 예
# ---------------------------------------------------------------------------


def test_class_mapping():
    by_code, by_name = class_lookup(CLASSES)
    assert major_code("110") == "100" and major_code(212) == "200" and major_code("700.0") == "700"
    assert major_code("abc") is None and major_code("1000") is None
    assert class_of({"L1_CODE": "300"}, by_code, by_name, None) == (3, "code")
    # 중분류 코드만 있어도 대분류로
    assert class_of({"L2_CODE": "110"}, by_code, by_name, None) == (5, "code")
    # 코드 없으면 이름: 환경부 표기 "시가화건조지역"이 온톨로지 "시가화·건조지역"과 맞는다
    assert class_of({"L1_NAME": "시가화건조지역"}, by_code, by_name, None) == (5, "name")
    assert class_of({"class": "Water"}, by_code, by_name, None) == (6, "name")
    assert class_of({"class": "논"}, by_code, by_name, None)[0] is None
    assert class_of({"LC": "400"}, by_code, by_name, "LC") == (4, "code")


def test_accuracy_metrics_known():
    # Congalton 식 예: 2항목
    cm = np.array([[40, 10], [5, 45]])
    m, rows = accuracy_metrics(cm)
    assert m["overall_accuracy"] == pytest.approx(0.85)
    pe = (50 * 45 + 50 * 55) / 100**2
    assert m["kappa"] == pytest.approx((0.85 - pe) / (1 - pe))
    assert rows[0]["producer_accuracy"] == pytest.approx(0.8) and rows[0]["user_accuracy"] == pytest.approx(
        40 / 45, abs=1e-4
    )
    assert m["mIoU"] == pytest.approx((40 / 55 + 45 / 60) / 2)
    # 참조·예측 모두 없는 항목은 평균에서 뺀다
    m3, rows3 = accuracy_metrics(np.array([[40, 10, 0], [5, 45, 0], [0, 0, 0]]))
    assert m3["mIoU"] == pytest.approx(m["mIoU"]) and rows3[2]["IoU"] is None


def test_ignore_unlabelled_in_loss_and_decode():
    logits = torch.zeros(1, 3, 2, 2)
    logits[0, 0] = 5.0  # 미분류 점수가 가장 높아도
    logits[0, 2, 0, 0] = 1.0
    assert decode(logits, exhaustive=True)[0, 0, 0] == 2 and decode(logits, exhaustive=True).min() >= 1
    assert decode(logits)[0, 0, 0] == 0
    target = torch.tensor([[[2, 0], [0, 0]]])
    a = dice_loss(logits, target, 3, ignore_index=0)
    logits2 = logits.clone()
    logits2[0, :, 1, 1] = torch.tensor([0.0, 9.0, 0.0])  # 미분류 화소의 예측을 바꿔도 손실은 같다
    assert dice_loss(logits2, target, 3, ignore_index=0) == pytest.approx(float(a))
    w = class_weights(np.array([[0, 0, 1, 2, 2, 2]]), 3, ignore=0)
    assert w[0] == 0 and w[1] > w[2] and float(w[1:].mean()) == pytest.approx(1.0)
