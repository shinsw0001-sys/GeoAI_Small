"""토지피복 분류 종단 시험: 운영 콘솔 → Studio Agent → 전문 에이전트 4종 → 처리 도구(별도 프로세스).

합성 항공 영상과 토지피복 다각형(환경부 토지피복지도 형식, L1_CODE)을 저장소에 넣고 성능 검증 작업을 요청한다.
정사영상 정합(범용) → 타일 분할(범용) → 토지피복 라벨(분야 전용) → 분할 모델 학습(범용, 미분류 제외)
→ 토지피복 분류 성능 검증(분야 전용) 5단계가 실제로 돌고, 분야 권장 완료 기준(전체 정확도, mIoU) 판정까지 끝나는지 본다.
대분류에 대응되지 않는 코드만 있는 벡터를 주면 라벨링 단계가 빨간 실패가 된다.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")
pytest.importorskip("geoai_tools")

from geoai_common.jobs import SubprocessJobRunner  # noqa: E402
from geoai_common.objectstore import LocalObjectStore, put_json  # noqa: E402
from geoai_tools.__main__ import seed_landcover_sample  # noqa: E402
from ops_console import create_app  # noqa: E402
from ops_console.devstack import stack_context  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LC = "https://ontology.geoai.local/domain/land-cover-classification#Domain"
GAI = "https://ontology.geoai.local/core#"

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def console(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("landcover")
    mp = pytest.MonkeyPatch()
    mp.setenv("GEOAI_ONTOLOGY_DIR", str(ROOT / "ontology"))
    mp.setenv("GEOAI_LOCAL_STORE", str(tmp / "store"))
    objects = LocalObjectStore(tmp / "store")
    meta = seed_landcover_sample(objects, size=1024)
    ring = [[198000, 552000], [198400, 552000], [198400, 551600], [198000, 551600], [198000, 552000]]
    bad = {"type": "Feature", "properties": {"L1_CODE": "900"}, "geometry": {"type": "Polygon", "coordinates": [ring]}}
    crs = {"type": "name", "properties": {"name": "EPSG:5186"}}
    put_json(objects, "s3://raw/sample-landcover/unknown-codes.geojson", {"type": "FeatureCollection", "crs": crs,
                                                                          "features": [bad]})  # fmt: skip
    runner = SubprocessJobRunner(objects, tmp / "jobs", env={"GEOAI_LOCAL_STORE": str(tmp / "store")}, poll_seconds=0.3)

    @asynccontextmanager
    async def deps_context():
        async with stack_context(objects, runner, seed_sample=False) as (_, deps):
            yield deps

    with TestClient(create_app(deps_context=deps_context)) as client:
        client.meta = meta
        yield client
    mp.undo()


def wait_finished(client, task_id, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        d = client.get(f"/api/tasks/{task_id}").json()
        if d["task"]["status"] in ("Succeeded", "Failed", "Rejected"):
            return d
        time.sleep(1.0)
    raise AssertionError(f"제한 시간 초과: {[(s['label'], s['status']) for s in d['steps']]}")


def request(client, **overrides):
    body = {
        "project_id": "e2e-landcover",
        "domain": LC,
        "task_type": GAI + "EvaluationTask",
        "image_uris": [client.meta["image"]],
        "label_source": client.meta["label_source"],
        "criteria": [],
        "parameters": {"epochs": 30, "tile_size": 128},
    }
    body.update(overrides)
    return client.post("/api/tasks", json=body)


def test_land_cover_end_to_end(console):
    r = request(console)
    assert r.status_code == 202, r.text
    plan = [p["task_type"].rsplit("#", 1)[-1] for p in r.json()["plan"]]
    assert plan == ["GeoDataProductionTask", "PreprocessingTask", "LabelingTask", "ModelTrainingTask", "EvaluationTask"]

    d = wait_finished(console, r.json()["task_id"])
    assert all(s["status"] == "Succeeded" for s in d["steps"]), [(s["label"], s["detail"]) for s in d["steps"]]
    steps = {s["task_type"].rsplit("#", 1)[-1]: s for s in d["steps"]}
    # 라벨링: 분류 코드로 대분류 대응, 전면 분류(미분류 0 제외)
    labels = d["artifacts"]["labels"]
    assert labels["coverage"] == "exhaustive" and labels["quality"]["coverage"] > 0.99
    assert labels["quality"]["matched_by_code"] == labels["quality"]["features"]
    assert steps["LabelingTask"]["metrics"]["classes_present"] == 7
    # 평가: 토지피복 분류 성능 검증(분야 전용)의 보고서와 예측 토지피복 다각형
    report = d["artifacts"]["report"]
    assert report["kind"] == "landcover-evaluation" and report["split"] == "test"
    assert len(report["confusion_matrix"]["matrix"]) == 7
    kinds = sorted(o["asset_type"].rsplit("#", 1)[-1] for o in steps["EvaluationTask"]["outputs"])
    assert kinds == ["EvaluationReport", "ResultProduct"]
    lc = console.get("/api/objects", params={"uri": report["predictions_uri"]})
    assert lc.headers["content-type"] == "application/geo+json" and lc.json()["features"]
    # 완료 기준은 분야 온톨로지 권장값(전체 정확도 ≥ 0.85, mIoU ≥ 0.5)
    assert {c["metric"] for c in d["task"]["criteria"]} == {"overall_accuracy", "mIoU"}
    metrics = report["metrics"]
    print("\n토지피복 분류 지표:", metrics)
    print(
        "항목별:", [(c["label"], c["producer_accuracy"], c["user_accuracy"], c["IoU"]) for c in report["class_summary"]]
    )
    assert metrics["overall_accuracy"] >= 0.7 and metrics["kappa"] > 0.5
    unmet = [c["metric"] for c in d["task"]["criteria"] if metrics[c["metric"]] < c["threshold"]]
    if unmet:
        assert d["task"]["status"] == "Failed" and all(m in d["task"]["detail"] for m in unmet), d["task"]["detail"]
    else:
        assert d["task"]["status"] == "Succeeded", d["task"]["detail"]


def test_unknown_codes_fail_red(console):
    r = request(
        console,
        label_source="s3://raw/sample-landcover/unknown-codes.geojson",
        parameters={"epochs": 1, "tile_size": 128},
    )
    assert r.status_code == 202, r.text
    d = wait_finished(console, r.json()["task_id"])
    steps = {s["task_type"].rsplit("#", 1)[-1]: s for s in d["steps"]}
    assert steps["LabelingTask"]["status"] == "Failed" and steps["LabelingTask"]["critical"]
    assert "L1_CODE=900" in d["task"]["detail"]
    assert steps["ModelTrainingTask"]["status"] == "Planned"
    alerts = console.get("/api/overview").json()["alerts"]
    assert any(a["task_id"] == d["task"]["task_id"] and a["title"] == "작업 실패" for a in alerts)
