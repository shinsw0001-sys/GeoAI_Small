"""건물 추출 종단 시험: 운영 콘솔 → Studio Agent → 전문 에이전트 4종 → 처리 도구(별도 프로세스).

합성 항공 영상·건물 윤곽을 저장소에 넣고, 운영 콘솔 응용 프로그램 인터페이스로 성능 검증 작업을
요청한다. 수집·가공 → 전처리 → 라벨링 → 학습 → 평가 5단계가 실제로 돌고, 완료 기준 판정까지
끝나는지 확인한다. 처리 도구(PyTorch, rasterio)가 설치되어 있어야 한다.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")
pytest.importorskip("geoai_tools")

from geoai_common.config import AgentSettings  # noqa: E402
from geoai_common.jobs import SubprocessJobRunner  # noqa: E402
from geoai_common.objectstore import LocalObjectStore  # noqa: E402
from geoai_tools.__main__ import seed_sample  # noqa: E402
from ops_console import create_app  # noqa: E402
from ops_console.devstack import stack_context  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BUILDING = "https://ontology.geoai.local/domain/building-extraction#Domain"
GAI = "https://ontology.geoai.local/core#"

pytestmark = pytest.mark.e2e


@pytest.fixture
def console(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOAI_ONTOLOGY_DIR", str(ROOT / "ontology"))
    monkeypatch.setenv("GEOAI_LOCAL_STORE", str(tmp_path / "store"))
    objects = LocalObjectStore(tmp_path / "store")
    meta = seed_sample(objects, size=1024)
    runner = SubprocessJobRunner(
        objects, tmp_path / "jobs", env={"GEOAI_LOCAL_STORE": str(tmp_path / "store")}, poll_seconds=0.3
    )

    @asynccontextmanager
    async def deps_context():
        async with stack_context(objects, runner, seed_sample=False) as (_, deps):
            yield deps

    assert AgentSettings.from_env("studio-agent").ontology_dir == str(ROOT / "ontology")
    with TestClient(create_app(deps_context=deps_context)) as client:
        client.meta = meta
        yield client


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
        "project_id": "e2e",
        "domain": BUILDING,
        "task_type": GAI + "EvaluationTask",
        "image_uris": [client.meta["image"]],
        "label_source": client.meta["label_source"],
        "criteria": [],
        "parameters": {"epochs": 15, "tile_size": 128},
    }
    body.update(overrides)
    return client.post("/api/tasks", json=body)


def test_building_extraction_end_to_end(console):
    r = request(console)
    assert r.status_code == 202, r.text
    plan = [p["task_type"].rsplit("#", 1)[-1] for p in r.json()["plan"]]
    assert plan == ["GeoDataProductionTask", "PreprocessingTask", "LabelingTask", "ModelTrainingTask", "EvaluationTask"]

    d = wait_finished(console, r.json()["task_id"])
    steps = {s["task_type"].rsplit("#", 1)[-1]: s for s in d["steps"]}
    assert all(s["status"] == "Succeeded" for s in d["steps"]), [(s["label"], s["detail"]) for s in d["steps"]]
    # 완료 기준은 분야 온톨로지 권장값(IoU ≥ 0.6, F1 ≥ 0.75)이 적용되었다
    assert {c["metric"] for c in d["task"]["criteria"]} == {"IoU", "F1"}
    report = d["artifacts"]["report"]
    assert report["split"] == "test" and report["tiles"] > 0
    assert d["artifacts"]["model"]["history"]
    assert d["artifacts"]["labels"]["quality"]["positive_pixel_ratio"] > 0
    assert steps["EvaluationTask"]["metrics"]["IoU"] == report["metrics"]["IoU"]
    print("\n성능 지표:", report["metrics"], "판정:", d["task"]["criteria"], "상태:", d["task"]["status"])
    assert d["task"]["status"] == "Succeeded", d["task"]["detail"]
    # 산출물 내려받기
    assert console.get("/api/objects", params={"uri": report["preview_uri"]}).headers["content-type"] == "image/png"
    # 현황 화면에 완료로 집계되고 위험 알림이 없다
    o = console.get("/api/overview").json()
    assert o["counts"]["Succeeded"] == 1 and o["alerts"] == []


def test_unreachable_criteria_raise_red_alert(console):
    r = request(
        console,
        criteria=[{"metric": "IoU", "operator": ">=", "threshold": 0.999}],
        parameters={"epochs": 1, "tile_size": 128},
    )
    d = wait_finished(console, r.json()["task_id"])
    assert d["task"]["status"] == "Failed" and d["task"]["criteria_failed"]
    assert d["task"]["critical"] is True
    o = console.get("/api/overview").json()
    assert o["alerts"][0]["title"] == "완료 기준 미달"


def test_missing_label_source_fails_labeling_step(console):
    r = request(console, label_source=None, parameters={"epochs": 1, "tile_size": 128})
    d = wait_finished(console, r.json()["task_id"])
    steps = {s["task_type"].rsplit("#", 1)[-1]: s for s in d["steps"]}
    assert steps["LabelingTask"]["status"] == "Failed" and steps["LabelingTask"]["critical"]
    assert "label_source" in d["task"]["detail"]
    assert steps["ModelTrainingTask"]["status"] == "Planned"
