"""변화탐지(건물·토지) 종단 시험: 운영 콘솔 → Studio Agent → 전문 에이전트 4종 → 처리 도구(별도 프로세스).

두 시점 합성 항공 영상·라이다·건물 윤곽·토지피복을 저장소에 넣는다. 먼저 건물 추출 모델을 짧게 학습하고,
그 모델을 초기 가중치로 변화탐지 성능 검증 작업을 요청한다. 다시기 정합(생성·가공) → 다채널 타일 분할(전처리)
→ 변화 라벨 도출(라벨링) → 학습 → 평가 5단계가 실제로 돌고, 분야 권장 완료 기준 판정까지 끝나는지 본다.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")
pytest.importorskip("geoai_tools")

from geoai_common.jobs import SubprocessJobRunner  # noqa: E402
from geoai_common.objectstore import LocalObjectStore  # noqa: E402
from geoai_tools.__main__ import seed_change_sample, seed_sample  # noqa: E402
from ops_console import create_app  # noqa: E402
from ops_console.devstack import stack_context  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BUILDING = "https://ontology.geoai.local/domain/building-extraction#Domain"
CHANGE = "https://ontology.geoai.local/domain/change-detection#Domain"
GAI = "https://ontology.geoai.local/core#"

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def console(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("change")
    mp = pytest.MonkeyPatch()
    mp.setenv("GEOAI_ONTOLOGY_DIR", str(ROOT / "ontology"))
    mp.setenv("GEOAI_LOCAL_STORE", str(tmp / "store"))
    objects = LocalObjectStore(tmp / "store")
    building = seed_sample(objects, size=768)
    change = seed_change_sample(objects, size=1024)
    runner = SubprocessJobRunner(objects, tmp / "jobs", env={"GEOAI_LOCAL_STORE": str(tmp / "store")}, poll_seconds=0.3)

    @asynccontextmanager
    async def deps_context():
        async with stack_context(objects, runner, seed_sample=False) as (_, deps):
            yield deps

    with TestClient(create_app(deps_context=deps_context)) as client:
        client.building, client.change = building, change
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


def acquisitions(meta, lidar=True):
    return [
        {
            "acquired_at": a["acquired_at"],
            "image_uris": [a["image"]],
            "lidar_uris": [a["lidar"]] if lidar else [],
            "building_source": a["buildings"],
            "landcover_source": a["landcover"],
        }
        for a in meta["acquisitions"]
    ]


@pytest.fixture(scope="module")
def building_model(console):
    """변화탐지 초기 가중치로 쓸 건물 추출 모델 (짧게 학습)."""
    r = console.post(
        "/api/tasks",
        json={
            "project_id": "e2e-building",
            "domain": BUILDING,
            "task_type": GAI + "ModelTrainingTask",
            "image_uris": [console.building["image"]],
            "label_source": console.building["label_source"],
            # 학습 단계 지표(검증 IoU)로 판정한다. 분야 권장 기준(IoU, F1)은 성능 검증 단계 지표다.
            "criteria": [{"metric": "val_IoU", "operator": ">=", "threshold": 0.0}],
            "parameters": {"epochs": 4, "tile_size": 128},
        },
    )
    assert r.status_code == 202, r.text
    d = wait_finished(console, r.json()["task_id"])
    assert d["task"]["status"] == "Succeeded", d["task"]["detail"]
    models = console.get("/api/models").json()["models"]
    [m] = [m for m in models if m["domain"] == BUILDING]
    assert m["in_channels"] == 3
    return m["uri"]


def test_change_detection_end_to_end(console, building_model):
    r = console.post(
        "/api/tasks",
        json={
            "project_id": "e2e-change",
            "domain": CHANGE,
            "task_type": GAI + "EvaluationTask",
            "acquisitions": acquisitions(console.change),
            "criteria": [],
            "parameters": {"epochs": 50, "tile_size": 128, "pretrained_uri": building_model},
        },
    )
    assert r.status_code == 202, r.text
    plan = [(p["task_type"].rsplit("#", 1)[-1], p["agent"]) for p in r.json()["plan"]]
    assert plan == [
        ("GeoDataProductionTask", "geo-data-production-agent"),
        ("PreprocessingTask", "preprocessing-agent"),
        ("LabelingTask", "model-training-agent"),
        ("ModelTrainingTask", "model-training-agent"),
        ("EvaluationTask", "analysis-evaluation-agent"),
    ]
    d = wait_finished(console, r.json()["task_id"], timeout=1500)
    assert all(s["status"] == "Succeeded" for s in d["steps"]), [(s["label"], s["detail"]) for s in d["steps"]]
    steps = {s["task_type"].rsplit("#", 1)[-1]: s for s in d["steps"]}
    # 생성·가공: 시점별 정사영상 + 라이다 지물 높이
    kinds = sorted(o["asset_type"].rsplit("#", 1)[-1] for o in steps["GeoDataProductionTask"]["outputs"])
    assert kinds == ["DigitalSurfaceModel", "DigitalSurfaceModel", "OrthoImage", "OrthoImage"]
    assert d["artifacts"]["ard"]["acquisitions"] == ["2023-05-12", "2025-05-20"]
    assert len(d["artifacts"]["ard"]["bands"]) == 8
    assert sum(d["artifacts"]["labels"]["quality"]["change_polygons"].values()) > 0
    model = d["artifacts"]["model"]
    assert model["pretrained"]["adapted"] == 1 and "building-extraction" in model["pretrained"]["domain"]
    report = d["artifacts"]["report"]
    assert len(report["class_summary"]) == 4 and report["preview_layout"] == "pair"
    # 완료 기준은 분야 온톨로지 권장값(변화 여부 F1 ≥ 0.7, IoU ≥ 0.5)
    assert {c["metric"] for c in d["task"]["criteria"]} == {"foreground_F1", "IoU"}
    print("\n변화탐지 지표:", report["metrics"], "\n유형별:", [(c["label"], c["IoU"]) for c in report["class_summary"]])
    # 변화 여부는 여유가 커서 기준을 늘 넘는다. 유형 평균 IoU 는 좁은 띠 모양인 건물 변경 유형 탓에
    # 기준(0.5) 근처이고 처리 스레드 수에 따라 0.42~0.60 으로 흔들린다(합성 1024 장면, 시험 타일 소수).
    # 그래서 성능 수치 대신 판정이 지표와 일치하는지를 확인한다.
    metrics = report["metrics"]
    assert metrics["foreground_F1"] >= 0.7
    unmet = [c["metric"] for c in d["task"]["criteria"] if metrics[c["metric"]] < c["threshold"]]
    if unmet:
        assert d["task"]["status"] == "Failed" and all(m in d["task"]["detail"] for m in unmet), d["task"]["detail"]
    else:
        assert d["task"]["status"] == "Succeeded", d["task"]["detail"]


def test_single_acquisition_is_red(console):
    one = acquisitions(console.change)[:1]
    r = console.post(
        "/api/tasks",
        json={"project_id": "e2e-change", "domain": CHANGE, "task_type": GAI + "EvaluationTask", "acquisitions": one,
              "parameters": {"epochs": 1, "tile_size": 128}},
    )  # fmt: skip
    assert r.status_code == 202, r.text
    d = wait_finished(console, r.json()["task_id"])
    first = d["steps"][0]
    assert first["status"] == "Failed" and first["critical"]
    assert d["task"]["critical"] and "insufficient_acquisitions" not in d["task"]["label"]
    assert "2시점 이상" in d["task"]["detail"]
    alerts = console.get("/api/overview").json()["alerts"]
    assert any(a["task_id"] == d["task"]["task_id"] and a["title"] == "작업 실패" for a in alerts)
