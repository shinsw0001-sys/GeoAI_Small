"""메모리 버스·메모리 저장소로 접수→계획→배정→완료 흐름 전체를 시험한다."""

import pytest
from conftest import asset, request_envelope
from fastapi.testclient import TestClient
from geoai_common import GAI, Envelope, InMemoryBus, InMemoryStateStore, Performative, TaskStatus
from geoai_common.testing import EchoSubAgent
from studio_agent import StudioAgent
from studio_agent.app import Runtime, create_app

SUB_AGENTS = ["geo-data-production-agent", "preprocessing-agent", "model-training-agent", "analysis-evaluation-agent"]


async def system(onto, fail_on=None):
    bus, store = InMemoryBus(), InMemoryStateStore()
    studio = StudioAgent("studio-agent", bus, store, onto)
    await studio.start()
    subs = {}
    for sid in SUB_AGENTS:
        subs[sid] = EchoSubAgent(sid, bus, store, onto, fail_on=fail_on)
        await subs[sid].start()
    requester = []

    async def collect(e: Envelope):
        requester.append(e)

    await bus.subscribe("ops-console", collect)
    return bus, store, studio, subs, requester


async def test_training_request_runs_to_completion(onto):
    bus, store, studio, subs, inbox = await system(onto)
    req = Envelope.model_validate(request_envelope("ModelTrainingTask", [asset("a1", "AerialImage")]))
    accept = await studio.handle(req)
    assert accept.performative is Performative.ACCEPT
    assert len(accept.payload["plan"]) == 4
    await bus.drain()

    parent = await store.get_task(accept.task_id)
    assert parent.status is TaskStatus.SUCCEEDED
    children = await store.list_children(parent.task_id)
    assert all(c.status is TaskStatus.SUCCEEDED for c in children)

    # 학습 단계는 전처리 산출과 라벨 산출을 입력으로 받았다
    train_req = subs["model-training-agent"].received[-1]
    got = {i["asset_type"] for i in train_req.payload["inputs"]}
    assert got == {str(GAI.AnalysisReadyData), str(GAI.LabelSet)}
    # 완료 기준은 마지막 단계에만 전달된다
    assert train_req.payload["completion_criteria"]
    assert not subs["geo-data-production-agent"].received[0].payload["completion_criteria"]

    results = [e for e in inbox if e.performative is Performative.RESULT]
    assert len(results) == 1
    assert results[0].correlation_id == req.message_id
    assert results[0].payload["outputs"][0]["asset_type"] == str(GAI.Model)
    assert all(e.conversation_id == req.conversation_id for e in bus.published)


async def test_subtask_failure_fails_parent(onto):
    bus, store, studio, _, inbox = await system(onto, fail_on={str(GAI.LabelingTask)})
    accept = await studio.handle(
        Envelope.model_validate(request_envelope("ModelTrainingTask", [asset("a1", "AerialImage")]))
    )
    await bus.drain()
    parent = await store.get_task(accept.task_id)
    assert parent.status is TaskStatus.FAILED
    statuses = [c.status for c in await store.list_children(parent.task_id)]
    assert statuses == [TaskStatus.SUCCEEDED, TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.PLANNED]
    assert [e.performative for e in inbox] == [Performative.ACCEPT, Performative.FAILURE]


@pytest.fixture
def client(onto):
    bus, store = InMemoryBus(), InMemoryStateStore()
    rt = Runtime(agent=StudioAgent("studio-agent", bus, store, onto), bus=bus, store=store)
    with TestClient(create_app(runtime=rt)) as c:
        yield c


def test_http_accept_and_status(client):
    r = client.post("/v1/messages", json=request_envelope("SpatialAnalysisTask", [asset("o1", "OrthoImage")]))
    assert r.status_code == 202, r.text
    task_id = r.json()["task_id"]
    s = client.get(f"/v1/tasks/{task_id}").json()
    assert s["task"]["status"] == "Running"
    assert [t["status"] for t in s["subtasks"]] == ["Running"]


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (request_envelope("Model", [asset("a1", "AerialImage")]), "invalid_task_type"),
        (request_envelope("StudioTask", [asset("a1", "AerialImage")]), "invalid_task_type"),
        (request_envelope("AnalysisEvaluationTask", [asset("o1", "OrthoImage")]), "unplannable"),
        (request_envelope("ModelTrainingTask", [asset("a1", "AerialImage", crs=None)]), "shape_violation"),
        (request_envelope("PreprocessingTask", [asset("m1", "Model", crs=None)]), "unplannable"),
        (
            request_envelope("ModelTrainingTask", [asset("a1", "AerialImage")], ontology_version="1.0.0"),
            "ontology_version_mismatch",
        ),
    ],
)
def test_http_rejects(client, body, reason):
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 422
    assert r.json()["performative"] == "reject"
    assert r.json()["payload"]["reason"] == reason


def test_agent_card(client):
    card = client.get("/v1/agent-card").json()
    assert card["ontology_version"] == "0.1.0"
    assert {a["task_type"].rsplit("#", 1)[-1] for a in card["accepts"]} == {
        "GeoDataProductionTask",
        "PreprocessingTask",
        "LabelingTask",
        "ModelTrainingTask",
        "InferenceTask",
        "SpatialAnalysisTask",
        "EvaluationTask",
    }


class LowScoreAgent(EchoSubAgent):
    """완료 기준보다 낮은 지표를 돌려주는 모의 하위 에이전트."""

    async def send(self, envelope: Envelope) -> None:
        if envelope.performative is Performative.RESULT:
            envelope.payload["metrics"] = {k: v - 0.2 for k, v in envelope.payload["metrics"].items()}
        await super().send(envelope)


async def test_completion_criteria_not_met_fails_parent(onto):
    bus, store = InMemoryBus(), InMemoryStateStore()
    studio = StudioAgent("studio-agent", bus, store, onto)
    await studio.start()
    for sid in SUB_AGENTS:
        cls = LowScoreAgent if sid == "model-training-agent" else EchoSubAgent
        await cls(sid, bus, store, onto).start()
    inbox = []

    async def collect(e):
        inbox.append(e)

    await bus.subscribe("ops-console", collect)
    accept = await studio.handle(
        Envelope.model_validate(request_envelope("ModelTrainingTask", [asset("a1", "AerialImage")]))
    )
    await bus.drain()
    parent = await store.get_task(accept.task_id)
    assert parent.status is TaskStatus.FAILED
    assert parent.detail.startswith("완료 기준 미달: mIoU 0.5")
    assert parent.metrics["mIoU"] == pytest.approx(0.5)
    failure = [e for e in inbox if e.performative is Performative.FAILURE]
    assert failure and failure[0].payload["reason"] == "completion_criteria_not_met"
    # 하위 작업 자체는 모두 완료 상태다(실행은 성공, 판정만 미달)
    assert all(c.status is TaskStatus.SUCCEEDED for c in await store.list_children(parent.task_id))


async def test_domain_default_criteria_are_applied(onto):
    bus, store, studio, subs, _ = await system(onto)
    body = request_envelope("ModelTrainingTask", [asset("a1", "AerialImage")])
    body["payload"]["domain"] = "https://ontology.geoai.local/domain/building-extraction#Domain"
    body["payload"]["completion_criteria"] = []
    accept = await studio.handle(Envelope.model_validate(body))
    assert accept.performative is Performative.ACCEPT
    await bus.drain()
    parent = await store.get_task(accept.task_id)
    assert {c["metric"] for c in parent.request["completion_criteria"]} == {"IoU", "F1"}
    train_req = subs["model-training-agent"].received[-1]
    assert {c["metric"] for c in train_req.payload["completion_criteria"]} == {"IoU", "F1"}
    assert parent.status is TaskStatus.SUCCEEDED


def test_http_task_list_and_messages(client):
    r = client.post("/v1/messages", json=request_envelope("SpatialAnalysisTask", [asset("o1", "OrthoImage")]))
    task_id = r.json()["task_id"]
    tasks = client.get("/v1/tasks", params={"project_id": "p-001"}).json()["tasks"]
    assert [t["task_id"] for t in tasks] == [task_id]
    assert client.get("/v1/tasks", params={"project_id": "none"}).json()["tasks"] == []
    msgs = client.get(f"/v1/tasks/{task_id}/messages").json()["messages"]
    assert [m["performative"] for m in msgs][:2] == ["request", "accept"]
    assert client.get("/v1/tasks/nope/messages").status_code == 404


def test_agent_card_domain_defaults(client):
    card = client.get("/v1/agent-card").json()
    building = next(d for d in card["domains"] if "building-extraction" in d["iri"])
    assert building["label"] == "건물 추출"
    assert {c["metric"] for c in building["default_criteria"]} == {"IoU", "F1"}
    assert building["label_concepts"][0]["label"] == "건물"
