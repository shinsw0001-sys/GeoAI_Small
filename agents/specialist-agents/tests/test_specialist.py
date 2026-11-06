"""전문 에이전트: 접수 판단, 실행 작업 명세, 산출물 검증, 실패 보고."""

from pathlib import Path

import pytest
from geoai_common import GAI, DataAssetRef, Envelope, InMemoryBus, InMemoryStateStore, OntologyRegistry, Performative
from geoai_common.jobs import JobError, JobResult
from geoai_common.objectstore import LocalObjectStore
from specialist_agents import SpecialistAgent

ROOT = Path(__file__).resolve().parents[3]
DOMAIN = "https://ontology.geoai.local/domain/building-extraction#Domain"


class FakeRunner:
    def __init__(self, objects, outcome="ok"):
        self.objects = objects
        self.outcome = outcome
        self.specs = []

    async def run(self, spec, on_progress=None):
        self.specs.append(spec)
        if on_progress:
            await on_progress(50.0, "절반")
        if self.outcome == "error":
            raise JobError("tool_error", "모의 오류")
        uri = f"{spec.output_prefix}/out.json"
        if self.outcome != "missing":
            self.objects.put_bytes(uri, b"{}")
        asset_type = str(GAI.Model) if self.outcome == "wrongtype" else str(GAI.AnalysisReadyData)
        return JobResult(
            status="succeeded",
            outputs=[DataAssetRef(asset_id="o", asset_type=asset_type, storage_uri=uri)],
            metrics={"m": 1.0},
        )


@pytest.fixture(scope="module")
def onto():
    return OntologyRegistry.load(ROOT / "ontology")


async def make(onto, tmp_path, agent_id="preprocessing-agent", outcome="ok"):
    bus, store = InMemoryBus(), InMemoryStateStore()
    objects = LocalObjectStore(tmp_path)
    runner = FakeRunner(objects, outcome)
    agent = SpecialistAgent(agent_id, bus, store, onto, objects=objects, runner=runner)
    await agent.start()
    inbox = []

    async def collect(e):
        inbox.append(e)

    await bus.subscribe("studio-agent", collect)
    return bus, agent, runner, objects, inbox


def request(task, inputs, **params):
    return Envelope(
        performative=Performative.REQUEST,
        sender="studio-agent",
        recipient="x",
        project_id="p1",
        task_id="T1.s1",
        task_type=str(GAI[task]),
        ontology_version="0.1.0",
        payload={"domain": DOMAIN, "inputs": inputs, "parameters": params},
    )


def ortho(uri="s3://processed/p1/o.tif"):
    return {"asset_id": "o", "asset_type": str(GAI.OrthoImage), "storage_uri": uri, "crs": "EPSG:5186"}


async def test_accept_run_and_report(onto, tmp_path):
    bus, agent, runner, objects, inbox = await make(onto, tmp_path)
    objects.put_bytes("s3://processed/p1/o.tif", b"x")
    reply = await agent.handle(request("PreprocessingTask", [ortho()], tile_size=256))
    assert reply.performative is Performative.ACCEPT
    await agent.wait_idle()
    await bus.drain()
    spec = runner.specs[0]
    assert spec.entrypoint == "tiling" and spec.job_type.endswith("#TilingJob")
    assert spec.output_prefix == "s3://processed/p1/T1.s1"
    assert spec.label_classes == [
        {"iri": DOMAIN.replace("Domain", "Building"), "label": "건물", "value": 1, "geometry": "Polygon"}
    ]
    assert spec.parameters == {"tile_size": 256}
    perfs = [e.performative for e in inbox]
    assert perfs[0] is Performative.ACCEPT and perfs[-1] is Performative.RESULT
    assert Performative.PROGRESS in perfs
    assert all(e.correlation_id == reply.correlation_id for e in inbox)


async def test_rejects_other_agents_task(onto, tmp_path):
    _, agent, *_ = await make(onto, tmp_path)
    reply = await agent.handle(request("ModelTrainingTask", [ortho()]))
    assert reply.payload["reason"] == "not_handled"


async def test_labeling_requires_label_source(onto, tmp_path):
    _, agent, _, objects, _ = await make(onto, tmp_path, agent_id="model-training-agent")
    ard = {"asset_id": "a", "asset_type": str(GAI.AnalysisReadyData), "storage_uri": "s3://processed/p1/m.json"}
    objects.put_bytes(ard["storage_uri"], b"{}")
    reply = await agent.handle(request("LabelingTask", [ard]))
    assert reply.payload["reason"] == "missing_parameter"
    reply = await agent.handle(request("LabelingTask", [ard], label_source="s3://raw/none.geojson"))
    assert reply.payload["reason"] == "input_not_found"


async def test_missing_input_object(onto, tmp_path):
    _, agent, *_ = await make(onto, tmp_path)
    reply = await agent.handle(request("PreprocessingTask", [ortho("s3://processed/p1/none.tif")]))
    assert reply.payload["reason"] == "input_not_found"


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [("error", "tool_error"), ("missing", "output_verification_failed"), ("wrongtype", "output_verification_failed")],
)
async def test_failures_are_reported(onto, tmp_path, outcome, reason):
    bus, agent, _, objects, inbox = await make(onto, tmp_path, outcome=outcome)
    objects.put_bytes("s3://processed/p1/o.tif", b"x")
    await agent.handle(request("PreprocessingTask", [ortho()]))
    await agent.wait_idle()
    await bus.drain()
    assert inbox[-1].performative is Performative.FAILURE
    assert inbox[-1].payload["reason"] == reason


async def test_capabilities(onto, tmp_path):
    _, agent, *_ = await make(onto, tmp_path, agent_id="model-training-agent")
    card = agent.capabilities()
    assert {a["entrypoint"] for a in card["accepts"]} == {"rasterize-labels", "segmentation-train"}


def test_unknown_agent_id(onto, tmp_path):
    with pytest.raises(ValueError):
        SpecialistAgent(
            "nope-agent", InMemoryBus(), InMemoryStateStore(), onto, objects=LocalObjectStore(tmp_path), runner=None
        )


CD = "https://ontology.geoai.local/domain/change-detection#Domain"


def raw(uri, kind="AerialImage", date=None):
    a = {"asset_id": uri.rsplit("/", 1)[-1], "asset_type": str(GAI[kind]), "storage_uri": uri, "crs": "EPSG:5186"}
    if date:
        a["acquired_at"] = date
    return a


def change_request(task, inputs, **params):
    e = request(task, inputs, **params)
    e.payload["domain"] = CD
    return e


async def test_change_ingest_needs_two_dates(onto, tmp_path):
    bus, agent, runner, objects, _ = await make(onto, tmp_path, agent_id="geo-data-production-agent")
    for u in ("s3://raw/a.tif", "s3://raw/b.tif", "s3://raw/a.las"):
        objects.put_bytes(u, b"x")
    reply = await agent.handle(change_request("GeoDataProductionTask", [raw("s3://raw/a.tif", date="2023-05-12")]))
    assert reply.payload["reason"] == "insufficient_acquisitions"
    inputs = [
        raw("s3://raw/a.tif", date="2023-05-12"),
        raw("s3://raw/b.tif", date="2025-05-20"),
        raw("s3://raw/a.las", "LidarPointCloud", "2023-05-12"),
    ]
    reply = await agent.handle(change_request("GeoDataProductionTask", inputs))
    assert reply.performative is Performative.ACCEPT
    await agent.wait_idle()
    spec = runner.specs[0]
    assert (spec.entrypoint, spec.memory) == ("multitemporal-ingest", "16Gi")
    assert [str(a.acquired_at) for a in spec.inputs] == ["2023-05-12", "2025-05-20", "2023-05-12"]


async def test_change_labels_parameters(onto, tmp_path):
    _, agent, runner, objects, _ = await make(onto, tmp_path, agent_id="model-training-agent")
    ard = {"asset_id": "a", "asset_type": str(GAI.AnalysisReadyData), "storage_uri": "s3://processed/p1/m.json"}
    objects.put_bytes(ard["storage_uri"], b"{}")
    objects.put_bytes("s3://raw/b1.geojson", b"{}")
    reply = await agent.handle(change_request("LabelingTask", [ard]))
    assert reply.payload["reason"] == "missing_parameter" and "label_sources" in reply.payload["detail"]
    sources = [
        {"acquired_at": "2023-05-12", "buildings": "s3://raw/b1.geojson"},
        {"acquired_at": "2025-05-20", "buildings": "s3://raw/b2.geojson"},
    ]
    reply = await agent.handle(change_request("LabelingTask", [ard], label_sources=sources))
    assert reply.payload["reason"] == "input_not_found" and "b2.geojson" in reply.payload["detail"]
    objects.put_bytes("s3://raw/b2.geojson", b"{}")
    reply = await agent.handle(change_request("LabelingTask", [ard], label_sources=sources))
    assert reply.performative is Performative.ACCEPT
    await agent.wait_idle()
    assert runner.specs[0].entrypoint == "change-labels"
    assert [c["value"] for c in runner.specs[0].label_classes] == [1, 2, 3, 4]


RD = "https://ontology.geoai.local/domain/road-extraction#Domain"


async def test_road_labels_and_evaluation_jobs(onto, tmp_path):
    _, agent, runner, objects, _ = await make(onto, tmp_path, agent_id="model-training-agent")
    ard = {"asset_id": "a", "asset_type": str(GAI.AnalysisReadyData), "storage_uri": "s3://processed/p1/m.json"}
    objects.put_bytes(ard["storage_uri"], b"{}")
    e = request("LabelingTask", [ard])
    e.payload["domain"] = RD
    reply = await agent.handle(e)
    assert reply.payload["reason"] == "missing_parameter" and "도로 중심선" in reply.payload["detail"]
    objects.put_bytes("s3://raw/roads.geojson", b"{}")
    e = request("LabelingTask", [ard], label_source="s3://raw/roads.geojson")
    e.payload["domain"] = RD
    reply = await agent.handle(e)
    assert reply.performative is Performative.ACCEPT
    await agent.wait_idle()
    spec = runner.specs[0]
    assert spec.entrypoint == "road-labels"
    assert spec.label_classes == [
        {"iri": RD.replace("Domain", "Road"), "label": "도로", "value": 1, "geometry": "LineString"}
    ]

    _, agent, runner, objects, _ = await make(onto, tmp_path, agent_id="analysis-evaluation-agent")
    inputs = [
        {"asset_id": "m", "asset_type": str(GAI.Model), "storage_uri": "s3://models/p1/model.json"},
        {"asset_id": "l", "asset_type": str(GAI.LabelSet), "storage_uri": "s3://labels/p1/manifest.json"},
    ]
    for a in inputs:
        objects.put_bytes(a["storage_uri"], b"{}")
    e = request("EvaluationTask", inputs)
    e.payload["domain"] = RD
    assert (await agent.handle(e)).performative is Performative.ACCEPT
    await agent.wait_idle()
    assert runner.specs[0].entrypoint == "road-evaluate"
