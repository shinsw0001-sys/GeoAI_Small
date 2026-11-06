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
    assert spec.label_classes == [{"iri": DOMAIN.replace("Domain", "Building"), "label": "건물", "value": 1}]
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
