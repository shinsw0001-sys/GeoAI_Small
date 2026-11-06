import json
from pathlib import Path

from geoai_common import Envelope, InMemoryStateStore, Performative, TaskRecord, TaskStatus, subject_for
from geoai_common.messages import json_schemas

ROOT = Path(__file__).resolve().parents[3]


def env(**kw) -> Envelope:
    base = dict(
        performative="request",
        sender="ops-console",
        recipient="studio-agent",
        project_id="p1",
        ontology_version="0.1.0",
    )
    base.update(kw)
    return Envelope(**base)


def test_committed_json_schemas_match_models():
    for name, schema in json_schemas().items():
        committed = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        assert committed == schema, f"schemas/{name} 갱신 필요: python -m geoai_common.messages schemas"


def test_reply_keeps_trace_ids():
    req = env()
    rep = req.reply(Performative.ACCEPT, "studio-agent")
    assert rep.conversation_id == req.conversation_id
    assert rep.correlation_id == req.message_id
    assert rep.recipient == "ops-console"


def test_subjects():
    assert subject_for(env()) == "geoai.studio.studio-agent.request"
    assert subject_for(env(recipient="super-agent", performative="result")) == "geoai.orchestration.super-agent.result"
    assert subject_for(env(recipient="ops-console", performative="result")) == "geoai.client.ops-console.result"
    assert subject_for(env(recipient=None, performative="event")) == "geoai.events.ops-console.event"


async def test_transition_is_compare_and_set():
    store = InMemoryStateStore()
    rec = TaskRecord(
        task_id="t", project_id="p", conversation_id="c", task_type="x", status=TaskStatus.PLANNED, agent="a"
    )
    await store.create_tasks([rec])
    first = await store.transition("t", {TaskStatus.PLANNED}, TaskStatus.RUNNING)
    second = await store.transition("t", {TaskStatus.PLANNED}, TaskStatus.RUNNING)
    assert first is not None and first.status is TaskStatus.RUNNING
    assert second is None
