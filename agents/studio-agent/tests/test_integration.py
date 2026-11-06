"""실제 PostgreSQL(PostGIS)·NATS JetStream 으로 흐름 전체를 시험한다.

실행:
  docker compose -f deploy/compose/test-deps.yaml up -d
  GEOAI_TEST_DATABASE_URL=postgresql://geoai:geoai@localhost:5432/geoai \\
  GEOAI_TEST_NATS_URL=nats://localhost:4222 pytest -m integration
"""

import asyncio
import os
import uuid

import pytest
from conftest import asset, request_envelope
from geoai_common import Envelope, NatsBus, Performative, PostgresStateStore, TaskRecord, TaskStatus
from geoai_common.bus import DLQ_SUBJECT
from geoai_common.testing import EchoSubAgent
from studio_agent import StudioAgent

DB = os.environ.get("GEOAI_TEST_DATABASE_URL")
NATS = os.environ.get("GEOAI_TEST_NATS_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (DB and NATS), reason="GEOAI_TEST_DATABASE_URL·GEOAI_TEST_NATS_URL 미설정"),
]

SUB_AGENTS = ["geo-data-production-agent", "preprocessing-agent", "model-training-agent", "analysis-evaluation-agent"]


async def wait_for(predicate, timeout=15.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(0.1)
    return False


async def test_postgres_compare_and_set():
    store = await PostgresStateStore(DB).open()
    try:
        tid = str(uuid.uuid4())
        rec = TaskRecord(
            task_id=tid, project_id="p", conversation_id="c", task_type="x", status=TaskStatus.PLANNED, agent="a"
        )
        await store.create_tasks([rec])
        results = await asyncio.gather(
            *[store.transition(tid, {TaskStatus.PLANNED}, TaskStatus.RUNNING) for _ in range(5)]
        )
        assert sum(r is not None for r in results) == 1
    finally:
        await store.close()


async def test_end_to_end_over_nats_and_postgres(onto):
    bus = await NatsBus(NATS).connect()
    store = await PostgresStateStore(DB).open()
    try:
        studio = StudioAgent("studio-agent", bus, store, onto)
        await studio.start()
        for sid in SUB_AGENTS:
            await EchoSubAgent(sid, bus, store, onto).start()
        inbox: list[Envelope] = []

        async def collect(e: Envelope):
            inbox.append(e)

        requester = f"it-{uuid.uuid4().hex[:8]}"
        await bus.subscribe(requester, collect)

        req = Envelope.model_validate(
            request_envelope(
                "InferenceTask",
                [asset("a1", "DroneImage")],
                sender=requester,
                traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            )
        )
        await bus.publish(req)

        assert await wait_for(lambda: _has(inbox, Performative.RESULT)), [e.performative for e in inbox]
        result = next(e for e in inbox if e.performative is Performative.RESULT)
        parent = await store.get_task(result.task_id)
        assert parent.status is TaskStatus.SUCCEEDED
        children = await store.list_children(parent.task_id)
        assert len(children) == 5 and all(c.status is TaskStatus.SUCCEEDED for c in children)
        assert result.traceparent == req.traceparent
        # 운영 콘솔 조회용: 최상위 작업 목록과 대화 메시지 기록
        assert parent.task_id in [t.task_id for t in await store.list_tasks(project_id=parent.project_id, limit=200)]
        history = await store.list_messages(parent.conversation_id)
        assert history[0].message_id == req.message_id
        assert {m.performative for m in history} >= {Performative.REQUEST, Performative.ACCEPT, Performative.RESULT}
    finally:
        await bus.close()
        await store.close()


async def _has(inbox, performative):
    return any(e.performative is performative for e in inbox)


async def test_failing_handler_goes_to_dead_letter_subject():
    bus = await NatsBus(NATS).connect()
    agent_id = f"broken-{uuid.uuid4().hex[:8]}"
    calls = 0

    async def broken(_: Envelope):
        nonlocal calls
        calls += 1
        raise RuntimeError("처리기 오류")

    try:
        dead: list[bytes] = []

        async def on_dead(msg):
            dead.append(msg.data)

        sub = await bus._nc.subscribe(f"{DLQ_SUBJECT}.{agent_id}", cb=on_dead)
        await bus.subscribe(agent_id, broken)
        await bus.publish(
            Envelope(
                performative="event", sender="ops-console", recipient=agent_id, project_id="p", ontology_version="0.1.0"
            )
        )

        async def landed():
            return bool(dead)

        assert await wait_for(landed, timeout=20)
        assert calls == 3
        await sub.unsubscribe()
    finally:
        await bus.close()
