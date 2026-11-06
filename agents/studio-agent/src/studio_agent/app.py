"""Studio Agent HTTP API.

- POST /v1/messages        메시지 봉투 접수 (동기 경로). request 는 accept/reject 답신을 바로 돌려준다.
- GET  /v1/tasks           최상위 작업 목록 (최근 순, project_id 로 거를 수 있다)
- GET  /v1/tasks/{task_id} 작업과 하위 작업 상태
- GET  /v1/tasks/{task_id}/messages  작업 대화의 메시지 기록 (진행 보고 포함)
- GET  /v1/agent-card      능력 명세: 받는 작업 유형, 분야, 온톨로지 버전
- GET  /healthz, /readyz   쿠버네티스 활성·준비 확인

비동기 경로는 메시지 버스(NATS JetStream)다. 두 경로 모두 같은 처리기(StudioAgent.handle)로 들어간다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from geoai_common import (
    Envelope,
    InMemoryBus,
    InMemoryStateStore,
    MessageBus,
    NatsBus,
    OntologyRegistry,
    Performative,
    PostgresStateStore,
    StateStore,
)
from geoai_common.config import AgentSettings

from .agent import StudioAgent

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    agent: StudioAgent
    bus: MessageBus
    store: StateStore


async def build_runtime(settings: AgentSettings) -> Runtime:
    ontology = OntologyRegistry.load(settings.ontology_dir, settings.domain_dirs)
    bus: MessageBus = await NatsBus(settings.nats_url).connect() if settings.nats_url else InMemoryBus()
    store: StateStore = (
        await PostgresStateStore(settings.database_url).open() if settings.database_url else InMemoryStateStore()
    )
    agent = StudioAgent(settings.agent_id, bus, store, ontology)
    await agent.start()
    log.info(
        "Studio Agent 기동: id=%s 온톨로지=%s 분야=%d 버스=%s 저장소=%s",
        settings.agent_id,
        ontology.version,
        len(ontology.domains()),
        type(bus).__name__,
        type(store).__name__,
    )
    return Runtime(agent=agent, bus=bus, store=store)


def create_app(settings: AgentSettings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or AgentSettings.from_env("studio-agent")

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        rt = runtime or await build_runtime(settings)
        app.state.runtime = rt
        yield
        if runtime is None:
            await rt.bus.close()
            await rt.store.close()

    app = FastAPI(title="GeoAI Studio Agent", version="0.1.0", lifespan=lifespan)

    def rt() -> Runtime:
        return app.state.runtime

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict:
        return {"status": "ready", "ontology_version": rt().agent.ontology.version}

    @app.get("/v1/agent-card")
    async def agent_card() -> dict:
        return rt().agent.capabilities()

    @app.post("/v1/messages")
    async def post_message(envelope: Envelope) -> JSONResponse:
        if envelope.recipient != rt().agent.agent_id:
            raise HTTPException(status_code=400, detail=f"수신자가 {rt().agent.agent_id} 가 아닙니다.")
        reply = await rt().agent.handle(envelope)
        if reply is None:
            return JSONResponse(status_code=202, content={"status": "received"})
        status = 202 if reply.performative is Performative.ACCEPT else 422
        return JSONResponse(status_code=status, content=reply.model_dump(mode="json"))

    @app.get("/v1/tasks")
    async def list_tasks(project_id: str | None = None, limit: int = 50) -> dict:
        tasks = await rt().store.list_tasks(project_id=project_id, limit=min(max(limit, 1), 200))
        return {"tasks": [t.model_dump(mode="json") for t in tasks]}

    @app.get("/v1/tasks/{task_id}/messages")
    async def task_messages(task_id: str) -> dict:
        task = await rt().store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="작업이 없습니다.")
        messages = await rt().store.list_messages(task.conversation_id)
        return {"messages": [m.model_dump(mode="json") for m in messages]}

    @app.get("/v1/tasks/{task_id}")
    async def get_task(task_id: str) -> dict:
        task = await rt().store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="작업이 없습니다.")
        children = await rt().store.list_children(task_id)
        return {
            "task": task.model_dump(mode="json"),
            "subtasks": [c.model_dump(mode="json") for c in children],
        }

    return app
