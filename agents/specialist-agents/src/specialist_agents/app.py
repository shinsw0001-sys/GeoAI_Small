"""전문 에이전트 HTTP API (쿠버네티스 활성·준비 확인과 동기 진입점).

- POST /v1/messages   메시지 봉투 접수. 비동기 경로는 메시지 버스다.
- GET  /v1/agent-card 담당 작업 유형, 실행 작업 유형, 실행 중 작업
- GET  /healthz, /readyz

GEOAI_AGENT_ID 로 어느 전문 에이전트로 동작할지 정한다. 이미지는 4종이 같다.
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
from geoai_common.jobs import job_runner_from_env
from geoai_common.objectstore import object_store_from_env

from .agent import SpecialistAgent

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    agent: SpecialistAgent
    bus: MessageBus
    store: StateStore


async def build_runtime(settings: AgentSettings) -> Runtime:
    ontology = OntologyRegistry.load(settings.ontology_dir, settings.domain_dirs)
    bus: MessageBus = await NatsBus(settings.nats_url).connect() if settings.nats_url else InMemoryBus()
    store: StateStore = (
        await PostgresStateStore(settings.database_url).open() if settings.database_url else InMemoryStateStore()
    )
    objects = object_store_from_env()
    agent = SpecialistAgent(
        settings.agent_id, bus, store, ontology, objects=objects, runner=job_runner_from_env(objects)
    )
    await agent.start()
    log.info(
        "전문 에이전트 기동: id=%s 담당=%s 실행기=%s",
        settings.agent_id,
        [ontology.label(t) for t in agent.handled_task_types()],
        type(agent.runner).__name__,
    )
    return Runtime(agent=agent, bus=bus, store=store)


def create_app(settings: AgentSettings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or AgentSettings.from_env("preprocessing-agent")

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        rt = runtime or await build_runtime(settings)
        app.state.runtime = rt
        yield
        if runtime is None:
            await rt.bus.close()
            await rt.store.close()

    app = FastAPI(title="GeoAI 전문 에이전트", version="0.1.0", lifespan=lifespan)

    def rt() -> Runtime:
        return app.state.runtime

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict:
        return {"status": "ready", "agent_id": rt().agent.agent_id}

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

    return app
