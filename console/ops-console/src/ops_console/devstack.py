"""개발용 일체형 실행 (geoai-dev).

Studio Agent, 전문 에이전트 4종, 운영 콘솔을 한 프로세스에 띄우고, 실행 작업은 같은 장비의
별도 프로세스(처리 도구)로 돌린다. 쿠버네티스 없이 분야별 전 과정(건물 추출, 변화탐지)을 화면에서 점검하는 용도다.

  geoai-dev [--port 8000] [--data .geoai-data] [--no-sample]

- 메시지 버스·상태 저장소: GEOAI_NATS_URL·GEOAI_DATABASE_URL 이 있으면 그것을, 없으면 메모리 구현
- 데이터·모델 저장소: GEOAI_S3_ENDPOINT 가 있으면 S3, 없으면 --data 디렉터리
- 합성 표본이 없으면 만든다: 건물 추출(s3://raw/sample/), 변화탐지 두 시점 영상·라이다·벡터(s3://raw/sample-change/)
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from geoai_common import (
    InMemoryBus,
    InMemoryStateStore,
    MessageBus,
    NatsBus,
    OntologyRegistry,
    PostgresStateStore,
    StateStore,
)
from geoai_common.config import AgentSettings
from geoai_common.jobs import JobRunner, SubprocessJobRunner
from geoai_common.objectstore import ObjectStore, object_store_from_env

from .app import ConsoleDeps, ConsoleSettings, create_app

log = logging.getLogger(__name__)


@dataclass
class Stack:
    studio_app: FastAPI
    studio: object
    specialists: list
    bus: MessageBus
    store: StateStore
    objects: ObjectStore


async def build_stack(objects: ObjectStore, runner: JobRunner, settings: AgentSettings | None = None) -> Stack:
    """Studio Agent 와 전문 에이전트 4종을 같은 버스·저장소에 붙인다."""
    from specialist_agents import SPECIALIST_AGENT_IDS, SpecialistAgent
    from studio_agent import StudioAgent
    from studio_agent.app import Runtime
    from studio_agent.app import create_app as create_studio

    settings = settings or AgentSettings.from_env("studio-agent")
    ontology = OntologyRegistry.load(settings.ontology_dir, settings.domain_dirs)
    bus: MessageBus = await NatsBus(settings.nats_url).connect() if settings.nats_url else InMemoryBus()
    store: StateStore = (
        await PostgresStateStore(settings.database_url).open() if settings.database_url else InMemoryStateStore()
    )
    studio = StudioAgent("studio-agent", bus, store, ontology)
    await studio.start()
    specialists = []
    for aid in SPECIALIST_AGENT_IDS:
        agent = SpecialistAgent(aid, bus, store, ontology, objects=objects, runner=runner)
        await agent.start()
        specialists.append(agent)
    runtime = Runtime(agent=studio, bus=bus, store=store)
    studio_app = create_studio(runtime=runtime)
    studio_app.state.runtime = runtime  # 프로세스 안 호출(ASGITransport)은 수명 주기를 거치지 않는다
    return Stack(studio_app=studio_app, studio=studio, specialists=specialists, bus=bus, store=store, objects=objects)


@asynccontextmanager
async def stack_context(
    objects: ObjectStore, runner: JobRunner, seed_sample: bool = True
) -> AsyncIterator[tuple[Stack, ConsoleDeps]]:
    if seed_sample:
        from geoai_tools.__main__ import seed_change_sample
        from geoai_tools.__main__ import seed_sample as seed

        if not objects.list("s3://raw/sample/"):
            log.info("건물 추출 합성 표본을 만듭니다 (s3://raw/sample/)")
            seed(objects)
        if not objects.list("s3://raw/sample-change/"):
            log.info("변화탐지 합성 표본(두 시점 영상·라이다)을 만듭니다 (s3://raw/sample-change/)")
            seed_change_sample(objects)
    stack = await build_stack(objects, runner)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=stack.studio_app), base_url="http://studio", timeout=60
    )
    try:
        yield stack, ConsoleDeps(studio=client, objects=objects, settings=ConsoleSettings.from_env())
    finally:
        for agent in stack.specialists:
            for task in list(agent._running.values()):
                task.cancel()
        await client.aclose()
        await stack.bus.close()
        await stack.store.close()


def create_dev_app(data_dir: str, seed_sample: bool = True) -> FastAPI:
    root = Path(data_dir).resolve()
    os.environ["GEOAI_LOCAL_STORE"] = str(root)
    objects = object_store_from_env()
    runner = SubprocessJobRunner(objects, root / "_jobs", env={"GEOAI_LOCAL_STORE": str(root)})

    @asynccontextmanager
    async def deps_context() -> AsyncIterator[ConsoleDeps]:
        async with stack_context(objects, runner, seed_sample) as (_, deps):
            yield deps

    return create_app(deps_context=deps_context)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("GEOAI_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(prog="geoai-dev")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data", default=".geoai-data")
    parser.add_argument("--no-sample", action="store_true")
    args = parser.parse_args()
    app = create_dev_app(args.data, seed_sample=not args.no_sample)
    log.info("운영 콘솔: http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
