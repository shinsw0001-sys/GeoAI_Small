"""메시지 버스.

주제(subject) 규칙: geoai.<계층>.<에이전트 식별자>.<수행 의도>
  예) geoai.studio.studio-agent.request
수신자가 없는 event 는 geoai.events.<보낸 에이전트>.event 로 보낸다.
처리기가 MAX_DELIVER(3)회 실패한 메시지는 배달 불가 주제 geoai.dlq.<에이전트> 로 옮긴다.
추적 문맥(traceparent)은 메시지 헤더로도 싣는다.

구현 두 가지:
- InMemoryBus: 단일 프로세스 시험·개발용
- NatsBus: NATS JetStream. 스트림 GEOAI 하나에 geoai.> 전체를 영속 저장한다.
  에이전트 식별자를 내구 소비자(durable consumer)와 큐 그룹 이름으로 써서
  같은 에이전트의 복제본(replica)이 메시지를 나눠 받는다.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from .messages import Envelope

log = logging.getLogger(__name__)

Handler = Callable[[Envelope], Awaitable[None]]

STREAM_NAME = "GEOAI"
SUBJECT_ROOT = "geoai"
MAX_DELIVER = 3
DLQ_SUBJECT = f"{SUBJECT_ROOT}.dlq"

AGENT_LAYERS: dict[str, str] = {
    "super-agent": "orchestration",
    "studio-agent": "studio",
    "geo-data-production-agent": "studio",
    "preprocessing-agent": "studio",
    "model-training-agent": "studio",
    "analysis-evaluation-agent": "studio",
    "base-operations-agent": "base",
    "dt-agent": "dt",
}


def layer_of(agent_id: str) -> str:
    """등록되지 않은 식별자(운영 콘솔, 외부 시스템)는 client 계층으로 본다."""
    return AGENT_LAYERS.get(agent_id, "client")


def subject_for(envelope: Envelope) -> str:
    if envelope.recipient is None:
        return f"{SUBJECT_ROOT}.events.{envelope.sender}.event"
    return f"{SUBJECT_ROOT}.{layer_of(envelope.recipient)}.{envelope.recipient}.{envelope.performative.value}"


def inbox_subject(agent_id: str) -> str:
    return f"{SUBJECT_ROOT}.{layer_of(agent_id)}.{agent_id}.>"


class MessageBus(Protocol):
    async def publish(self, envelope: Envelope) -> None: ...

    async def subscribe(self, agent_id: str, handler: Handler) -> None: ...

    async def close(self) -> None: ...


class InMemoryBus:
    """큐 하나로 메시지를 순서대로 전달한다. `drain()` 으로 처리 완료를 기다린다."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self._queue: asyncio.Queue[Envelope] = asyncio.Queue()
        self._pump: asyncio.Task[None] | None = None
        self.published: list[Envelope] = []
        self.undelivered: list[Envelope] = []
        self.dead_letters: list[Envelope] = []

    async def publish(self, envelope: Envelope) -> None:
        self.published.append(envelope)
        self._ensure_pump()
        await self._queue.put(envelope)

    async def subscribe(self, agent_id: str, handler: Handler) -> None:
        self._handlers.setdefault(agent_id, []).append(handler)
        self._ensure_pump()

    async def drain(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        if self._pump:
            self._pump.cancel()

    def _ensure_pump(self) -> None:
        if self._pump is None or self._pump.done():
            self._pump = asyncio.get_running_loop().create_task(self._run())

    async def _deliver(self, handler: Handler, envelope: Envelope) -> None:
        for attempt in range(1, MAX_DELIVER + 1):
            try:
                await handler(envelope)
                return
            except Exception:  # noqa: BLE001 - 버스는 처리기 오류로 멈추지 않는다
                log.exception("메시지 처리 실패 (%d/%d): %s", attempt, MAX_DELIVER, envelope.message_id)
        self.dead_letters.append(envelope)

    async def _run(self) -> None:
        while True:
            envelope = await self._queue.get()
            try:
                handlers = self._handlers.get(envelope.recipient or "", [])
                if not handlers:
                    self.undelivered.append(envelope)
                for handler in handlers[:1]:  # 큐 그룹과 같게 한 구독자만 받는다
                    await self._deliver(handler, envelope)
            finally:
                self._queue.task_done()


class NatsBus:
    def __init__(self, url: str) -> None:
        self._url = url
        self._nc = None
        self._js = None

    async def connect(self) -> NatsBus:
        import nats
        from nats.js.api import StreamConfig
        from nats.js.errors import BadRequestError

        self._nc = await nats.connect(self._url, name="geoai")
        self._js = self._nc.jetstream()
        config = StreamConfig(name=STREAM_NAME, subjects=[f"{SUBJECT_ROOT}.>"], duplicate_window=120)
        try:
            await self._js.add_stream(config)
        except BadRequestError:
            await self._js.update_stream(config)
        return self

    async def publish(self, envelope: Envelope) -> None:
        assert self._js is not None, "connect() 를 먼저 호출해야 합니다."
        headers = {"Nats-Msg-Id": envelope.message_id}  # 중복 전송 제거
        if envelope.traceparent:
            headers["traceparent"] = envelope.traceparent
        await self._js.publish(subject_for(envelope), envelope.model_dump_json().encode(), headers=headers)

    async def subscribe(self, agent_id: str, handler: Handler) -> None:
        assert self._js is not None, "connect() 를 먼저 호출해야 합니다."

        async def on_message(msg) -> None:  # type: ignore[no-untyped-def]
            try:
                envelope = Envelope.model_validate_json(msg.data)
            except ValueError:
                log.error("규약에 맞지 않는 메시지를 버립니다: %s", msg.subject)
                await msg.term()
                return
            try:
                await handler(envelope)
            except Exception as exc:  # noqa: BLE001
                delivered = msg.metadata.num_delivered
                if delivered >= MAX_DELIVER:
                    log.exception("재전달 %d회 실패, 배달 불가 주제로 옮깁니다: %s", delivered, envelope.message_id)
                    headers = dict(msg.headers or {})
                    headers.pop("Nats-Msg-Id", None)
                    headers.update({"GeoAI-Original-Subject": msg.subject, "GeoAI-Error": type(exc).__name__})
                    await self._js.publish(f"{DLQ_SUBJECT}.{agent_id}", msg.data, headers=headers)
                    await msg.term()
                else:
                    log.exception(
                        "메시지 처리 실패 (%d/%d), 재전달 요청: %s", delivered, MAX_DELIVER, envelope.message_id
                    )
                    await msg.nak(delay=2)
                return
            await msg.ack()

        await self._js.subscribe(
            inbox_subject(agent_id),
            queue=agent_id,
            durable=agent_id,
            cb=on_message,
            manual_ack=True,
        )

    async def close(self) -> None:
        if self._nc is not None:
            await self._nc.drain()
