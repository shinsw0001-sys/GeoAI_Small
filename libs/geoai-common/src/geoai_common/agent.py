"""에이전트 기반 클래스.

하위 클래스는 수행 의도별 처리기(on_request, on_result 등)만 구현한다.
기반 클래스는 메시지 기록, 온톨로지 버전 확인, 답신 발송을 맡는다.
"""

from __future__ import annotations

import logging

from .bus import MessageBus
from .messages import Envelope, Performative, RejectPayload
from .ontology import OntologyRegistry, compatible_versions
from .state import StateStore

log = logging.getLogger(__name__)


class BaseAgent:
    agent_type: str = ""  # 에이전트 클래스 IRI. 하위 클래스가 정한다.

    def __init__(self, agent_id: str, bus: MessageBus, store: StateStore, ontology: OntologyRegistry) -> None:
        self.agent_id = agent_id
        self.bus = bus
        self.store = store
        self.ontology = ontology

    async def start(self) -> None:
        await self.bus.subscribe(self.agent_id, self.handle)

    async def send(self, envelope: Envelope) -> None:
        await self.store.record_message(envelope)
        await self.bus.publish(envelope)

    async def handle(self, envelope: Envelope) -> Envelope | None:
        """메시지 하나를 처리한다. request 는 즉시 답신(accept/reject)을 돌려준다."""
        await self.store.record_message(envelope)
        if not compatible_versions(envelope.ontology_version, self.ontology.version):
            reply = envelope.reply(
                Performative.REJECT,
                self.agent_id,
                RejectPayload(
                    reason="ontology_version_mismatch",
                    detail=f"보낸 쪽 {envelope.ontology_version}, 받는 쪽 {self.ontology.version}",
                ),
                ontology_version=self.ontology.version,
            )
            await self.send(reply)
            return reply

        handler = {
            Performative.REQUEST: self.on_request,
            Performative.ACCEPT: self.on_accept,
            Performative.REJECT: self.on_reject,
            Performative.PROGRESS: self.on_progress,
            Performative.RESULT: self.on_result,
            Performative.FAILURE: self.on_failure,
            Performative.EVENT: self.on_event,
        }[envelope.performative]
        return await handler(envelope)

    async def on_request(self, envelope: Envelope) -> Envelope | None:
        reply = envelope.reply(
            Performative.REJECT,
            self.agent_id,
            RejectPayload(reason="not_supported", detail="이 에이전트는 요청을 받지 않습니다."),
        )
        await self.send(reply)
        return reply

    async def on_accept(self, envelope: Envelope) -> None:
        return None

    async def on_reject(self, envelope: Envelope) -> None:
        return None

    async def on_progress(self, envelope: Envelope) -> None:
        return None

    async def on_result(self, envelope: Envelope) -> None:
        return None

    async def on_failure(self, envelope: Envelope) -> None:
        return None

    async def on_event(self, envelope: Envelope) -> None:
        return None
