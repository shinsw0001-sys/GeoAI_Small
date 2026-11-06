"""Studio Agent.

GeoAI Studio의 대표 에이전트. 처리 흐름:

1. request 수신 → 작업 유형 확인(gai:StudioTask 하위) → 형상 제약 검증(SHACL)
2. 온톨로지 규칙으로 하위 작업 계획 → 상위·하위 작업을 상태 저장소에 기록 → accept 답신
3. 선행 작업이 없는 하위 작업부터 담당 에이전트에 request 발송 (Planned → Running)
4. 하위 작업 result 수신 → Succeeded 기록 → 선행 조건이 다 찬 후속 작업 발송
5. 모든 하위 작업 완료 → 상위 작업 Succeeded, 요청자에게 result 발송
   하위 작업 failure/reject → 상위 작업 Failed, 요청자에게 failure 발송

에이전트는 상태를 두지 않는다. 모든 상태는 프로젝트 상태 저장소에 있고, 상태 전이는
비교 후 교환으로만 하므로 복제본을 여러 개 띄워도 같은 하위 작업을 두 번 보내지 않는다.
"""

from __future__ import annotations

import logging
import uuid

from geoai_common import (
    GAI,
    AcceptPayload,
    BaseAgent,
    DataAssetRef,
    Envelope,
    FailurePayload,
    Performative,
    PlanStep,
    ProgressPayload,
    RejectPayload,
    ResultPayload,
    TaskRecord,
    TaskRequestPayload,
    TaskStatus,
)
from pydantic import ValidationError

from .planner import Planner, PlanningError

log = logging.getLogger(__name__)

ACTIVE = {TaskStatus.PLANNED, TaskStatus.RUNNING}


class StudioAgent(BaseAgent):
    agent_type = str(GAI.StudioAgent)

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.planner = Planner(self.ontology)

    # ------------------------------------------------------------------
    # 능력 명세 (agent card)
    # ------------------------------------------------------------------

    def capabilities(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "ontology_version": self.ontology.version,
            "accepts": [
                {
                    "task_type": t.iri,
                    "label": t.label,
                    "requires": list(t.requires),
                    "produces": t.produces,
                    "handled_by": t.handled_by,
                }
                for t in self.ontology.plannable_task_types()
            ],
            "domains": [{"iri": d, "label": self.ontology.label(d)} for d in self.ontology.domains()],
        }

    # ------------------------------------------------------------------
    # 요청 접수
    # ------------------------------------------------------------------

    async def _reject(self, envelope: Envelope, payload: RejectPayload) -> Envelope:
        reply = envelope.reply(Performative.REJECT, self.agent_id, payload, ontology_version=self.ontology.version)
        await self.send(reply)
        return reply

    async def on_request(self, envelope: Envelope) -> Envelope:
        task_type = str(envelope.task_type) if envelope.task_type else ""
        if (
            not task_type
            or not self.ontology.is_subclass(task_type, str(GAI.StudioTask))
            or task_type == str(GAI.StudioTask)
        ):
            return await self._reject(
                envelope,
                RejectPayload(reason="invalid_task_type", detail=f"Studio 작업 유형이 아닙니다: {task_type or '없음'}"),
            )
        try:
            request = TaskRequestPayload.model_validate(envelope.payload)
        except ValidationError as exc:
            return await self._reject(envelope, RejectPayload(reason="invalid_payload", detail=str(exc)))

        violations = self.ontology.validate_request(task_type, request)
        if violations:
            return await self._reject(
                envelope,
                RejectPayload(reason="shape_violation", detail="온톨로지 형상 제약 위반", violations=violations),
            )

        try:
            steps = self.planner.plan(task_type, request.inputs)
        except PlanningError as exc:
            return await self._reject(envelope, RejectPayload(reason="unplannable", detail=str(exc)))

        parent_id = envelope.task_id or str(uuid.uuid4())
        ids = {s.key: f"{parent_id}.{s.key}" for s in steps}
        target_key = steps[-1].key
        parent = TaskRecord(
            task_id=parent_id,
            project_id=envelope.project_id,
            conversation_id=envelope.conversation_id,
            task_type=task_type,
            status=TaskStatus.PLANNED,
            agent=self.agent_id,
            requester=envelope.sender,
            request_message_id=envelope.message_id,
            input_assets=request.inputs,
            request=request.model_dump(mode="json"),
        )
        children = [
            TaskRecord(
                task_id=ids[s.key],
                parent_id=parent_id,
                project_id=envelope.project_id,
                conversation_id=envelope.conversation_id,
                task_type=s.task_type,
                status=TaskStatus.PLANNED,
                agent=s.agent,
                depends_on=[ids[d] for d in s.depends_on],
                input_assets=s.input_assets,
                input_from=[ids[d] for d in s.depends_on],
                request={
                    "domain": str(request.domain),
                    "parameters": request.parameters,
                    # 완료 기준은 요청 작업(마지막 단계)에만 붙는다.
                    "completion_criteria": [c.model_dump() for c in request.completion_criteria]
                    if s.key == target_key
                    else [],
                },
            )
            for s in steps
        ]
        try:
            await self.store.create_tasks([parent, *children])
        except KeyError:
            return await self._reject(
                envelope, RejectPayload(reason="duplicate_task", detail=f"이미 접수된 작업: {parent_id}")
            )

        accept = envelope.reply(
            Performative.ACCEPT,
            self.agent_id,
            AcceptPayload(
                task_id=parent_id,
                plan=[
                    PlanStep(task_id=c.task_id, task_type=c.task_type, agent=c.agent, depends_on=c.depends_on)
                    for c in children
                ],
            ),
            task_id=parent_id,
            ontology_version=self.ontology.version,
        )
        await self.send(accept)
        await self._dispatch_ready(parent_id, cause=envelope)
        return accept

    # ------------------------------------------------------------------
    # 하위 작업 배정
    # ------------------------------------------------------------------

    async def _dispatch_ready(self, parent_id: str, cause: Envelope) -> None:
        parent = await self.store.get_task(parent_id)
        if parent is None or parent.status not in ACTIVE:
            return
        children = await self.store.list_children(parent_id)
        by_id = {c.task_id: c for c in children}
        for child in children:
            if child.status is not TaskStatus.PLANNED:
                continue
            if not all(by_id[d].status is TaskStatus.SUCCEEDED for d in child.depends_on):
                continue
            claimed = await self.store.transition(child.task_id, {TaskStatus.PLANNED}, TaskStatus.RUNNING)
            if claimed is None:
                continue  # 다른 복제본이 먼저 보냈다
            inputs: list[DataAssetRef] = [*child.input_assets]
            for dep in child.input_from:
                inputs.extend(by_id[dep].outputs)
            payload = TaskRequestPayload(
                domain=child.request["domain"],
                inputs=inputs,
                completion_criteria=child.request.get("completion_criteria", []),
                parameters=child.request.get("parameters", {}),
            )
            request = Envelope(
                conversation_id=parent.conversation_id,
                causation_id=cause.message_id,
                performative=Performative.REQUEST,
                sender=self.agent_id,
                recipient=child.agent,
                project_id=parent.project_id,
                task_id=child.task_id,
                task_type=child.task_type,
                ontology_version=self.ontology.version,
                traceparent=cause.traceparent,
                payload=payload.model_dump(mode="json"),
            )
            await self.store.transition(
                child.task_id, {TaskStatus.RUNNING}, TaskStatus.RUNNING, request_message_id=request.message_id
            )
            await self.send(request)
        await self.store.transition(parent_id, {TaskStatus.PLANNED}, TaskStatus.RUNNING)

    async def _child(self, envelope: Envelope) -> tuple[TaskRecord, TaskRecord] | None:
        if not envelope.task_id:
            return None
        child = await self.store.get_task(envelope.task_id)
        if child is None or child.parent_id is None:
            log.warning("알 수 없는 하위 작업 보고를 무시합니다: %s", envelope.task_id)
            return None
        parent = await self.store.get_task(child.parent_id)
        return (parent, child) if parent else None

    def _to_requester(self, parent: TaskRecord, cause: Envelope, performative: Performative, payload) -> Envelope:  # type: ignore[no-untyped-def]
        return Envelope(
            conversation_id=parent.conversation_id,
            correlation_id=parent.request_message_id,
            causation_id=cause.message_id,
            performative=performative,
            sender=self.agent_id,
            recipient=parent.requester,
            project_id=parent.project_id,
            task_id=parent.task_id,
            task_type=parent.task_type,
            ontology_version=self.ontology.version,
            traceparent=cause.traceparent,
            payload=payload.model_dump(mode="json"),
        )

    # ------------------------------------------------------------------
    # 하위 작업 보고 처리
    # ------------------------------------------------------------------

    async def on_accept(self, envelope: Envelope) -> None:
        return None  # 하위 에이전트 접수는 기록만 한다 (BaseAgent.handle)

    async def on_progress(self, envelope: Envelope) -> None:
        pair = await self._child(envelope)
        if pair is None:
            return
        parent, child = pair
        progress = ProgressPayload.model_validate(envelope.payload)
        note = f"[{child.task_id} {self.ontology.label(child.task_type)}] {progress.note}".strip()
        await self.send(
            self._to_requester(
                parent,
                envelope,
                Performative.PROGRESS,
                ProgressPayload(status=TaskStatus.RUNNING.value, percent=await self._percent(parent), note=note),
            )
        )

    async def _percent(self, parent: TaskRecord) -> float:
        children = await self.store.list_children(parent.task_id)
        done = sum(c.status is TaskStatus.SUCCEEDED for c in children)
        return round(100 * done / len(children), 1) if children else 0.0

    async def on_result(self, envelope: Envelope) -> None:
        pair = await self._child(envelope)
        if pair is None:
            return
        parent, child = pair
        result = ResultPayload.model_validate(envelope.payload)
        updated = await self.store.transition(
            child.task_id, {TaskStatus.RUNNING}, TaskStatus.SUCCEEDED, outputs=result.outputs, metrics=result.metrics
        )
        if updated is None:
            return  # 중복 보고
        children = await self.store.list_children(parent.task_id)
        if all(c.status is TaskStatus.SUCCEEDED for c in children):
            final = children[-1]
            done = await self.store.transition(
                parent.task_id, ACTIVE, TaskStatus.SUCCEEDED, outputs=final.outputs, metrics=final.metrics
            )
            if done is not None:
                await self.send(
                    self._to_requester(
                        parent,
                        envelope,
                        Performative.RESULT,
                        ResultPayload(outputs=final.outputs, metrics=final.metrics),
                    )
                )
            return
        await self._dispatch_ready(parent.task_id, cause=envelope)

    async def _fail(self, envelope: Envelope, reason: str, detail: str) -> None:
        pair = await self._child(envelope)
        if pair is None:
            return
        parent, child = pair
        await self.store.transition(child.task_id, ACTIVE, TaskStatus.FAILED, detail=detail)
        failed = await self.store.transition(
            parent.task_id, ACTIVE, TaskStatus.FAILED, detail=f"{child.task_id}: {detail}"
        )
        if failed is not None:
            await self.send(
                self._to_requester(
                    parent,
                    envelope,
                    Performative.FAILURE,
                    FailurePayload(reason=reason, detail=f"하위 작업 {child.task_id} 실패: {detail}"),
                )
            )

    async def on_failure(self, envelope: Envelope) -> None:
        failure = FailurePayload.model_validate(envelope.payload)
        await self._fail(envelope, failure.reason, failure.detail)

    async def on_reject(self, envelope: Envelope) -> None:
        reject = RejectPayload.model_validate(envelope.payload)
        await self._fail(envelope, f"subtask_rejected:{reject.reason}", reject.detail)
