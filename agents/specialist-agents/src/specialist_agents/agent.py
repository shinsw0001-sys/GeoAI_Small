"""전문 에이전트.

공간정보 생성·가공, 전처리, 모델·학습, 분석·평가 에이전트는 같은 골격을 쓰고 온톨로지 선언만 다르다.

1. 수신: request 의 작업 유형이 자기 담당(gai:handledBy)인지, 본문이 규약에 맞는지 확인
2. 판단: 작업 유형을 실현하는 실행 작업 유형(gai:realizesTask)을 온톨로지에서 고르고,
         필요 입력(gai:requiresInputType)·필수 매개변수·입력 자산 존재를 확인한다. 안 되면 reject.
3. 제출: 실행 작업 명세를 만들어 작업 실행기에 넘긴다(accept 후 비동기). 진행률은 progress 로 보고한다.
4. 검증·보고: 산출물이 작업 유형의 산출 유형(gai:producesOutputType) 하위인지, 저장소에 실제로
         있는지 확인한 뒤 result 를 보낸다. 실행 실패나 검증 실패는 failure 로 보낸다.

에이전트는 연산하지 않는다. 처리 도구(geoai-tools)가 작업 실행기 위에서 연산한다.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid

from geoai_common import (
    GAI,
    BaseAgent,
    Envelope,
    FailurePayload,
    JobTypeInfo,
    Performative,
    ProgressPayload,
    RejectPayload,
    ResultPayload,
    TaskRequestPayload,
    TaskStatus,
    agent_id_for,
)
from geoai_common.jobs import JobError, JobRunner, JobSpec, job_paths
from geoai_common.objectstore import ObjectStore
from pydantic import ValidationError

log = logging.getLogger(__name__)

# 산출 유형 → 버킷 (objectstore 모듈 설명 참조). 앞에서부터 하위 클래스 여부를 본다.
OUTPUT_BUCKETS = [
    (str(GAI.Model), "models"),
    (str(GAI.LabelSet), "labels"),
    (str(GAI.ResultProduct), "results"),
    (str(GAI.DataAsset), "processed"),
]

# 실행 작업별 필수 매개변수. 진입점 이름 → [(매개변수 후보, 설명)]. 후보 중 하나만 있으면 된다.
REQUIRED_PARAMETERS: dict[str, list[tuple[tuple[str, ...], str]]] = {
    "rasterize-labels": [(("label_source",), "참조 벡터(GeoJSON) 저장소 주소")],
    "road-labels": [(("label_source",), "도로 중심선(LineString, 폭 속성) 또는 노면 벡터(GeoJSON) 저장소 주소")],
    "change-labels": [
        (
            ("label_sources", "label_source"),
            "시점별 건물 윤곽·토지피복 벡터 목록 또는 변화 벡터(GeoJSON) 저장소 주소",
        )
    ],
}


def _s3_values(value: object) -> list[str]:
    """매개변수 값 안의 저장소 주소(s3://...)를 모두 꺼낸다. 목록·사전 안쪽까지 본다."""
    if isinstance(value, str):
        return [value] if value.startswith("s3://") else []
    if isinstance(value, dict):
        return [u for v in value.values() for u in _s3_values(v)]
    if isinstance(value, list):
        return [u for v in value for u in _s3_values(v)]
    return []


def _job_id(task_id: str) -> str:
    base = re.sub(r"[^a-z0-9.-]", "-", task_id.lower()).strip("-.") or "task"
    return f"{base[:80]}.{uuid.uuid4().hex[:8]}"


class SpecialistAgent(BaseAgent):
    def __init__(self, *args, objects: ObjectStore, runner: JobRunner, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.objects = objects
        self.runner = runner
        matches = [c for c in self.ontology.subclasses(str(GAI.Agent)) if agent_id_for(c) == self.agent_id]
        if not matches:
            raise ValueError(f"온톨로지에 없는 에이전트 식별자: {self.agent_id}")
        self.agent_type = matches[0]
        self._running: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # 능력 명세
    # ------------------------------------------------------------------

    def handled_task_types(self) -> list[str]:
        return [t.iri for t in self.ontology.plannable_task_types() if t.handled_by == self.agent_type]

    def capabilities(self) -> dict:
        jobs = {j.realizes: j for j in self.ontology.job_types() if not j.domains}
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "ontology_version": self.ontology.version,
            "accepts": [
                {
                    "task_type": t,
                    "label": self.ontology.label(t),
                    "job_type": jobs[t].iri if t in jobs else None,
                    "entrypoint": jobs[t].entrypoint if t in jobs else None,
                }
                for t in self.handled_task_types()
            ],
            "jobs": [
                {"job_type": j.iri, "entrypoint": j.entrypoint, "realizes": j.realizes, "domains": list(j.domains)}
                for j in self.ontology.job_types()
                if j.realizes in self.handled_task_types()
            ],
            "running": sorted(self._running),
        }

    # ------------------------------------------------------------------
    # 접수 판단
    # ------------------------------------------------------------------

    async def _reject(self, envelope: Envelope, reason: str, detail: str) -> Envelope:
        reply = envelope.reply(
            Performative.REJECT,
            self.agent_id,
            RejectPayload(reason=reason, detail=detail),
            ontology_version=self.ontology.version,
        )
        await self.send(reply)
        return reply

    def _problems(self, task_type: str, job: JobTypeInfo, request: TaskRequestPayload) -> list[tuple[str, str]]:
        problems = []
        types = [str(a.asset_type) for a in request.inputs]
        for required in self.ontology.task_type(task_type).requires:
            if not any(self.ontology.is_subclass(t, required) for t in types):
                problems.append(("missing_input", f"필요 입력 '{self.ontology.label(required)}' 이(가) 없습니다."))
        if job.min_acquisitions > 1:
            dates = {a.acquired_at for a in request.inputs if a.acquired_at is not None}
            if len(dates) < job.min_acquisitions:
                problems.append(
                    (
                        "insufficient_acquisitions",
                        f"{self.ontology.label(job.iri)}에는 취득 일자가 다른 입력이 {job.min_acquisitions}시점 이상"
                        f" 필요합니다(현재 {len(dates)}시점).",
                    )
                )
        for names, desc in REQUIRED_PARAMETERS.get(job.entrypoint, []):
            name = next((n for n in names if request.parameters.get(n)), None)
            if name is None:
                problems.append(("missing_parameter", f"매개변수 {' 또는 '.join(names)}({desc})가 필요합니다."))
                continue
            for uri in _s3_values(request.parameters[name]):
                if not self.objects.exists(uri):
                    problems.append(("input_not_found", f"{name} 주소에 자료가 없습니다: {uri}"))
        for asset in request.inputs:
            if asset.storage_uri.startswith("s3://") and not self.objects.exists(asset.storage_uri):
                problems.append(("input_not_found", f"입력 자산이 저장소에 없습니다: {asset.storage_uri}"))
        return problems

    async def on_request(self, envelope: Envelope) -> Envelope | None:
        task_type = str(envelope.task_type) if envelope.task_type else ""
        if task_type not in self.handled_task_types():
            return await self._reject(
                envelope, "not_handled", f"{self.agent_id} 담당 작업 유형이 아닙니다: {task_type}"
            )
        if not envelope.task_id:
            return await self._reject(envelope, "invalid_payload", "task_id 가 없습니다.")
        try:
            request = TaskRequestPayload.model_validate(envelope.payload)
        except ValidationError as exc:
            return await self._reject(envelope, "invalid_payload", str(exc))
        job = self.ontology.job_for(task_type, [str(a.asset_type) for a in request.inputs], str(request.domain))
        if job is None:
            return await self._reject(envelope, "no_job_type", f"입력을 처리할 실행 작업 유형이 없습니다: {task_type}")
        problems = await asyncio.to_thread(self._problems, task_type, job, request)
        if problems:
            return await self._reject(envelope, problems[0][0], " / ".join(p[1] for p in problems))
        if envelope.task_id in self._running:
            return None  # 버스 재전달: 이미 실행 중

        job_id = _job_id(envelope.task_id)
        _, progress_uri, result_uri = job_paths(job_id)
        bucket = next(b for cls, b in OUTPUT_BUCKETS if self.ontology.is_subclass(job.output_type or "", cls))
        spec = JobSpec(
            job_id=job_id,
            job_type=job.iri,
            entrypoint=job.entrypoint,
            accelerator=job.accelerator,
            memory=job.memory,
            task_id=envelope.task_id,
            task_type=task_type,
            project_id=envelope.project_id,
            domain=str(request.domain),
            label_classes=[
                {"iri": c.iri, "label": c.label, "value": c.value, "geometry": c.geometry}
                for c in self.ontology.label_concepts(str(request.domain))
            ],
            inputs=request.inputs,
            parameters=request.parameters,
            completion_criteria=[c.model_dump() for c in request.completion_criteria],
            output_prefix=f"s3://{bucket}/{envelope.project_id}/{envelope.task_id}",
            progress_uri=progress_uri,
            result_uri=result_uri,
        )
        accept = envelope.reply(Performative.ACCEPT, self.agent_id, ontology_version=self.ontology.version)
        await self.send(accept)
        self._running[envelope.task_id] = asyncio.get_running_loop().create_task(self._execute(envelope, spec))
        return accept

    # ------------------------------------------------------------------
    # 실행·검증·보고
    # ------------------------------------------------------------------

    def _verify(self, task_type: str, result_outputs: list) -> str | None:
        produces = self.ontology.task_type(task_type).produces or str(GAI.DataAsset)
        if not any(self.ontology.is_subclass(str(o.asset_type), produces) for o in result_outputs):
            return f"산출물에 '{self.ontology.label(produces)}' 유형이 없습니다."
        missing = [o.storage_uri for o in result_outputs if not self.objects.exists(o.storage_uri)]
        if missing:
            return f"저장소에 없는 산출물: {', '.join(missing)}"
        return None

    async def _execute(self, envelope: Envelope, spec: JobSpec) -> None:
        label = self.ontology.label(spec.job_type)

        async def report(percent: float, note: str) -> None:
            await self.send(
                envelope.reply(
                    Performative.PROGRESS,
                    self.agent_id,
                    ProgressPayload(status=TaskStatus.RUNNING.value, percent=percent, note=f"{label}: {note}"),
                    ontology_version=self.ontology.version,
                )
            )

        try:
            await report(0.0, "실행 작업 제출")
            result = await self.runner.run(spec, on_progress=report)
            problem = await asyncio.to_thread(self._verify, spec.task_type, result.outputs)
            if problem:
                raise JobError("output_verification_failed", problem)
            reply = envelope.reply(
                Performative.RESULT,
                self.agent_id,
                ResultPayload(outputs=result.outputs, metrics=result.metrics),
                ontology_version=self.ontology.version,
            )
        except JobError as exc:
            log.warning("실행 작업 실패 %s: %s", spec.job_id, exc)
            reply = envelope.reply(
                Performative.FAILURE,
                self.agent_id,
                FailurePayload(reason=exc.reason, detail=exc.detail),
                ontology_version=self.ontology.version,
            )
        except Exception as exc:  # noqa: BLE001 - 실패는 반드시 상위에 보고한다
            log.exception("실행 작업 처리 중 오류 %s", spec.job_id)
            reply = envelope.reply(
                Performative.FAILURE,
                self.agent_id,
                FailurePayload(reason="agent_error", detail=f"{type(exc).__name__}: {exc}"),
                ontology_version=self.ontology.version,
            )
        finally:
            self._running.pop(spec.task_id, None)
        await self.send(reply)

    async def wait_idle(self) -> None:
        """실행 중인 실행 작업이 모두 끝날 때까지 기다린다(시험·종료용)."""
        while self._running:
            await asyncio.gather(*list(self._running.values()), return_exceptions=True)
