"""에이전트 간 메시지 규약.

모든 메시지는 봉투(Envelope)와 본문(payload)으로 구성된다. 봉투는 전송·추적에
필요한 정보만 담고, 본문은 수행 의도(performative)별 모델을 따른다.
본문 안의 모든 유형 값(task_type, asset_type, domain)은 온톨로지 IRI다.

JSON Schema 산출본은 `schemas/` 디렉터리에 있으며 `python -m geoai_common.messages`
로 다시 만든다. 시험이 산출본과 이 모듈의 일치를 확인한다.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import AnyUrl, BaseModel, ConfigDict, Field

PROTOCOL_VERSION = "0.1"

AGENT_ID_PATTERN = r"^[a-z][a-z0-9-]*$"


class Performative(StrEnum):
    """수행 의도. 화행 이론 기반 에이전트 통신 언어(FIPA ACL)의 부분집합."""

    REQUEST = "request"  # 작업 요청
    ACCEPT = "accept"  # 요청 접수
    REJECT = "reject"  # 요청 거절 (검증 실패, 계획 불가)
    PROGRESS = "progress"  # 진행 보고
    RESULT = "result"  # 완료 결과
    FAILURE = "failure"  # 실행 실패
    EVENT = "event"  # 수신자 없는 상태 알림


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid.uuid4())


class Envelope(BaseModel):
    """메시지 봉투."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: str = Field(default=PROTOCOL_VERSION, description="메시지 규약 버전")
    message_id: str = Field(default_factory=_new_id, description="메시지 고유 식별자")
    conversation_id: str = Field(
        default_factory=_new_id,
        description="최상위 요청 하나의 생애 전체를 묶는 식별자. 하위 작업 메시지도 같은 값을 쓴다.",
    )
    correlation_id: str | None = Field(default=None, description="이 메시지가 답하는 요청 메시지의 message_id")
    causation_id: str | None = Field(default=None, description="이 메시지를 직접 일으킨 메시지의 message_id (추적용)")
    performative: Performative
    sender: str = Field(pattern=AGENT_ID_PATTERN, description="보내는 에이전트 식별자")
    recipient: str | None = Field(
        default=None,
        pattern=AGENT_ID_PATTERN,
        description="받는 에이전트 식별자. event는 비워 둔다.",
    )
    project_id: str = Field(min_length=1, description="프로젝트 식별자")
    task_id: str | None = Field(default=None, description="이 메시지가 다루는 작업 식별자")
    task_type: AnyUrl | None = Field(default=None, description="작업 유형 온톨로지 IRI")
    ontology_version: str = Field(description="보낸 쪽이 참조한 핵심 온톨로지 버전")
    created_at: datetime = Field(default_factory=_now)
    deadline: datetime | None = Field(default=None, description="응답 또는 완료 기한")
    traceparent: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$",
        description="W3C Trace Context 추적 문맥. 버스 전송 시 메시지 헤더로도 싣는다(OpenTelemetry 호환).",
    )
    payload: dict[str, Any] = Field(default_factory=dict)

    def reply(
        self,
        performative: Performative,
        sender: str,
        payload: BaseModel | dict[str, Any] | None = None,
        **overrides: Any,
    ) -> Envelope:
        """이 메시지에 대한 답신을 만든다. 추적 식별자를 이어받는다."""
        body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else (payload or {})
        fields: dict[str, Any] = {
            "conversation_id": self.conversation_id,
            "correlation_id": self.message_id,
            "causation_id": self.message_id,
            "performative": performative,
            "sender": sender,
            "recipient": self.sender,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "ontology_version": self.ontology_version,
            "traceparent": self.traceparent,
            "payload": body,
        }
        fields.update(overrides)
        return Envelope(**fields)


# ---------------------------------------------------------------------------
# 본문 모델
# ---------------------------------------------------------------------------


class DataAssetRef(BaseModel):
    """데이터·모델 저장소의 자산 참조."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    asset_type: AnyUrl = Field(description="데이터 자산 클래스 IRI (gai:DataAsset 하위)")
    storage_uri: str = Field(min_length=1, description="예: s3://raw/site-a/ortho.tif")
    crs: str | None = Field(default=None, description="예: EPSG:5186")
    ground_sample_distance: float | None = Field(default=None, description="미터 단위")


class CompletionCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1, description="예: mIoU, F1")
    operator: str = Field(description=">=, <=, >, <, == 중 하나")
    threshold: float


class TaskRequestPayload(BaseModel):
    """performative=request 본문."""

    model_config = ConfigDict(extra="forbid")

    domain: AnyUrl = Field(description="분야 개체 IRI (gai:Domain)")
    inputs: list[DataAssetRef] = Field(min_length=1)
    completion_criteria: list[CompletionCriterion] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_type: AnyUrl
    agent: str
    depends_on: list[str] = Field(default_factory=list)


class AcceptPayload(BaseModel):
    """performative=accept 본문. 접수한 작업과 하위 작업 계획."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    plan: list[PlanStep] = Field(default_factory=list)


class Violation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus: str | None = None
    path: str | None = None
    message: str


class RejectPayload(BaseModel):
    """performative=reject 본문."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(description="거절 사유 코드. 예: invalid_task_type, shape_violation, unplannable")
    detail: str = ""
    violations: list[Violation] = Field(default_factory=list)


class ProgressPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(description="작업 상태 이름 (gai:TaskStatus 개체 지역명)")
    percent: float | None = Field(default=None, ge=0, le=100)
    note: str = ""


class ResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outputs: list[DataAssetRef] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class FailurePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    detail: str = ""


PAYLOAD_MODELS: dict[Performative, type[BaseModel]] = {
    Performative.REQUEST: TaskRequestPayload,
    Performative.ACCEPT: AcceptPayload,
    Performative.REJECT: RejectPayload,
    Performative.PROGRESS: ProgressPayload,
    Performative.RESULT: ResultPayload,
    Performative.FAILURE: FailurePayload,
}


def parse_payload(envelope: Envelope) -> BaseModel | dict[str, Any]:
    """수행 의도에 맞는 본문 모델로 해석한다. event 본문은 사전 그대로 둔다."""
    model = PAYLOAD_MODELS.get(envelope.performative)
    return model.model_validate(envelope.payload) if model else envelope.payload


def json_schemas() -> dict[str, dict[str, Any]]:
    """`schemas/` 디렉터리에 둘 JSON Schema 산출본."""
    out = {"envelope.schema.json": Envelope.model_json_schema()}
    for perf, model in PAYLOAD_MODELS.items():
        out[f"payload-{perf.value}.schema.json"] = model.model_json_schema()
    return out


def _write_schemas(target: str) -> None:
    from pathlib import Path

    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    for name, schema in json_schemas().items():
        (root / name).write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    _write_schemas(sys.argv[1] if len(sys.argv) > 1 else "schemas")
