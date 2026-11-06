"""시험용 도구: 하위 에이전트 모의 구현.

하위 에이전트(전처리, 모델·학습, 분석·평가)가 아직 없으므로, 받은 작업 유형의 산출
유형으로 가짜 산출 자산을 만들어 result 를 돌려준다. 통합 시험과 개발 환경에서 쓴다.
"""

from __future__ import annotations

from .agent import BaseAgent
from .messages import DataAssetRef, Envelope, FailurePayload, Performative, ResultPayload, TaskRequestPayload


class EchoSubAgent(BaseAgent):
    def __init__(self, *args, fail_on: set[str] | None = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.fail_on = fail_on or set()
        self.received: list[Envelope] = []

    async def on_request(self, envelope: Envelope) -> None:
        self.received.append(envelope)
        request = TaskRequestPayload.model_validate(envelope.payload)
        task_type = str(envelope.task_type)
        if task_type in self.fail_on:
            await self.send(
                envelope.reply(
                    Performative.FAILURE, self.agent_id, FailurePayload(reason="tool_error", detail="모의 실패")
                )
            )
            return
        produces = self.ontology.task_type(task_type).produces
        output = DataAssetRef(
            asset_id=f"{envelope.task_id}.out",
            asset_type=produces,
            storage_uri=f"s3://results/{envelope.task_id}/output",
            crs=next((a.crs for a in request.inputs if a.crs), None),
        )
        await self.send(envelope.reply(Performative.ACCEPT, self.agent_id))
        await self.send(
            envelope.reply(Performative.RESULT, self.agent_id, ResultPayload(outputs=[output], metrics={"mock": 1.0}))
        )
