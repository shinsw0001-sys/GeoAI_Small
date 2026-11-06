"""온톨로지 규칙 기반 계획기.

요청 작업(목표)의 필요 입력 유형을 하나씩 확인하고, 요청 입력으로 채워지지 않는
유형은 그 유형을 산출하는 작업을 찾아 앞에 붙인다(역방향 연쇄, backward chaining).
같은 유형을 두 번 만들지 않도록 이미 계획한 단계의 산출을 재사용한다.
산출 후보가 여럿이면 총 단계 수가 가장 적은 후보를, 같으면 IRI 순으로 고른다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from geoai_common import DataAssetRef, OntologyRegistry, agent_id_for

MAX_DEPTH = 8


class PlanningError(Exception):
    pass


@dataclass
class Step:
    key: str  # 계획 안에서만 쓰는 단계 이름 (s0, s1 ...)
    task_type: str
    agent: str
    depends_on: list[str] = field(default_factory=list)  # 선행 단계 key
    input_assets: list[DataAssetRef] = field(default_factory=list)  # 요청 입력 중 이 단계가 쓰는 것
    produces: str = ""


class Planner:
    def __init__(self, ontology: OntologyRegistry) -> None:
        self.ontology = ontology

    def plan(self, target_task_type: str, inputs: list[DataAssetRef]) -> list[Step]:
        """위상 정렬된 단계 목록. 마지막 단계가 요청 작업이다."""
        steps, _ = self._plan_task(target_task_type, inputs, [], {}, depth=0, visiting=frozenset())
        return steps

    # 단계 목록, 산출 유형→단계 key 사상
    def _plan_task(
        self,
        task_type: str,
        inputs: list[DataAssetRef],
        steps: list[Step],
        produced: dict[str, str],
        depth: int,
        visiting: frozenset[str],
    ) -> tuple[list[Step], dict[str, str]]:
        if depth > MAX_DEPTH:
            raise PlanningError(f"계획 깊이가 {MAX_DEPTH}를 넘었습니다: {task_type}")
        if task_type in visiting:
            raise PlanningError(f"작업 유형 순환: {task_type}")
        info = self.ontology.task_type(task_type)
        if not (info.requires and info.produces and info.handled_by):
            raise PlanningError(f"계획 규칙(필요 입력·산출·담당)이 선언되지 않은 작업 유형: {task_type}")

        steps = list(steps)
        produced = dict(produced)
        depends_on: list[str] = []
        used_inputs: list[DataAssetRef] = []

        for required in info.requires:
            direct = [a for a in inputs if self.ontology.is_subclass(str(a.asset_type), required)]
            if direct:
                used_inputs.extend(a for a in direct if a not in used_inputs)
                continue
            reuse = next((k for cls, k in produced.items() if self.ontology.is_subclass(cls, required)), None)
            if reuse is not None:
                if reuse not in depends_on:
                    depends_on.append(reuse)
                continue
            steps, produced, key = self._best_producer(required, inputs, steps, produced, depth, visiting | {task_type})
            depends_on.append(key)

        key = f"s{len(steps)}"
        steps.append(
            Step(
                key=key,
                task_type=task_type,
                agent=agent_id_for(info.handled_by),
                depends_on=depends_on,
                input_assets=used_inputs,
                produces=info.produces,
            )
        )
        produced[info.produces] = key
        return steps, produced

    def _best_producer(
        self,
        required: str,
        inputs: list[DataAssetRef],
        steps: list[Step],
        produced: dict[str, str],
        depth: int,
        visiting: frozenset[str],
    ) -> tuple[list[Step], dict[str, str], str]:
        candidates = []
        errors = []
        for producer in sorted(self.ontology.producers_of(required), key=lambda p: p.iri):
            if producer.iri in visiting:
                continue
            try:
                new_steps, new_produced = self._plan_task(producer.iri, inputs, steps, produced, depth + 1, visiting)
            except PlanningError as exc:
                errors.append(str(exc))
                continue
            candidates.append((len(new_steps), producer.iri, new_steps, new_produced))
        if not candidates:
            label = self.ontology.label(required)
            detail = f" ({'; '.join(errors)})" if errors else ""
            raise PlanningError(f"'{label}' 입력이 없고 이를 만들 수 있는 작업도 없습니다{detail}")
        _, _, new_steps, new_produced = min(candidates, key=lambda c: (c[0], c[1]))
        return new_steps, new_produced, new_steps[-1].key
