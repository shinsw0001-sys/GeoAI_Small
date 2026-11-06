"""온톨로지 레지스트리.

핵심 온톨로지, 형상 제약, 분야 온톨로지를 읽어 에이전트에 다음 기능을 준다.

- 클래스 포함 관계 판정 (rdfs:subClassOf 추이 폐포)
- 계획 규칙 조회 (gai:requiresInputType, gai:producesOutputType, gai:handledBy)
- 작업 요청 본문의 형상 제약 검증 (SHACL)

디렉터리 구조는 저장소의 `ontology/` 를 따른다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from pyshacl import validate
from rdflib import OWL, RDF, RDFS, SH, Graph, Literal, Namespace, URIRef
from rdflib.namespace import XSD

from .messages import TaskRequestPayload, Violation

GAI = Namespace("https://ontology.geoai.local/core#")
CORE_ONTOLOGY_IRI = URIRef("https://ontology.geoai.local/core")
REQUEST_NS = Namespace("urn:geoai:request:")


@dataclass(frozen=True)
class TaskTypeInfo:
    iri: str
    label: str
    requires: tuple[str, ...]
    produces: str | None
    handled_by: str | None


def agent_id_for(agent_class_iri: str) -> str:
    """에이전트 클래스 IRI를 에이전트 식별자로 바꾼다. 예: gai:PreprocessingAgent → preprocessing-agent"""
    local = agent_class_iri.rsplit("#", 1)[-1]
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "-", local).lower()


class OntologyRegistry:
    def __init__(self, ontology: Graph, shapes: Graph) -> None:
        self.graph = ontology
        self.shapes = shapes
        version = ontology.value(CORE_ONTOLOGY_IRI, OWL.versionInfo)
        if version is None:
            raise ValueError("핵심 온톨로지에 owl:versionInfo 가 없습니다.")
        self.version = str(version)
        self._superclasses = cache(self._superclasses_uncached)

    # ------------------------------------------------------------------
    # 적재
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, root: str | Path, extra_domain_dirs: list[str | Path] | None = None) -> OntologyRegistry:
        """`root` 는 core/, domains/ 를 가진 온톨로지 디렉터리."""
        root = Path(root)
        ontology = Graph()
        ontology.parse(root / "core" / "geoai-core.ttl", format="turtle")
        shapes = Graph()
        shapes.parse(root / "core" / "geoai-core-shapes.ttl", format="turtle")
        domain_dirs = [root / "domains", *(Path(d) for d in extra_domain_dirs or [])]
        for directory in domain_dirs:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.ttl")):
                if path.name.startswith("_"):
                    continue
                ontology.parse(path, format="turtle")
        return cls(ontology, shapes)

    # ------------------------------------------------------------------
    # 클래스 관계
    # ------------------------------------------------------------------

    def _superclasses_uncached(self, iri: str) -> frozenset[str]:
        node = URIRef(iri)
        return frozenset(str(c) for c in self.graph.transitive_objects(node, RDFS.subClassOf))

    def is_class(self, iri: str) -> bool:
        return (URIRef(iri), RDF.type, OWL.Class) in self.graph

    def is_subclass(self, iri: str, ancestor: str) -> bool:
        """반사적 포함 관계. iri ⊑ ancestor 이면 참."""
        return str(ancestor) in self._superclasses(str(iri))

    def subclasses(self, ancestor: str) -> list[str]:
        node = URIRef(ancestor)
        return sorted(str(c) for c in self.graph.transitive_subjects(RDFS.subClassOf, node) if c != node)

    def label(self, iri: str, lang: str = "ko") -> str:
        for value in self.graph.objects(URIRef(iri), RDFS.label):
            if isinstance(value, Literal) and value.language == lang:
                return str(value)
        return iri.rsplit("#", 1)[-1]

    def domains(self) -> list[str]:
        return sorted(str(d) for d in self.graph.subjects(RDF.type, GAI.Domain))

    # ------------------------------------------------------------------
    # 계획 규칙
    # ------------------------------------------------------------------

    def task_type(self, iri: str) -> TaskTypeInfo:
        node = URIRef(iri)
        produces = self.graph.value(node, GAI.producesOutputType)
        handled_by = self.graph.value(node, GAI.handledBy)
        return TaskTypeInfo(
            iri=iri,
            label=self.label(iri),
            requires=tuple(sorted(str(c) for c in self.graph.objects(node, GAI.requiresInputType))),
            produces=str(produces) if produces else None,
            handled_by=str(handled_by) if handled_by else None,
        )

    def plannable_task_types(self, ancestor: str = str(GAI.StudioTask)) -> list[TaskTypeInfo]:
        """세 계획 속성을 모두 갖춘 작업 클래스."""
        infos = [self.task_type(c) for c in self.subclasses(ancestor)]
        return [i for i in infos if i.requires and i.produces and i.handled_by]

    def producers_of(self, asset_class: str, ancestor: str = str(GAI.StudioTask)) -> list[TaskTypeInfo]:
        """산출 유형이 asset_class 의 하위(또는 같은) 클래스인 작업 클래스."""
        return [i for i in self.plannable_task_types(ancestor) if self.is_subclass(i.produces, asset_class)]

    # ------------------------------------------------------------------
    # 형상 제약 검증
    # ------------------------------------------------------------------

    def request_graph(self, task_type: str, payload: TaskRequestPayload) -> Graph:
        """작업 요청 본문을 RDF 그래프로 바꾼다."""
        g = Graph()
        task = REQUEST_NS["task"]
        g.add((task, RDF.type, URIRef(task_type)))
        g.add((task, GAI.inDomain, URIRef(str(payload.domain))))
        for asset in payload.inputs:
            node = REQUEST_NS[f"asset/{asset.asset_id}"]
            g.add((task, GAI.hasInput, node))
            g.add((node, RDF.type, URIRef(str(asset.asset_type))))
            g.add((node, GAI.storageUri, Literal(asset.storage_uri, datatype=XSD.anyURI)))
            if asset.crs is not None:
                g.add((node, GAI.crs, Literal(asset.crs)))
            if asset.ground_sample_distance is not None:
                g.add(
                    (node, GAI.groundSampleDistance, Literal(str(asset.ground_sample_distance), datatype=XSD.decimal))
                )
        for idx, crit in enumerate(payload.completion_criteria):
            node = REQUEST_NS[f"criterion/{idx}"]
            g.add((task, GAI.hasCompletionCriterion, node))
            g.add((node, RDF.type, GAI.CompletionCriterion))
            g.add((node, GAI.metric, Literal(crit.metric)))
            g.add((node, GAI.operator, Literal(crit.operator)))
            g.add((node, GAI.threshold, Literal(str(crit.threshold), datatype=XSD.decimal)))
        return g

    def validate_request(self, task_type: str, payload: TaskRequestPayload) -> list[Violation]:
        """형상 제약 위반 목록. 비어 있으면 통과."""
        violations: list[Violation] = []
        for asset in payload.inputs:
            if not self.is_subclass(str(asset.asset_type), str(GAI.DataAsset)):
                violations.append(
                    Violation(focus=asset.asset_id, message=f"등록되지 않은 데이터 자산 유형: {asset.asset_type}")
                )
        if violations:
            return violations

        # 요청 그래프에 온톨로지를 합쳐 검증한다. 분야 개체 선언과 클래스 계층이 함께 있어야 한다.
        data = self.graph + self.request_graph(task_type, payload)
        conforms, report, _ = validate(
            data,
            shacl_graph=self.shapes,
            # sh:class·sh:targetClass 는 rdfs:subClassOf 를 스스로 따라간다.
            # rdfs 추론을 켜면 rdfs:range 때문에 미등록 분야도 gai:Domain 으로 추론되므로 끈다.
            inference="none",
            abort_on_first=False,
            allow_warnings=True,
        )
        if conforms:
            return []
        for result in report.subjects(RDF.type, SH.ValidationResult):
            focus = report.value(result, SH.focusNode)
            path = report.value(result, SH.resultPath)
            message = report.value(result, SH.resultMessage)
            violations.append(
                Violation(
                    focus=str(focus).removeprefix(str(REQUEST_NS)) if focus else None,
                    path=str(path) if path else None,
                    message=str(message) if message else "형상 제약 위반",
                )
            )
        return violations


def compatible_versions(a: str, b: str) -> bool:
    """주 버전(major)이 같으면 호환으로 본다."""
    return a.split(".", 1)[0] == b.split(".", 1)[0]
