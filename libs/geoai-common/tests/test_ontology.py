from pathlib import Path

import pytest
from geoai_common import GAI, OntologyRegistry, TaskRequestPayload, agent_id_for

ROOT = Path(__file__).resolve().parents[3]
EX = "https://ontology.geoai.local/domain/example#"


@pytest.fixture(scope="module")
def onto() -> OntologyRegistry:
    return OntologyRegistry.load(ROOT / "ontology", [ROOT / "ontology" / "examples"])


def request(**overrides):
    body = {
        "domain": EX + "Domain",
        "inputs": [
            {
                "asset_id": "a1",
                "asset_type": str(GAI.AerialImage),
                "storage_uri": "s3://raw/a1.tif",
                "crs": "EPSG:5186",
                "ground_sample_distance": 0.25,
            }
        ],
        "completion_criteria": [{"metric": "mIoU", "operator": ">=", "threshold": 0.7}],
    }
    body.update(overrides)
    return TaskRequestPayload.model_validate(body)


def test_version_and_domains(onto):
    assert onto.version == "0.1.0"
    year1 = [
        f"https://ontology.geoai.local/domain/{d}#Domain"
        for d in ("building-extraction", "change-detection", "land-cover-classification", "road-extraction")
    ]
    assert onto.domains() == sorted([*year1, EX + "Domain"])
    for d in year1:
        assert onto.validate_request(str(GAI.ModelTrainingTask), request(domain=d)) == []


def test_template_is_not_loaded(onto):
    assert not any("CHANGE-ME" in d for d in onto.domains())


def test_subclass_closure(onto):
    assert onto.is_subclass(str(GAI.AerialImage), str(GAI.DataAsset))
    assert onto.is_subclass(str(GAI.AerialImage), "http://www.w3.org/ns/prov#Entity")
    assert onto.is_subclass(str(GAI.EvaluationReport), str(GAI.ResultProduct))
    assert not onto.is_subclass(str(GAI.Model), str(GAI.RawData))


def test_every_concrete_studio_task_is_plannable(onto):
    # 하위 클래스가 있는 묶음 클래스(예: AnalysisEvaluationTask)는 계획 대상이 아니다.
    declared = {c for c in onto.subclasses(str(GAI.StudioTask)) if not onto.subclasses(c)}
    plannable = {t.iri for t in onto.plannable_task_types()}
    assert declared == plannable
    for t in onto.plannable_task_types():
        assert onto.is_subclass(t.produces, str(GAI.DataAsset))
        assert onto.is_subclass(t.handled_by, str(GAI.Agent))


def test_agent_ids():
    assert agent_id_for(str(GAI.PreprocessingAgent)) == "preprocessing-agent"
    assert agent_id_for(str(GAI.GeoDataProductionAgent)) == "geo-data-production-agent"
    assert agent_id_for(str(GAI.AnalysisEvaluationAgent)) == "analysis-evaluation-agent"
    assert agent_id_for(str(GAI.DTAgent)) == "dt-agent"


def test_valid_request_conforms(onto):
    assert onto.validate_request(str(GAI.ModelTrainingTask), request()) == []


def test_unknown_domain_is_rejected(onto):
    v = onto.validate_request(str(GAI.ModelTrainingTask), request(domain=EX + "Nope"))
    assert any(x.path == str(GAI.inDomain) for x in v)


def test_raw_data_without_crs_is_rejected(onto):
    body = request()
    body.inputs[0].crs = None
    v = onto.validate_request(str(GAI.ModelTrainingTask), body)
    assert any(x.path == str(GAI.crs) for x in v)


def test_bad_crs_and_operator(onto):
    body = request(completion_criteria=[{"metric": "F1", "operator": "~", "threshold": 1}])
    body.inputs[0].crs = "5186"
    paths = {x.path for x in onto.validate_request(str(GAI.ModelTrainingTask), body)}
    assert {str(GAI.crs), str(GAI.operator)} <= paths


def test_unknown_asset_type(onto):
    body = request()
    body.inputs[0].asset_type = "https://ontology.geoai.local/core#Nope"
    v = onto.validate_request(str(GAI.ModelTrainingTask), body)
    assert v and "등록되지 않은" in v[0].message


def test_acquisition_date_becomes_owl_time_instant(onto):
    from rdflib import Literal, Namespace
    from rdflib.namespace import XSD

    time = Namespace("http://www.w3.org/2006/time#")
    body = request().model_dump(mode="json")
    body["inputs"][0]["acquired_at"] = "2025-05-20"
    g = onto.request_graph(str(GAI.ModelTrainingTask), TaskRequestPayload.model_validate(body))
    instants = list(g.objects(None, GAI.temporalExtent))
    assert len(instants) == 1
    assert g.value(instants[0], time.inXSDDate) == Literal("2025-05-20", datatype=XSD.date)
    assert onto.validate_request(str(GAI.ModelTrainingTask), TaskRequestPayload.model_validate(body)) == []
