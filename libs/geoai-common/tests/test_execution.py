"""실행 계층: 실행 작업 유형 등록제, 분야 조회, 완료 기준 판정, 저장소, 쿠버네티스 Job 명세."""

from pathlib import Path

import pytest
from geoai_common import GAI, OntologyRegistry
from geoai_common.criteria import describe_unmet, evaluate_criteria
from geoai_common.jobs import JobSpec, build_k8s_job, k8s_job_name
from geoai_common.objectstore import LocalObjectStore, get_json, put_json, split_uri

ROOT = Path(__file__).resolve().parents[3]
BLD = "https://ontology.geoai.local/domain/building-extraction#"


@pytest.fixture(scope="module")
def onto():
    return OntologyRegistry.load(ROOT / "ontology")


def test_every_studio_task_has_a_job(onto):
    for t in onto.plannable_task_types():
        if t.iri in (str(GAI.InferenceTask), str(GAI.SpatialAnalysisTask)):
            continue  # 1년차 1순위 경로 밖
        concrete = {str(GAI.RawData): str(GAI.AerialImage), str(GAI.SpatialProduct): str(GAI.OrthoImage)}
        inputs = [concrete.get(r, r) for r in t.requires]
        job = onto.job_for(t.iri, inputs)
        assert job is not None, t.iri
        assert onto.is_subclass(job.output_type, t.produces), job.iri


def test_job_selection_respects_input_type(onto):
    job = onto.job_for(str(GAI.GeoDataProductionTask), [str(GAI.AerialImage)])
    assert (job.entrypoint, job.accelerator) == ("ortho-ingest", False)
    assert onto.job_for(str(GAI.GeoDataProductionTask), [str(GAI.LidarPointCloud)]) is None
    assert onto.job_for(str(GAI.ModelTrainingTask), [str(GAI.AnalysisReadyData), str(GAI.LabelSet)]).accelerator


def test_domain_defaults_and_labels(onto):
    assert onto.default_criteria(BLD + "Domain") == [
        {"metric": "F1", "operator": ">=", "threshold": 0.75},
        {"metric": "IoU", "operator": ">=", "threshold": 0.6},
    ]
    [c] = onto.label_concepts(BLD + "Domain")
    assert (c.label, c.value) == ("건물", 1)
    lc = onto.label_concepts("https://ontology.geoai.local/domain/land-cover-classification#Domain")
    assert len(lc) == 7 and [c.value for c in lc] == list(range(1, 8))


def test_criteria():
    crit = [{"metric": "IoU", "operator": ">=", "threshold": 0.6}, {"metric": "F1", "operator": ">", "threshold": 0.7}]
    res = evaluate_criteria(crit, {"IoU": 0.61, "F1": 0.7})
    assert [r["met"] for r in res] == [True, False]
    assert describe_unmet(res) == "F1 0.7 (기준 > 0.7)"
    assert evaluate_criteria(crit[:1], {})[0]["met"] is False


def test_local_store(tmp_path):
    s = LocalObjectStore(tmp_path)
    put_json(s, "s3://raw/a/b.json", {"x": 1})
    assert get_json(s, "s3://raw/a/b.json") == {"x": 1}
    assert [o.uri for o in s.list("s3://raw/a")] == ["s3://raw/a/b.json"]
    assert s.list("s3://none/") == []
    for bad in ("raw/a", "s3://raw/../x", "s3://raw"):
        with pytest.raises(ValueError):
            split_uri(bad) if bad != "s3://raw/../x" else s.exists(bad)


def spec(accelerator):
    return JobSpec(
        job_id="T1_ABC.s3.deadbeef".lower().replace("_", "-"),
        job_type="https://ontology.geoai.local/jobs#SegmentationTrainJob",
        entrypoint="segmentation-train",
        accelerator=accelerator,
        task_id="T1.s3",
        task_type=str(GAI.ModelTrainingTask),
        project_id="p",
        domain=BLD + "Domain",
        inputs=[{"asset_id": "a", "asset_type": str(GAI.LabelSet), "storage_uri": "s3://labels/x"}],
        output_prefix="s3://models/p/T1.s3",
        progress_uri="s3://jobs/x/progress.json",
        result_uri="s3://jobs/x/result.json",
    )


def test_k8s_job_manifest():
    m = build_k8s_job(
        spec(True), namespace="geoai-jobs", image="reg/geoai-tools:0.1.0", s3_secret="s", s3_endpoint="http://s3"
    )
    pod = m["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert m["metadata"]["name"] == "geoai-t1-abc-s3-deadbeef"
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert pod["tolerations"][0]["key"] == "nvidia.com/gpu"
    assert c["args"] == ["run", "--spec-env", "GEOAI_JOB_SPEC"]
    assert JobSpec.model_validate_json(c["env"][0]["value"]).entrypoint == "segmentation-train"
    assert pod["securityContext"]["runAsNonRoot"] is True
    cpu = build_k8s_job(spec(True), namespace="n", image="i", s3_secret="s", s3_endpoint="e", use_gpu=False)
    assert "nvidia.com/gpu" not in cpu["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
    assert len(k8s_job_name("x" * 100)) <= 63


CD = "https://ontology.geoai.local/domain/change-detection#"


def test_domain_specific_job_selection(onto):
    images = [str(GAI.AerialImage), str(GAI.LidarPointCloud)]
    # 분야 전용 실행 작업을 범용보다 먼저 고르고, 다른 분야에는 쓰지 않는다
    mt = onto.job_for(str(GAI.GeoDataProductionTask), images, CD + "Domain")
    assert (mt.entrypoint, mt.min_acquisitions, mt.memory) == ("multitemporal-ingest", 2, "16Gi")
    assert onto.job_for(str(GAI.GeoDataProductionTask), images, BLD + "Domain").entrypoint == "ortho-ingest"
    assert onto.job_for(str(GAI.GeoDataProductionTask), images).entrypoint == "ortho-ingest"
    ard = [str(GAI.AnalysisReadyData)]
    assert onto.job_for(str(GAI.LabelingTask), ard, CD + "Domain").entrypoint == "change-labels"
    assert onto.job_for(str(GAI.LabelingTask), ard, BLD + "Domain").entrypoint == "rasterize-labels"
    # 전처리·학습·평가는 범용 도구를 그대로 쓴다
    assert onto.job_for(str(GAI.PreprocessingTask), [str(GAI.OrthoImage)], CD + "Domain").entrypoint == "tiling"
    assert onto.acquisitions_required(CD + "Domain") == 2
    assert onto.acquisitions_required(BLD + "Domain") == 1


def test_change_detection_domain(onto):
    concepts = onto.label_concepts(CD + "Domain")
    assert [(c.iri.rsplit("#", 1)[-1], c.value) for c in concepts] == [
        ("DemolishedBuilding", 1),
        ("LandChange", 2),
        ("ModifiedBuilding", 3),
        ("NewBuilding", 4),
    ]
    assert {c["metric"] for c in onto.default_criteria(CD + "Domain")} == {"foreground_F1", "IoU"}


def test_k8s_job_memory_from_ontology():
    s = spec(False).model_copy(update={"memory": "16Gi"})
    m = build_k8s_job(s, namespace="n", image="i", s3_secret="s", s3_endpoint="e")
    res = m["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert res["requests"]["memory"] == res["limits"]["memory"] == "16Gi"


RD = "https://ontology.geoai.local/domain/road-extraction#"


def test_road_extraction_domain(onto):
    # 선형 객체 분야: 라벨 개념의 기하 유형이 선(sf:LineString)
    [c] = onto.label_concepts(RD + "Domain")
    assert (c.label, c.value, c.geometry, c.linear) == ("도로", 1, "LineString", True)
    assert onto.label_geometry(RD + "Domain") == "LineString"
    assert onto.label_geometry(BLD + "Domain") == "Polygon"
    # 라벨링·평가만 분야 전용, 나머지는 범용
    ard = [str(GAI.AnalysisReadyData)]
    assert onto.job_for(str(GAI.LabelingTask), ard, RD + "Domain").entrypoint == "road-labels"
    ev = onto.job_for(str(GAI.EvaluationTask), [str(GAI.Model), str(GAI.LabelSet)], RD + "Domain")
    assert (ev.entrypoint, ev.accelerator) == ("road-evaluate", True)
    assert onto.job_for(str(GAI.EvaluationTask), [str(GAI.Model), str(GAI.LabelSet)], BLD + "Domain").entrypoint == (
        "segmentation-evaluate"
    )
    assert onto.job_for(str(GAI.GeoDataProductionTask), [str(GAI.AerialImage)], RD + "Domain").entrypoint == (
        "ortho-ingest"
    )
    assert onto.job_for(str(GAI.ModelTrainingTask), ard + [str(GAI.LabelSet)], RD + "Domain").entrypoint == (
        "segmentation-train"
    )
    assert onto.acquisitions_required(RD + "Domain") == 1
    assert {c["metric"] for c in onto.default_criteria(RD + "Domain")} == {"IoU", "APLS"}
