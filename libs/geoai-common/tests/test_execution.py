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
