import pytest
from conftest import asset
from geoai_common import GAI, DataAssetRef
from studio_agent import Planner, PlanningError


def refs(*items):
    return [DataAssetRef.model_validate(i) for i in items]


def names(steps):
    return [s.task_type.rsplit("#", 1)[-1] for s in steps]


def test_training_from_raw_chains_production_preprocessing_labeling(onto):
    steps = Planner(onto).plan(str(GAI.ModelTrainingTask), refs(asset("a1", "AerialImage")))
    assert names(steps) == ["GeoDataProductionTask", "PreprocessingTask", "LabelingTask", "ModelTrainingTask"]
    prod, pre, lab, train = steps
    assert pre.depends_on == [prod.key]
    # 전처리 산출을 라벨링과 학습이 같이 쓴다 (전처리는 한 번만)
    assert lab.depends_on == [pre.key]
    assert set(train.depends_on) == {pre.key, lab.key}
    assert [s.agent for s in steps] == [
        "geo-data-production-agent",
        "preprocessing-agent",
        "model-training-agent",
        "model-training-agent",
    ]
    assert [a.asset_id for a in prod.input_assets] == ["a1"]


def test_inference_with_given_model(onto):
    steps = Planner(onto).plan(
        str(GAI.InferenceTask), refs(asset("a1", "SatelliteImage"), asset("m1", "Model", crs=None))
    )
    assert names(steps) == ["GeoDataProductionTask", "PreprocessingTask", "InferenceTask"]
    assert [a.asset_id for a in steps[-1].input_assets] == ["m1"]


def test_inference_without_model_trains_one(onto):
    steps = Planner(onto).plan(str(GAI.InferenceTask), refs(asset("a1", "LidarPointCloud")))
    assert names(steps) == [
        "GeoDataProductionTask",
        "PreprocessingTask",
        "LabelingTask",
        "ModelTrainingTask",
        "InferenceTask",
    ]


def test_processed_input_skips_production(onto):
    steps = Planner(onto).plan(str(GAI.SpatialAnalysisTask), refs(asset("o1", "OrthoImage")))
    assert names(steps) == ["SpatialAnalysisTask"]
    steps = Planner(onto).plan(str(GAI.ModelTrainingTask), refs(asset("o1", "OrthoImage")))
    assert names(steps) == ["PreprocessingTask", "LabelingTask", "ModelTrainingTask"]


def test_spatial_analysis_from_raw_picks_shortest_chain(onto):
    # 가공 데이터는 생성·가공(1단계)과 전처리(2단계) 모두로 얻을 수 있다. 짧은 쪽을 고른다.
    steps = Planner(onto).plan(str(GAI.SpatialAnalysisTask), refs(asset("d1", "DroneImage")))
    assert names(steps) == ["GeoDataProductionTask", "SpatialAnalysisTask"]


def test_abstract_task_type_is_unplannable(onto):
    with pytest.raises(PlanningError):
        Planner(onto).plan(str(GAI.AnalysisEvaluationTask), refs(asset("o1", "OrthoImage")))


def test_unplannable(onto):
    with pytest.raises(PlanningError):
        Planner(onto).plan(str(GAI.PreprocessingTask), refs(asset("m1", "Model", crs=None)))
