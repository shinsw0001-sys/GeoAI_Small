"""도로 추출 처리 도구 연쇄: 정사영상 정합 → 타일 분할 → 도로 중심선 라벨 → 학습 → 도로망 성능 검증.
도로망 지표(세선화, 그래프, APLS, 완전성·정확성)는 정답을 아는 작은 예로 따로 확인한다."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
from geoai_common.jobs import JobSpec  # noqa: E402
from geoai_common.objectstore import LocalObjectStore, get_json, put_json  # noqa: E402
from geoai_tools import ENTRYPOINTS  # noqa: E402
from geoai_tools.__main__ import seed_road_sample  # noqa: E402
from geoai_tools.road_labels import snap_endpoints  # noqa: E402
from geoai_tools.road_network import (  # noqa: E402
    apls,
    buffer_metrics,
    lines_graph,
    node_counts,
    prune,
    skeleton_graph,
    thin,
    to_lines,
)
from geoai_tools.runtime import run_job  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

GAI = "https://ontology.geoai.local/core#"
RD = "https://ontology.geoai.local/domain/road-extraction#"
CLASSES = [{"iri": RD + "Road", "label": "도로", "value": 1, "geometry": "LineString"}]


def spec(name, entry, inputs, **params):
    return JobSpec(
        job_id=f"r-{name}",
        job_type=f"https://ontology.geoai.local/jobs#{name}",
        entrypoint=entry,
        task_id=f"task.{name}",
        task_type=GAI + "X",
        project_id="p",
        domain=RD + "Domain",
        label_classes=CLASSES,
        inputs=inputs,
        parameters=params,
        completion_criteria=[{"metric": "APLS", "operator": ">=", "threshold": 0.1}],
        output_prefix=f"s3://work/p/{name}",
        progress_uri=f"s3://jobs/r-{name}/progress.json",
        result_uri=f"s3://jobs/r-{name}/result.json",
    )


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = LocalObjectStore(tmp_path_factory.mktemp("store"))
    s.meta = seed_road_sample(s, size=512)
    return s


@pytest.fixture(scope="module")
def chain(store):
    raw = {"asset_id": "a1", "asset_type": GAI + "AerialImage", "storage_uri": store.meta["image"], "crs": "EPSG:5186"}
    out = {}
    r = run_job(spec("ingest", "ortho-ingest", [raw]), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    r = run_job(spec("tiling", "tiling", r.outputs, tile_size=128, block_tiles=1), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["ard"] = r
    r = run_job(spec("labels", "road-labels", r.outputs, label_source=store.meta["label_source"]), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["labels"] = r
    r = run_job(spec("train", "segmentation-train", out["ard"].outputs + r.outputs, epochs=3), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["model"] = r
    r = run_job(spec("eval", "road-evaluate", r.outputs + out["labels"].outputs), store, ENTRYPOINTS)
    assert r.status == "succeeded", r.detail
    out["eval"] = r
    return out


def test_road_labels(store, chain):
    ls = get_json(store, chain["labels"].outputs[0].storage_uri)
    q = ls["quality"]
    assert ls["label_geometry"] == "LineString"
    assert q["lines"] > 0 and q["width_from_attribute"] == q["lines"] and q["width_default"] == 0
    # 국지도로는 간선 위 점에서 시작하므로 교차점이 생기고, 막다른 길은 끝점이 된다
    assert q["junctions"] >= 2 and q["dead_ends"] >= 1 and q["centerline_km"] > 0.2
    # 도로 폭(4~14 m)으로 넓힌 노면 비율: 장면의 일부
    assert 0.03 < q["positive_pixel_ratio"] < 0.5
    ref = json.loads(store.get_bytes(ls["reference_network_uri"]))
    assert ref["features"] and all(f["geometry"]["type"] == "LineString" for f in ref["features"])


def test_road_report(store, chain):
    report = get_json(store, chain["eval"].outputs[0].storage_uri)
    m = report["metrics"]
    assert {"IoU", "F1", "completeness", "correctness", "quality", "APLS", "connectivity"} <= set(m)
    assert all(0.0 <= m[k] <= 1.0 for k in ("IoU", "completeness", "correctness", "quality", "APLS", "connectivity"))
    assert report["kind"] == "road-network-evaluation" and report["preview_layout"] == "network"
    assert report["network"]["reference_source"] == "centerline"
    assert report["network"]["reference"]["length_km"] > 0
    assert store.exists(report["preview_uri"])
    net = json.loads(store.get_bytes(report["predictions_uri"]))
    assert all(f["geometry"]["type"] == "LineString" for f in net["features"])
    assert str(chain["eval"].outputs[1].asset_type).endswith("ResultProduct")


def test_road_labels_need_lines(store, chain):
    empty = "s3://raw/sample-road/points.geojson"
    point = {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [198100, 551900]}}
    crs = {"type": "name", "properties": {"name": "EPSG:5186"}}
    put_json(store, empty, {"type": "FeatureCollection", "crs": crs, "features": [point]})
    r = run_job(spec("nolines", "road-labels", chain["ard"].outputs, label_source=empty), store, ENTRYPOINTS)
    assert (r.status, r.reason) == ("failed", "no_road_geometry")


# ---------------------------------------------------------------------------
# 도로망 지표: 정답을 아는 예
# ---------------------------------------------------------------------------


def cross_mask(size=101, width=7, gap=None):
    m = np.zeros((size, size), dtype=bool)
    c = size // 2
    m[c - width // 2 : c + width // 2 + 1, 5:-5] = True
    m[5:-5, c - width // 2 : c + width // 2 + 1] = True
    if gap:
        m[c - width // 2 : c + width // 2 + 1, gap[0] : gap[1]] = False
    return m


def test_skeleton_of_cross_is_one_junction():
    g = skeleton_graph(thin(cross_mask()), 0.0, 101.0, 1.0)
    prune(g, spur_m=4, min_component_m=5)
    n = node_counts(g)
    assert n["components"] == 1 and n["junctions"] == 1 and n["dead_ends"] == 4


def test_apls_perfect_and_broken():
    # 마스크 십자(화소 5~95)와 같은 위치의 참조 중심선
    ref = lines_graph([LineString([(5, 50), (95, 50)]), LineString([(50, 5), (50, 95)])], step=1.0)
    same = skeleton_graph(thin(cross_mask(width=5)), -0.5, 100.5, 1.0)
    prune(same, spur_m=4, min_component_m=5)
    good = apls(ref, same, spacing=10, buffer_m=4)
    assert good["APLS"] > 0.85 and good["connectivity"] > 0.95
    # 가로 도로 한쪽을 끊으면 화소는 거의 같아도 경로가 끊겨 APLS 가 크게 떨어진다
    broken = skeleton_graph(thin(cross_mask(width=5, gap=(30, 36))), -0.5, 100.5, 1.0)
    prune(broken, spur_m=4, min_component_m=5)
    bad = apls(ref, broken, spacing=10, buffer_m=4)
    assert bad["APLS"] < good["APLS"] - 0.1 and bad["connectivity"] < good["connectivity"]


def test_buffer_metrics():
    ref = [LineString([(0, 0), (100, 0)])]
    m = buffer_metrics(ref, [LineString([(0, 1), (50, 1)]), LineString([(0, 30), (50, 30)])], buffer_m=3)
    # 버퍼 끝은 둥글어 예측 끝(50 m)에서 3 m 더 덮는다: 완전성 53/100
    assert m["completeness"] == pytest.approx(0.53, abs=0.01)
    assert m["correctness"] == pytest.approx(0.5, abs=0.01)
    assert m["quality"] == pytest.approx(0.53 * 0.5 / (0.53 - 0.53 * 0.5 + 0.5), abs=0.01)


def test_snap_and_lines():
    a = LineString([(0, 0), (100, 0)])
    b = LineString([(50, 0.6), (50, 50)])  # 0.6 m 떨어져 닿지 않은 끝점
    out, n = snap_endpoints([(a, 10, 1), (b, 6, 1)], snap_m=1.0)
    assert n == 1 and out[1][0].coords[0] == (50.0, 0.0)
    g = lines_graph([ln for ln, _, _ in out], step=5)
    assert node_counts(g)["junctions"] == 1 and len(to_lines(g)) == 3
