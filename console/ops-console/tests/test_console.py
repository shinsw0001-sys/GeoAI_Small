"""운영 콘솔: 화면 제공, 자료 목록·올리기, 작업 요청·거절, 위험 알림 분류 (모의 하위 에이전트 사용)."""

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from geoai_common import InMemoryBus, InMemoryStateStore, OntologyRegistry
from geoai_common.objectstore import LocalObjectStore
from geoai_common.testing import EchoSubAgent
from ops_console import ConsoleDeps, ConsoleSettings, create_app
from studio_agent import StudioAgent
from studio_agent.app import Runtime
from studio_agent.app import create_app as create_studio

ROOT = Path(__file__).resolve().parents[3]
GAI = "https://ontology.geoai.local/core#"
BUILDING = "https://ontology.geoai.local/domain/building-extraction#Domain"
SUBS = ["geo-data-production-agent", "preprocessing-agent", "model-training-agent", "analysis-evaluation-agent"]


@pytest.fixture
def client(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    objects = LocalObjectStore(tmp_path)
    img = tmp_path / "a.tif"
    with rasterio.open(
        img, "w", driver="GTiff", width=64, height=64, count=3, dtype="uint8", crs="EPSG:5186",
        transform=from_origin(198000, 552000, 0.5, 0.5),
    ) as dst:  # fmt: skip
        dst.write(np.full((3, 64, 64), 100, dtype=np.uint8))
    objects.put_file("s3://raw/site/a.tif", img)
    objects.put_bytes("s3://raw/site/b.geojson", b'{"type":"FeatureCollection","features":[]}')
    onto = OntologyRegistry.load(ROOT / "ontology")

    @asynccontextmanager
    async def deps_context():
        bus, store = InMemoryBus(), InMemoryStateStore()
        studio = StudioAgent("studio-agent", bus, store, onto)
        await studio.start()
        for sid in SUBS:
            await EchoSubAgent(sid, bus, store, onto, fail_on={GAI + "LabelingTask"}).start()
        app = create_studio(runtime=Runtime(agent=studio, bus=bus, store=store))
        app.state.runtime = Runtime(agent=studio, bus=bus, store=store)
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://studio")
        yield ConsoleDeps(studio=http, objects=objects, settings=ConsoleSettings())
        await http.aclose()

    with TestClient(create_app(deps_context=deps_context)) as c:
        yield c


def test_index_and_static(client):
    assert "GeoAI 운영 콘솔" in client.get("/").text
    css = client.get("/static/app.css").text
    assert "--critical: #c62828" in css  # 위험 항목만 빨간색
    assert "https://" not in client.get("/static/app.js").text.replace("https://ontology.geoai.local", "")


def test_catalog(client):
    cat = client.get("/api/catalog").json()
    building = next(d for d in cat["domains"] if d["iri"] == BUILDING)
    assert {c["metric"] for c in building["default_criteria"]} == {"IoU", "F1"}
    assert cat["labels"][GAI + "EvaluationTask"] == "성능 검증 작업"


def test_assets_and_upload(client):
    a = client.get("/api/assets").json()
    assert a["images"][0]["crs"] == "EPSG:5186" and a["images"][0]["ground_sample_distance"] == 0.5
    assert [v["uri"] for v in a["vectors"]] == ["s3://raw/site/b.geojson"]
    r = client.post("/api/assets", files={"file": ("x.geojson", b'{"type":"FeatureCollection","features":[]}')})
    assert r.json()["uri"] == "s3://raw/uploads/x.geojson"
    assert client.post("/api/assets", files={"file": ("x.exe", b"MZ")}).status_code == 422
    assert client.post("/api/assets", files={"file": ("y.geojson", b"[]")}).status_code == 422


def test_object_access_is_limited_to_buckets(client):
    assert client.get("/api/objects", params={"uri": "s3://raw/site/b.geojson"}).status_code == 200
    assert client.get("/api/objects", params={"uri": "s3://secret/x"}).status_code == 403
    assert client.get("/api/objects", params={"uri": "/etc/passwd"}).status_code == 400


def body(**kw):
    b = {
        "domain": BUILDING,
        "task_type": GAI + "ModelTrainingTask",
        "image_uris": ["s3://raw/site/a.tif"],
        "label_source": "s3://raw/site/b.geojson",
        "criteria": [{"metric": "IoU", "operator": ">=", "threshold": 0.5}],
    }
    b.update(kw)
    return b


def test_failed_subtask_becomes_red_alert(client):
    r = client.post("/api/tasks", json=body())
    assert r.status_code == 202, r.text
    d = client.get(f"/api/tasks/{r.json()['task_id']}").json()
    assert d["task"]["status"] == "Failed" and d["task"]["critical"] is True
    assert [s["status"] for s in d["steps"]] == ["Succeeded", "Succeeded", "Failed", "Planned"]
    assert d["steps"][2]["critical"] is True
    assert d["request"]["parameters"]["label_source"] == "s3://raw/site/b.geojson"
    alerts = client.get("/api/overview").json()["alerts"]
    assert alerts[0]["title"] == "작업 실패"


def test_success_has_no_alert(client):
    r = client.post("/api/tasks", json=body(task_type=GAI + "PreprocessingTask"))
    d = client.get(f"/api/tasks/{r.json()['task_id']}").json()
    assert d["task"]["status"] == "Succeeded" and d["overall_percent"] == 100
    assert client.get("/api/overview").json()["alerts"] == []


def test_rejection_is_returned(client):
    r = client.post("/api/tasks", json=body(domain="https://ontology.geoai.local/domain/nope#Domain"))
    assert r.status_code == 422 and r.json()["reason"] == "shape_violation"
    r = client.post("/api/tasks", json=body(image_uris=["s3://raw/none.tif"]))
    assert r.status_code == 422


CD = "https://ontology.geoai.local/domain/change-detection#Domain"


def test_change_request_builds_dated_inputs(client):
    client.post("/api/assets", files={"file": ("t1.las", b"LASF")})
    cat = client.get("/api/catalog").json()
    assert next(d for d in cat["domains"] if d["iri"] == CD)["acquisitions"] == 2
    acq = [
        {
            "acquired_at": "2023-05-12",
            "image_uris": ["s3://raw/site/a.tif"],
            "lidar_uris": ["s3://raw/uploads/t1.las"],
            "building_source": "s3://raw/site/b.geojson",
        },
        {
            "acquired_at": "2025-05-20",
            "image_uris": ["s3://raw/site/a.tif"],
            "building_source": "s3://raw/site/b.geojson",
        },
    ]
    r = client.post(
        "/api/tasks",
        json={"domain": CD, "task_type": GAI + "PreprocessingTask", "acquisitions": acq, "criteria": []},
    )
    assert r.status_code == 202, r.text
    d = client.get(f"/api/tasks/{r.json()['task_id']}").json()
    inputs = d["inputs"]
    assert [(i["asset_type"].rsplit("#", 1)[-1], i["acquired_at"]) for i in inputs] == [
        ("AerialImage", "2023-05-12"),
        ("LidarPointCloud", "2023-05-12"),
        ("AerialImage", "2025-05-20"),
    ]
    assert [s["acquired_at"] for s in d["request"]["parameters"]["label_sources"]] == ["2023-05-12", "2025-05-20"]
    assets = client.get("/api/assets").json()
    assert [p["uri"] for p in assets["point_clouds"]] == ["s3://raw/uploads/t1.las"]
    assert client.get("/api/models").json() == {"models": []}
    assert client.post("/api/tasks", json={"domain": CD, "task_type": GAI + "PreprocessingTask"}).status_code == 422


RD = "https://ontology.geoai.local/domain/road-extraction#Domain"


def test_road_domain_catalog_and_preset(client):
    from geoai_common.objectstore import put_json

    cat = client.get("/api/catalog").json()
    road = next(d for d in cat["domains"] if d["iri"] == RD)
    # 화면은 온톨로지의 라벨 기하(선)로 도로 입력 안내·도로망 결과 화면을 고른다
    assert road["label_geometry"] == "LineString" and road["label_concepts"][0]["geometry"] == "LineString"
    assert {c["metric"] for c in road["default_criteria"]} == {"IoU", "APLS"}
    assert next(d for d in cat["domains"] if d["iri"] == BUILDING)["label_geometry"] == "Polygon"
    objects = client.app.state.deps.objects
    put_json(objects, "s3://raw/sample-road/sample.json",
             {"domain": RD, "image": "s3://raw/site/a.tif", "label_source": "s3://raw/site/b.geojson"})  # fmt: skip
    presets = client.get("/api/assets").json()["presets"]
    assert [p["domain"] for p in presets] == [RD]
    js = client.get("/static/app.js").text
    assert "road-network-evaluation" in js and "도로망 위상 평가" in js


LC = "https://ontology.geoai.local/domain/land-cover-classification#Domain"


def test_landcover_domain_catalog(client):
    cat = client.get("/api/catalog").json()
    lc = next(d for d in cat["domains"] if d["iri"] == LC)
    # 화면은 개념의 분류 코드(skos:notation)로 토지피복 입력 안내를 고른다
    assert sorted(c["notation"] for c in lc["label_concepts"]) == ["100", "200", "300", "400", "500", "600", "700"]
    assert lc["label_geometry"] == "Polygon"
    assert {c["metric"] for c in lc["default_criteria"]} == {"overall_accuracy", "mIoU"}
    assert all("notation" not in c for c in next(d for d in cat["domains"] if d["iri"] == BUILDING)["label_concepts"])
    js = client.get("/static/app.js").text
    assert "landcover-evaluation" in js and "혼동 행렬" in js and "L1_CODE" in js
