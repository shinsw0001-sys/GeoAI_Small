"""운영 콘솔 백엔드.

운영 콘솔은 1년차에 서비스·사용자 계층을 대행하는 Studio Agent의 상위 발신자(ops-console)다.
Studio Agent의 HTTP 응용 프로그램 인터페이스로 작업을 요청·조회하고, 데이터·모델 저장소에서
산출물(보고서, 미리보기, 학습 이력)을 읽어 화면에 보여 준다. 상태는 두지 않는다.

화면용 응용 프로그램 인터페이스
- GET  /api/catalog            분야(권장 완료 기준·라벨 개념), 요청 가능한 작업 유형
- GET  /api/overview           상태별 건수, 위험 알림, 최근 작업
- GET  /api/tasks/{task_id}    단계별 진행·지표·완료 기준 판정·산출물·메시지 기록
- POST /api/tasks              작업 요청 (화면 입력 → 메시지 봉투 → Studio Agent)
- GET  /api/assets             원본 자료 목록 (영상: 좌표 참조 체계·해상도, 라이다 점군, 합성 표본 시점 정보)
- GET  /api/models             학습된 모델 목록 (다른 분야 학습의 초기 가중치 선택용)
- POST /api/assets             원본 자료 올리기
- GET  /api/objects?uri=       저장소 객체 내려받기 (산출물 보기용)

위험 알림(빨간색) 기준: 작업 실패, 거절, 완료 기준 미달, 하위 작업 실패.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import tempfile
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from geoai_common.criteria import evaluate_criteria
from geoai_common.objectstore import BUCKETS, ObjectStore, get_json, object_store_from_env, split_uri
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
GAI = "https://ontology.geoai.local/core#"
CONSOLE_ID = "ops-console"
CRITICAL = {"Failed", "Rejected"}
RASTER_EXT = (".tif", ".tiff")
VECTOR_EXT = (".geojson", ".json")
POINT_EXT = (".las",)


@dataclass
class ConsoleSettings:
    studio_url: str = "http://localhost:8080"
    studio_agent_id: str = "studio-agent"
    default_project: str = "building-extraction"

    @classmethod
    def from_env(cls) -> ConsoleSettings:
        return cls(
            studio_url=os.environ.get("GEOAI_STUDIO_URL", cls.studio_url),
            studio_agent_id=os.environ.get("GEOAI_STUDIO_AGENT_ID", cls.studio_agent_id),
            default_project=os.environ.get("GEOAI_DEFAULT_PROJECT", cls.default_project),
        )


@dataclass
class ConsoleDeps:
    studio: httpx.AsyncClient
    objects: ObjectStore
    settings: ConsoleSettings
    _catalog: dict | None = field(default=None)


class CriterionIn(BaseModel):
    metric: str
    operator: str = ">="
    threshold: float


class AcquisitionIn(BaseModel):
    """다시기 분야의 시점 하나. 영상·라이다·시점별 벡터를 묶는다."""

    acquired_at: date
    image_uris: list[str] = Field(min_length=1)
    lidar_uris: list[str] = Field(default_factory=list)
    building_source: str | None = None
    landcover_source: str | None = None


class TaskIn(BaseModel):
    project_id: str = Field(default="", max_length=64)
    domain: str
    task_type: str
    image_uris: list[str] = Field(default_factory=list)
    acquisitions: list[AcquisitionIn] = Field(default_factory=list)
    label_source: str | None = None
    criteria: list[CriterionIn] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


def raster_info(objects: ObjectStore, uri: str) -> dict:
    """영상 머리부에서 좌표 참조 체계·해상도·크기를 읽는다."""
    import rasterio

    with tempfile.TemporaryDirectory() as tmp:
        path = objects.get_file(uri, Path(tmp) / "img.tif")
        with rasterio.open(path) as ds:
            epsg = ds.crs.to_epsg() if ds.crs else None
            return {
                "crs": f"EPSG:{epsg}" if epsg else (ds.crs.to_string() if ds.crs else None),
                "ground_sample_distance": round(abs(ds.res[0]), 4),
                "width": ds.width,
                "height": ds.height,
                "bands": ds.count,
            }


def create_app(
    deps: ConsoleDeps | None = None,
    settings: ConsoleSettings | None = None,
    deps_context: Callable[[], AbstractAsyncContextManager[ConsoleDeps]] | None = None,
) -> FastAPI:
    """deps: 이미 만든 의존성(시험용). deps_context: 의존성을 만들고 정리하는 문맥(일체형 실행용)."""
    settings = settings or ConsoleSettings.from_env()

    @asynccontextmanager
    async def default_context():  # type: ignore[no-untyped-def]
        client = httpx.AsyncClient(base_url=settings.studio_url, timeout=30.0)
        try:
            yield ConsoleDeps(studio=client, objects=object_store_from_env(), settings=settings)
        finally:
            await client.aclose()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        if deps is not None:
            app.state.deps = deps
            yield
            return
        async with (deps_context or default_context)() as made:
            app.state.deps = made
            yield

    app = FastAPI(title="GeoAI 운영 콘솔", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def d() -> ConsoleDeps:
        return app.state.deps

    async def studio_get(path: str, **params: Any) -> dict:
        try:
            r = await d().studio.get(path, params=params or None)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Studio Agent 연결 실패: {exc}") from exc
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="작업이 없습니다.")
        r.raise_for_status()
        return r.json()

    async def catalog() -> dict:
        if d()._catalog is None:
            card = await studio_get("/v1/agent-card")
            labels = {a["task_type"]: a["label"] for a in card["accepts"]}
            d()._catalog = {
                "ontology_version": card["ontology_version"],
                "domains": card["domains"],
                "task_types": card["accepts"],
                "labels": labels,
                "default_project": d().settings.default_project,
            }
        return d()._catalog

    def summarize(task: dict, cat: dict) -> dict:
        req = task.get("request", {})
        domain = req.get("domain", "")
        dom_label = next((x["label"] for x in cat["domains"] if x["iri"] == domain), domain.rsplit("#", 1)[-1])
        checks = evaluate_criteria(req.get("completion_criteria", []), task.get("metrics", {}))
        criteria_failed = task["status"] == "Failed" and task.get("detail", "").startswith("완료 기준 미달")
        return {
            "task_id": task["task_id"],
            "project_id": task["project_id"],
            "task_type": task["task_type"],
            "label": cat["labels"].get(task["task_type"], task["task_type"].rsplit("#", 1)[-1]),
            "domain": domain,
            "domain_label": dom_label,
            "status": task["status"],
            "detail": task.get("detail", ""),
            "critical": task["status"] in CRITICAL,
            "criteria_failed": criteria_failed,
            "criteria": checks if task["status"] in ("Succeeded", "Failed") and task.get("metrics") else [],
            "metrics": task.get("metrics", {}),
            "created_at": task["created_at"],
            "updated_at": task["updated_at"],
        }

    # ------------------------------------------------------------------
    # 화면
    # ------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/api/catalog")
    async def get_catalog() -> dict:
        return await catalog()

    @app.get("/api/overview")
    async def overview(project_id: str | None = None) -> dict:
        cat = await catalog()
        params = {"limit": 100, **({"project_id": project_id} if project_id else {})}
        tasks = [summarize(t, cat) for t in (await studio_get("/v1/tasks", **params))["tasks"]]
        counts = {
            s: sum(t["status"] == s for t in tasks) for s in ("Planned", "Running", "Succeeded", "Failed", "Rejected")
        }
        alerts = [
            {
                "task_id": t["task_id"],
                "title": "완료 기준 미달"
                if t["criteria_failed"]
                else ("작업 실패" if t["status"] == "Failed" else "작업 거절"),
                "label": t["label"],
                "domain_label": t["domain_label"],
                "detail": t["detail"],
                "at": t["updated_at"],
            }
            for t in tasks
            if t["critical"]
        ]
        return {"counts": counts, "alerts": alerts, "tasks": tasks[:30]}

    @app.get("/api/tasks/{task_id}")
    async def task_detail(task_id: str) -> dict:
        cat = await catalog()
        data, msgs = await asyncio.gather(
            studio_get(f"/v1/tasks/{task_id}"), studio_get(f"/v1/tasks/{task_id}/messages")
        )
        task, subtasks = data["task"], data["subtasks"]
        progress: dict[str, dict] = {}
        timeline = []
        for m in msgs["messages"]:
            p = m["payload"]
            if m["performative"] == "progress" and m["sender"] != d().settings.studio_agent_id and m.get("task_id"):
                progress[m["task_id"]] = {"percent": p.get("percent"), "note": p.get("note", "")}
            note = p.get("note") or p.get("detail") or p.get("reason") or ""
            if m["performative"] == "request" and m["sender"] == d().settings.studio_agent_id:
                note = cat["labels"].get(m.get("task_type") or "", "")
            timeline.append(
                {
                    "at": m["created_at"],
                    "performative": m["performative"],
                    "sender": m["sender"],
                    "recipient": m.get("recipient"),
                    "task_id": m.get("task_id"),
                    "note": note,
                }
            )
        steps = []
        for s in subtasks:
            pr = progress.get(s["task_id"], {})
            percent = 100.0 if s["status"] == "Succeeded" else (pr.get("percent") or 0.0)
            steps.append(
                {
                    "task_id": s["task_id"],
                    "task_type": s["task_type"],
                    "label": cat["labels"].get(s["task_type"], s["task_type"].rsplit("#", 1)[-1]),
                    "agent": s["agent"],
                    "status": s["status"],
                    "percent": percent,
                    "note": pr.get("note", "") if s["status"] == "Running" else "",
                    "detail": s.get("detail", ""),
                    "metrics": s.get("metrics", {}),
                    "outputs": s.get("outputs", []),
                    "critical": s["status"] in CRITICAL,
                }
            )
        summary = summarize(task, cat)
        summary["criteria"] = evaluate_criteria(
            task.get("request", {}).get("completion_criteria", []), task.get("metrics", {})
        )
        done = sum(s["status"] == "Succeeded" for s in steps)
        running = [s["percent"] for s in steps if s["status"] == "Running"]
        overall = (done + (sum(running) / 100.0)) / len(steps) * 100 if steps else 0.0

        artifacts: dict[str, Any] = {}

        def load(uri: str) -> Any:
            try:
                return get_json(d().objects, uri)
            except (OSError, ValueError, KeyError):
                return None
            except Exception:  # noqa: BLE001 - 저장소 오류는 화면을 막지 않는다
                log.warning("산출물 읽기 실패: %s", uri)
                return None

        for s in steps:
            for o in s["outputs"]:
                t = o["asset_type"]
                if t == GAI + "EvaluationReport":
                    artifacts["report"] = await asyncio.to_thread(load, o["storage_uri"])
                elif t == GAI + "Model":
                    card = await asyncio.to_thread(load, o["storage_uri"])
                    if card:
                        artifacts["model"] = {
                            k: card.get(k)
                            for k in (
                                "history",
                                "best_epoch",
                                "val_IoU",
                                "device",
                                "train_seconds",
                                "parameters",
                                "hyperparameters",
                                "weights_uri",
                                "pretrained",
                                "bands",
                                "class_weights",
                            )
                        }
                elif t == GAI + "LabelSet":
                    ls = await asyncio.to_thread(load, o["storage_uri"])
                    if ls:
                        artifacts["labels"] = {
                            k: ls.get(k)
                            for k in ("quality", "label_source", "label_sources", "compared", "change_reference_uri")
                        }
                elif t == GAI + "AnalysisReadyData":
                    ard = await asyncio.to_thread(load, o["storage_uri"])
                    if ard:
                        artifacts["ard"] = {
                            k: ard.get(k)
                            for k in ("split_counts", "quality", "tile_size", "crs", "bands", "acquisitions")
                        }
        return {
            "task": summary,
            "request": task.get("request", {}),
            "inputs": task.get("input_assets", []),
            "outputs": task.get("outputs", []),
            "overall_percent": round(overall, 1),
            "steps": steps,
            "artifacts": artifacts,
            "timeline": timeline,
        }

    @app.post("/api/tasks")
    async def create_task(body: TaskIn) -> JSONResponse:
        cat = await catalog()
        if not body.image_uris and not body.acquisitions:
            raise HTTPException(status_code=422, detail="입력 영상이 없습니다.")

        async def image_ref(uri: str, acquired_at: date | None) -> dict:
            try:
                split_uri(uri)
                info = await asyncio.to_thread(raster_info, d().objects, uri)
            except Exception as exc:  # noqa: BLE001 - 사용자 입력 오류로 돌려준다
                raise HTTPException(status_code=422, detail=f"영상을 읽지 못했습니다: {uri} ({exc})") from exc
            ref = {
                "asset_id": uri.rsplit("/", 1)[-1],
                "asset_type": GAI + "AerialImage",
                "storage_uri": uri,
                "crs": info["crs"],
                "ground_sample_distance": info["ground_sample_distance"],
            }
            if acquired_at:
                ref["acquired_at"] = acquired_at.isoformat()
            return ref

        inputs = [await image_ref(uri, None) for uri in body.image_uris]
        params = dict(body.parameters)
        label_sources = []
        for acq in sorted(body.acquisitions, key=lambda a: a.acquired_at):
            refs = [await image_ref(uri, acq.acquired_at) for uri in acq.image_uris]
            inputs.extend(refs)
            for uri in acq.lidar_uris:
                if not uri.lower().endswith(POINT_EXT):
                    raise HTTPException(status_code=422, detail=f"라이다 점군은 LAS(.las)여야 합니다: {uri}")
                # 점군의 좌표 참조 체계는 파일 안 WKT 를 처리 도구가 우선 읽는다. 여기서는 같은 시점 영상 값을 둔다.
                inputs.append(
                    {
                        "asset_id": uri.rsplit("/", 1)[-1],
                        "asset_type": GAI + "LidarPointCloud",
                        "storage_uri": uri,
                        "crs": refs[0]["crs"],
                        "acquired_at": acq.acquired_at.isoformat(),
                    }
                )
            if acq.building_source or acq.landcover_source:
                label_sources.append(
                    {
                        "acquired_at": acq.acquired_at.isoformat(),
                        "buildings": acq.building_source,
                        "landcover": acq.landcover_source,
                    }
                )
        if label_sources:
            params["label_sources"] = label_sources
        if body.label_source:
            params["label_source"] = body.label_source
        project = re.sub(r"[^a-z0-9-]", "-", (body.project_id or d().settings.default_project).lower()).strip("-")
        envelope = {
            "performative": "request",
            "sender": CONSOLE_ID,
            "recipient": d().settings.studio_agent_id,
            "project_id": project or "default",
            "task_id": str(uuid.uuid4()),
            "task_type": body.task_type,
            "ontology_version": cat["ontology_version"],
            "payload": {
                "domain": body.domain,
                "inputs": inputs,
                "completion_criteria": [c.model_dump() for c in body.criteria],
                "parameters": params,
            },
        }
        try:
            r = await d().studio.post("/v1/messages", json=envelope)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Studio Agent 연결 실패: {exc}") from exc
        reply = r.json()
        if r.status_code == 202 and reply.get("performative") == "accept":
            return JSONResponse(
                status_code=202, content={"task_id": reply["task_id"], "plan": reply["payload"]["plan"]}
            )
        payload = reply.get("payload", {}) if isinstance(reply, dict) else {}
        return JSONResponse(
            status_code=422,
            content={
                "reason": payload.get("reason", "error"),
                "detail": payload.get("detail") or reply.get("detail", ""),
                "violations": payload.get("violations", []),
            },
        )

    # ------------------------------------------------------------------
    # 자료
    # ------------------------------------------------------------------

    @app.get("/api/assets")
    async def assets() -> dict:
        objs = await asyncio.to_thread(d().objects.list, "s3://raw/")
        images, vectors, clouds, presets = [], [], [], []
        for o in objs:
            name = o.uri.lower()
            if name.endswith(RASTER_EXT):
                images.append({"uri": o.uri, "size": o.size})
            elif name.endswith(POINT_EXT):
                clouds.append({"uri": o.uri, "size": o.size})
            elif name.endswith("sample.json"):
                try:
                    meta = await asyncio.to_thread(get_json, d().objects, o.uri)
                except Exception:  # noqa: BLE001
                    log.warning("표본 등록 정보 읽기 실패: %s", o.uri)
                    continue
                if isinstance(meta, dict) and meta.get("acquisitions"):
                    presets.append(meta)
            elif name.endswith(VECTOR_EXT):
                vectors.append({"uri": o.uri, "size": o.size})
        for img in images:
            try:
                img.update(await asyncio.to_thread(raster_info, d().objects, img["uri"]))
            except Exception:  # noqa: BLE001
                img["error"] = "영상 머리부를 읽지 못했습니다."
        return {"images": images, "vectors": vectors, "point_clouds": clouds, "presets": presets}

    @app.get("/api/models")
    async def models() -> dict:
        objs = await asyncio.to_thread(d().objects.list, "s3://models/")
        out = []
        for o in objs:
            if not o.uri.endswith("/model.json"):
                continue
            try:
                card = await asyncio.to_thread(get_json, d().objects, o.uri)
            except Exception:  # noqa: BLE001
                log.warning("모델 명세 읽기 실패: %s", o.uri)
                continue
            if not isinstance(card, dict) or card.get("kind") != "segmentation-model":
                continue
            arch = card.get("architecture", {})
            out.append(
                {
                    "uri": o.uri,
                    "domain": card.get("domain"),
                    "val_IoU": card.get("val_IoU"),
                    "in_channels": arch.get("in_channels"),
                    "num_classes": arch.get("num_classes"),
                    "base_channels": arch.get("base_channels"),
                    "classes": [c.get("label") for c in card.get("classes", [])[1:]],
                    "task_id": o.uri.split("/")[-3] if o.uri.count("/") >= 5 else "",
                }
            )
        return {"models": out}

    @app.post("/api/assets")
    async def upload(file: UploadFile) -> dict:
        name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "upload").name)
        if not name.lower().endswith(RASTER_EXT + VECTOR_EXT + POINT_EXT):
            raise HTTPException(
                status_code=422, detail="GeoTIFF(.tif), GeoJSON(.geojson), 라이다 LAS(.las)만 올릴 수 있습니다."
            )
        uri = f"s3://raw/uploads/{name}"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            with path.open("wb") as f:
                while chunk := await file.read(1 << 20):
                    f.write(chunk)
            if name.lower().endswith(VECTOR_EXT):
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(doc, dict) or not isinstance(doc.get("features"), list):
                        raise ValueError("features 없음")
                except ValueError as exc:
                    raise HTTPException(
                        status_code=422, detail="GeoJSON 지물 모음(FeatureCollection)이 아닙니다."
                    ) from exc
            await asyncio.to_thread(d().objects.put_file, uri, path)
        return {"uri": uri}

    @app.get("/api/objects")
    async def get_object(uri: str) -> Response:
        try:
            bucket, key = split_uri(uri)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if bucket not in BUCKETS:
            raise HTTPException(status_code=403, detail="허용하지 않는 버킷입니다.")
        try:
            data = await asyncio.to_thread(d().objects.get_bytes, uri)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail="객체가 없습니다.") from exc
        media = mimetypes.guess_type(key)[0] or "application/octet-stream"
        if key.endswith(".geojson"):
            media = "application/geo+json"
        filename = key.rsplit("/", 1)[-1]
        return Response(
            content=data, media_type=media, headers={"Content-Disposition": f'inline; filename="{filename}"'}
        )

    return app
