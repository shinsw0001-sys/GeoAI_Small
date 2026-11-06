"""처리 도구 실행 문맥.

진입점 함수는 `ToolContext` 하나를 받아 산출 자산과 지표를 돌려준다. 문맥은 실행 작업 명세,
데이터·모델 저장소, 임시 작업 디렉터리, 진행 보고를 제공한다. 결과(result.json)는 이 모듈이
성공·실패와 관계없이 반드시 쓴다. 작업 실행기는 그 파일만 읽는다.
"""

from __future__ import annotations

import logging
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geoai_common.jobs import JobResult, JobSpec
from geoai_common.messages import DataAssetRef
from geoai_common.objectstore import ObjectStore, join_uri, put_json

log = logging.getLogger("geoai_tools")


class ToolError(Exception):
    """처리 도구가 판단한 실패. reason 은 운영 콘솔에 그대로 보인다."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class ToolOutput:
    outputs: list[DataAssetRef]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ToolContext:
    spec: JobSpec
    store: ObjectStore
    workdir: Path
    _last_report: float = 0.0

    def param(self, name: str, default: Any) -> Any:
        value = self.spec.parameters.get(name, default)
        return type(default)(value) if default is not None and value is not None else value

    def out(self, *parts: str) -> str:
        return join_uri(self.spec.output_prefix, *parts)

    def fetch(self, uri: str, name: str | None = None) -> Path:
        return self.store.get_file(uri, self.workdir / "in" / (name or uri.rsplit("/", 1)[-1]))

    def progress(self, percent: float, note: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_report < 1.0:
            return
        self._last_report = now
        put_json(self.store, self.spec.progress_uri, {"percent": round(max(0.0, min(100.0, percent)), 1), "note": note})
        log.info("진행 %.1f%% %s", percent, note)


Entrypoint = Callable[[ToolContext], ToolOutput]


def run_job(spec: JobSpec, store: ObjectStore, entrypoints: dict[str, Entrypoint]) -> JobResult:
    fn = entrypoints.get(spec.entrypoint)
    with tempfile.TemporaryDirectory(prefix=f"geoai-{spec.job_id[:20]}-") as tmp:
        ctx = ToolContext(spec=spec, store=store, workdir=Path(tmp))
        try:
            if fn is None:
                raise ToolError("unknown_entrypoint", f"처리 도구에 없는 진입점: {spec.entrypoint}")
            ctx.progress(0.0, "시작", force=True)
            out = fn(ctx)
            ctx.progress(100.0, "완료", force=True)
            result = JobResult(status="succeeded", outputs=out.outputs, metrics=out.metrics)
        except ToolError as exc:
            log.error("실행 작업 실패: %s", exc)
            result = JobResult(status="failed", reason=exc.reason, detail=exc.detail)
        except Exception as exc:  # noqa: BLE001 - 어떤 오류든 결과 파일로 남긴다
            log.error("처리 도구 오류\n%s", traceback.format_exc())
            result = JobResult(status="failed", reason="tool_error", detail=f"{type(exc).__name__}: {exc}")
    put_json(store, spec.result_uri, result.model_dump(mode="json"))
    return result
