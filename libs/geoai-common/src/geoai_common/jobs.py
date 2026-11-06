"""실행 작업(gai:Job) 명세와 작업 실행기.

전문 에이전트는 판단만 하고 연산은 실행 작업으로 넘긴다. 에이전트는 실행 작업 명세(JobSpec)를
만들어 작업 실행기에 넘기고, 처리 도구(geoai-tools)는 명세를 읽어 연산한 뒤 진행·결과를
데이터·모델 저장소의 정해진 주소(progress_uri, result_uri)에 쓴다. 실행기는 그 두 주소만 읽으므로
실행 방식(하위 프로세스, 쿠버네티스 Job)이 달라도 에이전트 코드는 같다.

구현 두 가지:
- SubprocessJobRunner: 같은 장비의 별도 프로세스로 실행. 개발·시험 환경.
- KubernetesJobRunner: geoai-jobs 네임스페이스에 batch/v1 Job 을 만들어 처리 도구 이미지로 실행.
  그래픽 처리 장치가 필요한 실행 작업은 장치 자원을 요청하고 장치 노드의 오염(taint)을 허용한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .messages import DataAssetRef
from .objectstore import ObjectStore, get_json, join_uri

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], Awaitable[None]]


class JobSpec(BaseModel):
    """실행 작업 명세. 처리 도구의 유일한 입력이다."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    job_type: str = Field(description="실행 작업 클래스 IRI")
    entrypoint: str = Field(description="처리 도구 진입점 (gai:toolEntrypoint)")
    accelerator: bool = False
    task_id: str
    task_type: str
    project_id: str
    domain: str
    label_classes: list[dict[str, Any]] = Field(default_factory=list, description="[{iri, label, value}]")
    inputs: list[DataAssetRef]
    parameters: dict[str, Any] = Field(default_factory=dict)
    completion_criteria: list[dict[str, Any]] = Field(default_factory=list)
    output_prefix: str = Field(description="산출물을 쓸 저장소 주소 접두사")
    progress_uri: str
    result_uri: str


class JobResult(BaseModel):
    """처리 도구가 result_uri 에 쓰는 결과."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(description="succeeded | failed")
    outputs: list[DataAssetRef] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    reason: str = ""
    detail: str = ""


class JobError(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def job_paths(job_id: str) -> tuple[str, str, str]:
    """명세·진행·결과 주소."""
    base = f"s3://jobs/{job_id}"
    return join_uri(base, "spec.json"), join_uri(base, "progress.json"), join_uri(base, "result.json")


class JobRunner(Protocol):
    async def run(self, spec: JobSpec, on_progress: ProgressCallback | None = None) -> JobResult: ...


async def _poll_progress(store: ObjectStore, spec: JobSpec, on_progress: ProgressCallback | None, last: list) -> None:
    if on_progress is None:
        return
    try:
        if not await asyncio.to_thread(store.exists, spec.progress_uri):
            return
        data = await asyncio.to_thread(get_json, store, spec.progress_uri)
    except (OSError, ValueError):
        return  # 쓰는 중이면 다음 주기에 읽는다
    key = (data.get("percent"), data.get("note"))
    if key != tuple(last):
        last[:] = list(key)
        await on_progress(float(data.get("percent") or 0.0), str(data.get("note") or ""))


async def _read_result(store: ObjectStore, spec: JobSpec) -> JobResult | None:
    if not await asyncio.to_thread(store.exists, spec.result_uri):
        return None
    return JobResult.model_validate(await asyncio.to_thread(get_json, store, spec.result_uri))


def _finish(result: JobResult | None, fallback: str) -> JobResult:
    if result is None:
        raise JobError("job_crashed", fallback)
    if result.status != "succeeded":
        raise JobError(result.reason or "job_failed", result.detail or fallback)
    return result


class SubprocessJobRunner:
    """`python -m geoai_tools run --spec <파일>` 을 별도 프로세스로 실행한다."""

    def __init__(
        self,
        store: ObjectStore,
        workdir: str | Path,
        python: str = sys.executable,
        env: dict[str, str] | None = None,
        poll_seconds: float = 0.5,
        timeout_seconds: float = 6 * 3600,
    ) -> None:
        self.store = store
        self.workdir = Path(workdir)
        self.python = python
        self.env = env or {}
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    async def run(self, spec: JobSpec, on_progress: ProgressCallback | None = None) -> JobResult:
        self.workdir.mkdir(parents=True, exist_ok=True)
        spec_path = self.workdir / f"{spec.job_id}.json"
        spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        log_path = self.workdir / f"{spec.job_id}.log"
        with log_path.open("wb") as log_file:
            proc = await asyncio.create_subprocess_exec(
                self.python,
                "-m",
                "geoai_tools",
                "run",
                "--spec",
                str(spec_path),
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, **self.env},
            )
            last: list = [None, None]
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.timeout_seconds
            while proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=self.poll_seconds)
                except TimeoutError:
                    pass
                await _poll_progress(self.store, spec, on_progress, last)
                if loop.time() > deadline:
                    proc.kill()
                    raise JobError("job_timeout", f"{self.timeout_seconds:.0f}초 안에 끝나지 않았습니다.")
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        return _finish(await _read_result(self.store, spec), f"종료 코드 {proc.returncode}. 기록 끝부분: {tail}")


# ---------------------------------------------------------------------------
# 쿠버네티스 Job 실행기
# ---------------------------------------------------------------------------

SA_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")


def k8s_job_name(job_id: str) -> str:
    name = re.sub(r"[^a-z0-9-]", "-", job_id.lower()).strip("-")
    return ("geoai-" + name)[:63].rstrip("-")


def build_k8s_job(
    spec: JobSpec,
    *,
    namespace: str,
    image: str,
    s3_secret: str,
    s3_endpoint: str,
    service_account: str = "geoai-job",
    use_gpu: bool = True,
    cpu: str = "2",
    memory: str = "8Gi",
) -> dict[str, Any]:
    """처리 도구 이미지로 실행 작업 하나를 수행하는 batch/v1 Job 매니페스트."""
    gpu = spec.accelerator and use_gpu
    limits: dict[str, str] = {"memory": memory}
    if gpu:
        limits["nvidia.com/gpu"] = "1"
    pod_spec: dict[str, Any] = {
        "serviceAccountName": service_account,
        "automountServiceAccountToken": False,
        "restartPolicy": "Never",
        "securityContext": {"runAsNonRoot": True, "runAsUser": 10001, "seccompProfile": {"type": "RuntimeDefault"}},
        "containers": [
            {
                "name": "tool",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "args": ["run", "--spec-env", "GEOAI_JOB_SPEC"],
                "env": [
                    {"name": "GEOAI_JOB_SPEC", "value": spec.model_dump_json()},
                    {"name": "GEOAI_S3_ENDPOINT", "value": s3_endpoint},
                    {
                        "name": "GEOAI_S3_ACCESS_KEY",
                        "valueFrom": {"secretKeyRef": {"name": s3_secret, "key": "access-key"}},
                    },
                    {
                        "name": "GEOAI_S3_SECRET_KEY",
                        "valueFrom": {"secretKeyRef": {"name": s3_secret, "key": "secret-key"}},
                    },
                ],
                "resources": {"requests": {"cpu": cpu, "memory": memory}, "limits": limits},
                "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                "volumeMounts": [{"name": "work", "mountPath": "/tmp"}],  # noqa: S108 - 읽기 전용 루트 대신 빈 볼륨
            }
        ],
        "volumes": [{"name": "work", "emptyDir": {}}],
    }
    if gpu:
        pod_spec["tolerations"] = [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
    labels = {
        "app.kubernetes.io/part-of": "geoai",
        "app.kubernetes.io/component": "job",
        "geoai.local/entrypoint": spec.entrypoint,
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": k8s_job_name(spec.job_id),
            "namespace": namespace,
            "labels": labels,
            "annotations": {"geoai.local/task-id": spec.task_id, "geoai.local/job-type": spec.job_type},
        },
        "spec": {
            "backoffLimit": 1,
            "ttlSecondsAfterFinished": 86400,
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }


class KubernetesJobRunner:
    """파드 안의 서비스 계정 토큰으로 쿠버네티스 응용 프로그램 인터페이스를 직접 호출한다(추가 의존성 없음)."""

    def __init__(
        self,
        store: ObjectStore,
        *,
        image: str,
        namespace: str = "geoai-jobs",
        s3_secret: str = "geoai-object-store-client",  # noqa: S107 - 비밀 객체 이름
        s3_endpoint: str = "",
        use_gpu: bool = True,
        poll_seconds: float = 5.0,
        timeout_seconds: float = 12 * 3600,
        api_server: str = "https://kubernetes.default.svc",
    ) -> None:
        self.store = store
        self.image = image
        self.namespace = namespace
        self.s3_secret = s3_secret
        self.s3_endpoint = s3_endpoint
        self.use_gpu = use_gpu
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.api_server = api_server

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        token = (SA_DIR / "token").read_text().strip()
        ctx = ssl.create_default_context(cafile=str(SA_DIR / "ca.crt"))
        req = urllib.request.Request(  # noqa: S310 - 클러스터 내부 고정 주소
            self.api_server + path,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:  # noqa: S310
            return json.load(resp)

    async def run(self, spec: JobSpec, on_progress: ProgressCallback | None = None) -> JobResult:
        manifest = build_k8s_job(
            spec,
            namespace=self.namespace,
            image=self.image,
            s3_secret=self.s3_secret,
            s3_endpoint=self.s3_endpoint,
            use_gpu=self.use_gpu,
        )
        name = manifest["metadata"]["name"]
        base = f"/apis/batch/v1/namespaces/{self.namespace}/jobs"
        try:
            await asyncio.to_thread(self._call, "POST", base, manifest)
        except urllib.error.HTTPError as exc:
            if exc.code != 409:  # 409: 이미 있음(재전달). 기존 Job 을 계속 지켜본다.
                raise JobError("job_submit_failed", f"쿠버네티스 Job 생성 실패: HTTP {exc.code}") from exc
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_seconds
        last: list = [None, None]
        while True:
            await asyncio.sleep(self.poll_seconds)
            await _poll_progress(self.store, spec, on_progress, last)
            status = (await asyncio.to_thread(self._call, "GET", f"{base}/{name}")).get("status", {})
            if status.get("succeeded") or status.get("failed"):
                break
            if loop.time() > deadline:
                raise JobError("job_timeout", f"쿠버네티스 Job {name} 이 제한 시간 안에 끝나지 않았습니다.")
        return _finish(await _read_result(self.store, spec), f"쿠버네티스 Job {name} 이 결과 없이 끝났습니다.")


def job_runner_from_env(store: ObjectStore) -> JobRunner:
    """GEOAI_JOB_RUNNER=kubernetes 이면 쿠버네티스 Job, 아니면 하위 프로세스."""
    if os.environ.get("GEOAI_JOB_RUNNER", "subprocess") == "kubernetes":
        return KubernetesJobRunner(
            store,
            image=os.environ["GEOAI_TOOLS_IMAGE"],
            namespace=os.environ.get("GEOAI_JOB_NAMESPACE", "geoai-jobs"),
            s3_secret=os.environ.get("GEOAI_JOB_S3_SECRET", "geoai-object-store-client"),
            s3_endpoint=os.environ.get("GEOAI_S3_ENDPOINT", ""),
            use_gpu=os.environ.get("GEOAI_JOB_GPU", "true").lower() == "true",
        )
    return SubprocessJobRunner(store, os.environ.get("GEOAI_JOB_WORKDIR", ".geoai-jobs"))
