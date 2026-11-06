"""데이터·모델 저장소 접근.

에이전트·처리 도구·운영 콘솔은 저장소를 `s3://<버킷>/<키>` 주소로만 다룬다. 구현 두 가지:

- S3ObjectStore: S3 호환 저장소(SeaweedFS 등). 운영·쿠버네티스 환경.
- LocalObjectStore: 디렉터리 하나를 버킷 묶음처럼 쓴다(`<루트>/<버킷>/<키>`). 개발·시험 환경.

버킷 용도 (1년차 고정)
  raw        원본 데이터 (영상, 참조 벡터)
  processed  공간정보 산출물, 분석 준비 데이터
  labels     라벨 세트
  models     모델
  results    결과 산출물, 성능 검증 보고
  jobs       실행 작업 명세·진행·결과 기록
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

BUCKETS = ("raw", "processed", "labels", "models", "results", "jobs")


@dataclass(frozen=True)
class ObjectInfo:
    uri: str
    size: int


def split_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"s3:// 주소가 아닙니다: {uri}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key or ".." in key.split("/"):
        raise ValueError(f"잘못된 저장소 주소: {uri}")
    return bucket, key


def join_uri(prefix: str, *parts: str) -> str:
    return "/".join([prefix.rstrip("/"), *(p.strip("/") for p in parts)])


class ObjectStore(Protocol):
    def put_file(self, uri: str, path: str | Path) -> None: ...

    def get_file(self, uri: str, path: str | Path) -> Path: ...

    def put_bytes(self, uri: str, data: bytes) -> None: ...

    def get_bytes(self, uri: str) -> bytes: ...

    def exists(self, uri: str) -> bool: ...

    def list(self, prefix_uri: str) -> list[ObjectInfo]: ...


def put_json(store: ObjectStore, uri: str, data: Any) -> None:
    store.put_bytes(uri, json.dumps(data, ensure_ascii=False, indent=2).encode())


def get_json(store: ObjectStore, uri: str) -> Any:
    return json.loads(store.get_bytes(uri))


class LocalObjectStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, uri: str) -> Path:
        bucket, key = split_uri(uri)
        path = (self.root / bucket / key).resolve()
        if self.root not in path.parents:
            raise ValueError(f"저장소 밖 경로: {uri}")
        return path

    def put_file(self, uri: str, path: str | Path) -> None:
        target = self._path(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    def get_file(self, uri: str, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(uri), path)
        return path

    def put_bytes(self, uri: str, data: bytes) -> None:
        target = self._path(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(target)  # 읽는 쪽이 반쯤 쓴 파일을 보지 않게 한다

    def get_bytes(self, uri: str) -> bytes:
        return self._path(uri).read_bytes()

    def exists(self, uri: str) -> bool:
        return self._path(uri).is_file()

    def list(self, prefix_uri: str) -> list[ObjectInfo]:
        bucket, _, key = prefix_uri[5:].partition("/")
        base = self.root / bucket
        if not base.is_dir():
            return []
        out = []
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base).as_posix()
            if p.is_file() and rel.startswith(key) and not rel.endswith(".part"):
                out.append(ObjectInfo(uri=f"s3://{bucket}/{rel}", size=p.stat().st_size))
        return out


class S3ObjectStore:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, region: str = "us-east-1") -> None:
        import boto3
        from botocore.config import Config

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 5}),
        )
        self._buckets: set[str] = set()

    def _ensure_bucket(self, bucket: str) -> None:
        if bucket in self._buckets:
            return
        from botocore.exceptions import ClientError

        try:
            self._client.head_bucket(Bucket=bucket)
        except ClientError:
            self._client.create_bucket(Bucket=bucket)
        self._buckets.add(bucket)

    def put_file(self, uri: str, path: str | Path) -> None:
        bucket, key = split_uri(uri)
        self._ensure_bucket(bucket)
        self._client.upload_file(str(path), bucket, key)

    def get_file(self, uri: str, path: str | Path) -> Path:
        bucket, key = split_uri(uri)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(bucket, key, str(path))
        return path

    def put_bytes(self, uri: str, data: bytes) -> None:
        bucket, key = split_uri(uri)
        self._ensure_bucket(bucket)
        self._client.put_object(Bucket=bucket, Key=key, Body=data)

    def get_bytes(self, uri: str) -> bytes:
        bucket, key = split_uri(uri)
        return self._client.get_object(Bucket=bucket, Key=key)["Body"].read()

    def exists(self, uri: str) -> bool:
        from botocore.exceptions import ClientError

        bucket, key = split_uri(uri)
        try:
            self._client.head_object(Bucket=bucket, Key=key)
        except ClientError:
            return False
        return True

    def list(self, prefix_uri: str) -> list[ObjectInfo]:
        from botocore.exceptions import ClientError

        bucket, _, prefix = prefix_uri[5:].partition("/")
        out = []
        try:
            for page in self._client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
                out.extend(
                    ObjectInfo(uri=f"s3://{bucket}/{o['Key']}", size=o["Size"]) for o in page.get("Contents", [])
                )
        except ClientError:
            return []
        return out


def object_store_from_env() -> ObjectStore:
    """GEOAI_S3_ENDPOINT 가 있으면 S3, 없으면 GEOAI_LOCAL_STORE 디렉터리(기본 .geoai-data)."""
    endpoint = os.environ.get("GEOAI_S3_ENDPOINT")
    if endpoint:
        return S3ObjectStore(
            endpoint,
            os.environ["GEOAI_S3_ACCESS_KEY"],
            os.environ["GEOAI_S3_SECRET_KEY"],
            os.environ.get("GEOAI_S3_REGION", "us-east-1"),
        )
    return LocalObjectStore(os.environ.get("GEOAI_LOCAL_STORE", ".geoai-data"))
