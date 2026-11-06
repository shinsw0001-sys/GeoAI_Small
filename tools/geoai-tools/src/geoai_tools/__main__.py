"""처리 도구 명령.

geoai-tools run --spec <명세 파일 | s3://주소>     실행 작업 하나 수행 (작업 실행기가 호출)
geoai-tools run --spec-env GEOAI_JOB_SPEC           명세를 환경 변수에서 읽기 (쿠버네티스 Job)
geoai-tools sample [--size 2048] [--prefix s3://raw/sample]
                                                    합성 항공 영상·건물 윤곽을 데이터·모델 저장소에 넣기
geoai-tools sample --kind change [--size 1024] [--prefix s3://raw/sample-change]
                                                    합성 두 시점 영상·라이다·건물 윤곽·토지피복을 넣기
geoai-tools sample --kind road [--size 1024] [--prefix s3://raw/sample-road]
                                                    합성 항공 영상·도로 중심선(폭 속성)을 넣기
geoai-tools sample --kind landcover [--size 1024] [--prefix s3://raw/sample-landcover]
                                                    합성 항공 영상·토지피복 다각형(L1_CODE)을 넣기
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from geoai_common.jobs import JobSpec
from geoai_common.objectstore import get_json, join_uri, object_store_from_env, put_json

from . import ENTRYPOINTS
from .runtime import run_job
from .sample import CRS, GSD, generate
from .sample_change import generate_change
from .sample_landcover import generate_landcover
from .sample_road import generate_road


def _load_spec(args: argparse.Namespace, store) -> JobSpec:  # type: ignore[no-untyped-def]
    if args.spec_env:
        return JobSpec.model_validate_json(os.environ[args.spec_env])
    if args.spec.startswith("s3://"):
        return JobSpec.model_validate(get_json(store, args.spec))
    return JobSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8"))


def seed_sample(store, prefix: str = "s3://raw/sample", size: int = 2048, seed: int = 7) -> dict:  # type: ignore[no-untyped-def]
    """합성 자료를 저장소에 올리고 등록 정보(sidecar)를 함께 쓴다."""
    with tempfile.TemporaryDirectory() as tmp:
        tif, geojson = generate(tmp, size=size, seed=seed)
        image_uri = join_uri(prefix, tif.name)
        vector_uri = join_uri(prefix, geojson.name)
        store.put_file(image_uri, tif)
        store.put_file(vector_uri, geojson)
    meta = {"image": image_uri, "label_source": vector_uri, "crs": CRS, "ground_sample_distance": GSD, "size_px": size}
    put_json(store, join_uri(prefix, "sample.json"), meta)
    return meta


def seed_change_sample(store, prefix: str = "s3://raw/sample-change", size: int = 1024, seed: int = 11) -> dict:  # type: ignore[no-untyped-def]
    """합성 변화탐지 자료(두 시점)를 올리고 등록 정보(sample.json)를 쓴다. 운영 콘솔이 시점 입력을 미리 채운다."""
    with tempfile.TemporaryDirectory() as tmp:
        files = generate_change(tmp, size=size, seed=seed)
        acquisitions = []
        for f in files:
            entry = {"acquired_at": f["acquired_at"]}
            for key in ("image", "lidar", "buildings", "landcover"):
                uri = join_uri(prefix, f[key].name)
                store.put_file(uri, f[key])
                entry[key] = uri
            acquisitions.append(entry)
    meta = {
        "kind": "change-detection",
        "domain": "https://ontology.geoai.local/domain/change-detection#Domain",
        "acquisitions": acquisitions,
        "crs": CRS,
        "ground_sample_distance": GSD,
        "size_px": size,
    }
    put_json(store, join_uri(prefix, "sample.json"), meta)
    return meta


def seed_road_sample(store, prefix: str = "s3://raw/sample-road", size: int = 1024, seed: int = 21) -> dict:  # type: ignore[no-untyped-def]
    """합성 도로 자료(영상 + 도로 중심선)를 올리고 등록 정보(sample.json)를 쓴다. 운영 콘솔이 입력을 미리 고른다."""
    with tempfile.TemporaryDirectory() as tmp:
        tif, geojson = generate_road(tmp, size=size, seed=seed)
        image_uri = join_uri(prefix, tif.name)
        vector_uri = join_uri(prefix, geojson.name)
        store.put_file(image_uri, tif)
        store.put_file(vector_uri, geojson)
    meta = {
        "kind": "road-extraction",
        "domain": "https://ontology.geoai.local/domain/road-extraction#Domain",
        "image": image_uri,
        "label_source": vector_uri,
        "crs": CRS,
        "ground_sample_distance": GSD,
        "size_px": size,
    }
    put_json(store, join_uri(prefix, "sample.json"), meta)
    return meta


def seed_landcover_sample(store, prefix: str = "s3://raw/sample-landcover", size: int = 1024, seed: int = 31) -> dict:  # type: ignore[no-untyped-def]
    """합성 토지피복 자료(영상 + 토지피복 다각형)를 올리고 등록 정보(sample.json)를 쓴다."""
    with tempfile.TemporaryDirectory() as tmp:
        tif, geojson = generate_landcover(tmp, size=size, seed=seed)
        image_uri = join_uri(prefix, tif.name)
        vector_uri = join_uri(prefix, geojson.name)
        store.put_file(image_uri, tif)
        store.put_file(vector_uri, geojson)
    meta = {
        "kind": "land-cover-classification",
        "domain": "https://ontology.geoai.local/domain/land-cover-classification#Domain",
        "image": image_uri,
        "label_source": vector_uri,
        "crs": CRS,
        "ground_sample_distance": GSD,
        "size_px": size,
    }
    put_json(store, join_uri(prefix, "sample.json"), meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="geoai-tools")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    src = run.add_mutually_exclusive_group(required=True)
    src.add_argument("--spec")
    src.add_argument("--spec-env")
    sample = sub.add_parser("sample")
    sample.add_argument("--prefix", default="s3://raw/sample")
    sample.add_argument("--size", type=int, default=2048)
    sample.add_argument("--seed", type=int, default=None)
    sample.add_argument("--kind", choices=["building", "change", "road", "landcover"], default="building")
    args = parser.parse_args(argv)

    store = object_store_from_env()
    if args.command == "sample":
        if args.kind == "landcover":
            prefix = args.prefix if args.prefix != "s3://raw/sample" else "s3://raw/sample-landcover"
            meta = seed_landcover_sample(store, prefix, args.size if args.size != 2048 else 1024, args.seed or 31)
        elif args.kind == "road":
            prefix = args.prefix if args.prefix != "s3://raw/sample" else "s3://raw/sample-road"
            meta = seed_road_sample(store, prefix, args.size if args.size != 2048 else 1024, args.seed or 21)
        elif args.kind == "change":
            prefix = args.prefix if args.prefix != "s3://raw/sample" else "s3://raw/sample-change"
            meta = seed_change_sample(store, prefix, args.size if args.size != 2048 else 1024, args.seed or 11)
        else:
            meta = seed_sample(store, args.prefix, args.size, args.seed or 7)
        print(json.dumps(meta, ensure_ascii=False))
        return 0
    result = run_job(_load_spec(args, store), store, ENTRYPOINTS)
    return 0 if result.status == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
