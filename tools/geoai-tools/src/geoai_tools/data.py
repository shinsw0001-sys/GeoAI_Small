"""분석 준비 데이터·라벨 세트 읽기."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from geoai_common.objectstore import ObjectStore, get_json


@dataclass
class TileSet:
    manifest: dict
    images: np.ndarray  # (N, 3, T, T) uint8
    valid: np.ndarray  # (N, T, T) bool
    masks: np.ndarray | None = None  # (N, T, T) uint8, 라벨 세트를 읽었을 때만

    def split(self, name: str) -> np.ndarray:
        return np.array([t["id"] for t in self.manifest["tiles"] if t["split"] == name], dtype=np.int64)


def load_npz(store: ObjectStore, uri: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(store.get_bytes(uri))) as z:
        return {k: z[k] for k in z.files}


def load_ard(store: ObjectStore, manifest_uri: str) -> TileSet:
    manifest = get_json(store, manifest_uri)
    arrays = load_npz(store, manifest["tiles_uri"])
    return TileSet(manifest=manifest, images=arrays["images"], valid=arrays["valid"])


def load_labelled(store: ObjectStore, labelset_uri: str) -> tuple[dict, TileSet]:
    """라벨 세트 manifest 와, 그것이 가리키는 분석 준비 데이터 + 마스크."""
    labels = get_json(store, labelset_uri)
    tiles = load_ard(store, labels["ard_manifest"])
    tiles.masks = load_npz(store, labels["masks_uri"])["masks"]
    return labels, tiles
