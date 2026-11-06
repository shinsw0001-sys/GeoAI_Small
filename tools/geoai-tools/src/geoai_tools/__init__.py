"""GeoAI 처리 도구. 진입점 이름은 온톨로지의 gai:toolEntrypoint 값과 같다."""

from . import evaluate, ingest, labels, tiling, train
from .runtime import Entrypoint

ENTRYPOINTS: dict[str, Entrypoint] = {
    "ortho-ingest": ingest.run,
    "tiling": tiling.run,
    "rasterize-labels": labels.run,
    "segmentation-train": train.run,
    "segmentation-evaluate": evaluate.run,
}

__all__ = ["ENTRYPOINTS"]
