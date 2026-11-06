"""GeoAI 처리 도구. 진입점 이름은 온톨로지의 gai:toolEntrypoint 값과 같다."""

from . import change_labels, evaluate, ingest, labels, multitemporal, tiling, train
from .runtime import Entrypoint

ENTRYPOINTS: dict[str, Entrypoint] = {
    "ortho-ingest": ingest.run,
    "tiling": tiling.run,
    "rasterize-labels": labels.run,
    "segmentation-train": train.run,
    "segmentation-evaluate": evaluate.run,
    "multitemporal-ingest": multitemporal.run,
    "change-labels": change_labels.run,
}

__all__ = ["ENTRYPOINTS"]
