from pathlib import Path

import pytest
from geoai_common import GAI, OntologyRegistry

ROOT = Path(__file__).resolve().parents[3]
EX = "https://ontology.geoai.local/domain/example#"


@pytest.fixture(scope="session")
def onto() -> OntologyRegistry:
    return OntologyRegistry.load(ROOT / "ontology", [ROOT / "ontology" / "examples"])


def asset(asset_id: str, cls: str, crs: str | None = "EPSG:5186") -> dict:
    return {"asset_id": asset_id, "asset_type": str(GAI[cls]), "storage_uri": f"s3://raw/{asset_id}", "crs": crs}


def request_envelope(task_cls: str, inputs: list[dict], **kw) -> dict:
    body = {
        "performative": "request",
        "sender": "ops-console",
        "recipient": "studio-agent",
        "project_id": "p-001",
        "task_type": str(GAI[task_cls]),
        "ontology_version": "0.1.0",
        "payload": {
            "domain": EX + "Domain",
            "inputs": inputs,
            "completion_criteria": [{"metric": "mIoU", "operator": ">=", "threshold": 0.7}],
        },
    }
    body.update(kw)
    return body
