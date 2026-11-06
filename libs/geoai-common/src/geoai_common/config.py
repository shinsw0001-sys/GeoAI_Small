"""환경 변수 설정. 모든 에이전트가 같은 이름을 쓴다."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentSettings:
    agent_id: str
    ontology_dir: str = "ontology"
    domain_dirs: list[str] = field(default_factory=list)
    nats_url: str | None = None  # 비우면 InMemoryBus
    database_url: str | None = None  # 비우면 InMemoryStateStore
    llm_base_url: str | None = None  # OpenAI 호환 엔드포인트 (vLLM 등)
    llm_model: str | None = None
    llm_api_key: str | None = None

    @classmethod
    def from_env(cls, default_agent_id: str) -> AgentSettings:
        extra = os.environ.get("GEOAI_DOMAIN_DIRS", "")
        return cls(
            agent_id=os.environ.get("GEOAI_AGENT_ID", default_agent_id),
            ontology_dir=os.environ.get("GEOAI_ONTOLOGY_DIR", "ontology"),
            domain_dirs=[d for d in extra.split(":") if d],
            nats_url=os.environ.get("GEOAI_NATS_URL") or None,
            database_url=os.environ.get("GEOAI_DATABASE_URL") or None,
            llm_base_url=os.environ.get("GEOAI_LLM_BASE_URL") or None,
            llm_model=os.environ.get("GEOAI_LLM_MODEL") or None,
            llm_api_key=os.environ.get("GEOAI_LLM_API_KEY") or None,
        )
