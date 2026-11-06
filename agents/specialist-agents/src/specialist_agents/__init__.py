"""GeoAI Studio 전문 에이전트 4종."""

from .agent import SpecialistAgent

SPECIALIST_AGENT_IDS = (
    "geo-data-production-agent",
    "preprocessing-agent",
    "model-training-agent",
    "analysis-evaluation-agent",
)

__all__ = ["SPECIALIST_AGENT_IDS", "SpecialistAgent"]
