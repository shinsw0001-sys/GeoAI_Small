"""완료 기준(gai:CompletionCriterion) 판정."""

from __future__ import annotations

import operator
from typing import Any

OPERATORS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}


def evaluate_criteria(criteria: list[dict[str, Any]], metrics: dict[str, float]) -> list[dict[str, Any]]:
    """기준마다 {metric, operator, threshold, value, met}. 지표가 없으면 미충족이다."""
    out = []
    for c in criteria:
        value = metrics.get(c["metric"])
        op = OPERATORS.get(c["operator"])
        met = value is not None and op is not None and bool(op(float(value), float(c["threshold"])))
        out.append({**c, "value": value, "met": met})
    return out


def describe_unmet(results: list[dict[str, Any]]) -> str:
    parts = []
    for r in results:
        if r["met"]:
            continue
        value = "지표 없음" if r["value"] is None else f"{r['value']:.4g}"
        parts.append(f"{r['metric']} {value} (기준 {r['operator']} {r['threshold']:g})")
    return ", ".join(parts)
