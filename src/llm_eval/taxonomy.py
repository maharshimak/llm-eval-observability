"""Deterministic failure taxonomy for evaluated cases.

Categories mirror the pass conditions of ``ExperimentRunner`` exactly, so a
case is classified with no categories if and only if it passed.
"""

from __future__ import annotations

from llm_eval.models import CaseMetrics

FORBIDDEN_CONTENT = "forbidden_content"
LOW_RELEVANCE = "low_relevance"
MISSING_EXPECTED_CITATION = "missing_expected_citation"
LATENCY_BUDGET_EXCEEDED = "latency_budget_exceeded"

FAILURE_CATEGORIES: tuple[str, ...] = (
    FORBIDDEN_CONTENT,
    LOW_RELEVANCE,
    MISSING_EXPECTED_CITATION,
    LATENCY_BUDGET_EXCEEDED,
)


def classify_case(
    metrics: CaseMetrics,
    *,
    min_relevance: float,
    min_citation_coverage: float,
    max_latency_ms: float,
) -> tuple[str, ...]:
    """Return every failure category that applies, in ``FAILURE_CATEGORIES`` order."""
    categories: list[str] = []
    if metrics.forbidden_hit:
        categories.append(FORBIDDEN_CONTENT)
    if not metrics.relevance >= min_relevance:
        categories.append(LOW_RELEVANCE)
    if not metrics.citation_coverage >= min_citation_coverage:
        categories.append(MISSING_EXPECTED_CITATION)
    if not metrics.latency_ms <= max_latency_ms:
        categories.append(LATENCY_BUDGET_EXCEEDED)
    return tuple(categories)


def count_failures(classified: list[tuple[str, ...]]) -> dict[str, int]:
    """Count cases per category; every category is present (zero when unused)."""
    return {
        category: sum(1 for categories in classified if category in categories)
        for category in FAILURE_CATEGORIES
    }
