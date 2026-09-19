from dataclasses import dataclass

from llm_eval.models import ExperimentSummary


@dataclass(frozen=True, slots=True)
class ExperimentDelta:
    pass_rate: float
    relevance: float
    citation_coverage: float
    latency_p95_ms: float
    mean_cost_usd: float


@dataclass(frozen=True, slots=True)
class ComparisonDecision:
    acceptable: bool
    delta: ExperimentDelta
    reasons: list[str]


def compare(
    baseline: ExperimentSummary,
    candidate: ExperimentSummary,
    *,
    max_pass_rate_drop: float = 0.02,
    max_relevance_drop: float = 0.03,
    max_latency_increase_ms: float = 500.0,
    max_cost_increase_usd: float | None = None,
) -> ComparisonDecision:
    delta = ExperimentDelta(
        pass_rate=candidate.pass_rate - baseline.pass_rate,
        relevance=candidate.mean_relevance - baseline.mean_relevance,
        citation_coverage=(candidate.mean_citation_coverage - baseline.mean_citation_coverage),
        latency_p95_ms=(candidate.latency_p95_ms - baseline.latency_p95_ms),
        mean_cost_usd=candidate.mean_cost_usd - baseline.mean_cost_usd,
    )

    reasons: list[str] = []
    if delta.pass_rate < -max_pass_rate_drop:
        reasons.append("pass-rate regression exceeds budget")
    if delta.relevance < -max_relevance_drop:
        reasons.append("relevance regression exceeds budget")
    if delta.latency_p95_ms > max_latency_increase_ms:
        reasons.append("p95 latency increase exceeds budget")
    if max_cost_increase_usd is not None and delta.mean_cost_usd > max_cost_increase_usd:
        reasons.append("mean cost increase exceeds budget")

    return ComparisonDecision(
        acceptable=not reasons,
        delta=delta,
        reasons=reasons,
    )
