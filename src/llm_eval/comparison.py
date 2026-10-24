from dataclasses import dataclass
from math import isfinite

from llm_eval.models import CaseMetrics, ExperimentSummary
from llm_eval.statistics import paired_bootstrap_compare


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


def _validate_regression_budgets(
    required: tuple[float, ...], optional_cost: float | None,
) -> None:
    values = required if optional_cost is None else (*required, optional_cost)
    if any(type(value) not in (int, float) or not isfinite(value) or value < 0 for value in values):
        raise ValueError("Regression budgets must be finite and non-negative.")


def compare(
    baseline: ExperimentSummary,
    candidate: ExperimentSummary,
    *,
    max_pass_rate_drop: float = 0.02,
    max_relevance_drop: float = 0.03,
    max_citation_drop: float = 0.03,
    max_latency_increase_ms: float = 500.0,
    max_cost_increase_usd: float | None = None,
) -> ComparisonDecision:
    _validate_regression_budgets(
        (max_pass_rate_drop, max_relevance_drop, max_citation_drop, max_latency_increase_ms),
        max_cost_increase_usd,
    )
    for summary in (baseline, candidate):
        values = (
            summary.pass_rate, summary.mean_relevance, summary.mean_citation_coverage,
            summary.latency_p95_ms, summary.mean_cost_usd,
        )
        if any(type(value) not in (int, float) or not isfinite(value) for value in values):
            raise ValueError("Comparison summary measurements must be finite numbers.")
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
    if delta.citation_coverage < -max_citation_drop:
        reasons.append("citation-coverage regression exceeds budget")
    if delta.latency_p95_ms > max_latency_increase_ms:
        reasons.append("p95 latency increase exceeds budget")
    if max_cost_increase_usd is not None and delta.mean_cost_usd > max_cost_increase_usd:
        reasons.append("mean cost increase exceeds budget")

    return ComparisonDecision(
        acceptable=not reasons,
        delta=delta,
        reasons=reasons,
    )


@dataclass(frozen=True, slots=True)
class PairedMetricEvidence:
    metric: str
    mean_delta: float
    lower: float
    upper: float
    higher_is_better: bool
    significant_improvement: bool
    significant_regression: bool


@dataclass(frozen=True, slots=True)
class StatisticalComparisonDecision:
    acceptable: bool
    metrics: tuple[PairedMetricEvidence, ...]
    reasons: tuple[str, ...]


def _case_map(summary: ExperimentSummary) -> dict[str, CaseMetrics]:
    if not summary.cases:
        raise ValueError("paired comparison requires per-case metrics")
    mapped = {case.case_id: case for case in summary.cases}
    if len(mapped) != len(summary.cases):
        raise ValueError("case IDs must be unique for paired comparison")
    return mapped


def compare_paired_cases(
    baseline: ExperimentSummary,
    candidate: ExperimentSummary,
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 42,
    max_pass_rate_drop: float = 0.02,
    max_relevance_drop: float = 0.03,
    max_citation_drop: float = 0.03,
    max_latency_increase_ms: float = 500.0,
    max_cost_increase_usd: float | None = None,
) -> StatisticalComparisonDecision:
    """Compare the same eval cases with practical budgets and bootstrap uncertainty."""
    _validate_regression_budgets(
        (max_pass_rate_drop, max_relevance_drop, max_citation_drop, max_latency_increase_ms),
        max_cost_increase_usd,
    )
    base = _case_map(baseline)
    cand = _case_map(candidate)
    if set(base) != set(cand):
        raise ValueError("baseline and candidate must contain the same case IDs")
    case_ids = sorted(base)

    specs = (
        (
            "pass_rate",
            [1.0 if base[case_id].passed else 0.0 for case_id in case_ids],
            [1.0 if cand[case_id].passed else 0.0 for case_id in case_ids],
            True,
            max_pass_rate_drop,
        ),
        (
            "relevance",
            [base[case_id].relevance for case_id in case_ids],
            [cand[case_id].relevance for case_id in case_ids],
            True,
            max_relevance_drop,
        ),
        (
            "citation_coverage",
            [base[case_id].citation_coverage for case_id in case_ids],
            [cand[case_id].citation_coverage for case_id in case_ids],
            True,
            max_citation_drop,
        ),
        (
            "latency_ms",
            [base[case_id].latency_ms for case_id in case_ids],
            [cand[case_id].latency_ms for case_id in case_ids],
            False,
            max_latency_increase_ms,
        ),
        (
            "cost_usd",
            [base[case_id].estimated_cost_usd for case_id in case_ids],
            [cand[case_id].estimated_cost_usd for case_id in case_ids],
            False,
            max_cost_increase_usd,
        ),
    )

    evidence: list[PairedMetricEvidence] = []
    reasons: list[str] = []
    for metric, baseline_values, candidate_values, higher_is_better, practical_budget in specs:
        comparison = paired_bootstrap_compare(
            baseline_values,
            candidate_values,
            higher_is_better=higher_is_better,
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        )
        interval = comparison.interval
        significant_regression = interval.upper < 0
        evidence.append(
            PairedMetricEvidence(
                metric=metric,
                mean_delta=comparison.mean_delta,
                lower=interval.lower,
                upper=interval.upper,
                higher_is_better=higher_is_better,
                significant_improvement=comparison.improved,
                significant_regression=significant_regression,
            )
        )
        if (
            practical_budget is not None
            and comparison.mean_delta < -float(practical_budget)
            and significant_regression
        ):
            reasons.append(
                f"{metric} regression exceeds practical budget with "
                f"{confidence:.0%} paired-bootstrap support"
            )

    return StatisticalComparisonDecision(
        acceptable=not reasons,
        metrics=tuple(evidence),
        reasons=tuple(reasons),
    )
