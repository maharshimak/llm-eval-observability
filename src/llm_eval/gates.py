from dataclasses import dataclass
from math import isfinite

from llm_eval.models import ExperimentSummary


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    reasons: list[str]


def _is_number(value: object) -> bool:
    return type(value) in (int, float) and isfinite(value)


def _validate_thresholds(
    min_pass_rate: float,
    min_relevance: float,
    max_p95_latency_ms: float,
    max_mean_cost_usd: float | None,
) -> None:
    # Every ordered comparison with NaN is False, so an unvalidated NaN threshold
    # would silently pass any summary. Reject invalid configuration instead.
    for name, value in (("min_pass_rate", min_pass_rate), ("min_relevance", min_relevance)):
        if not _is_number(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number between 0 and 1.")
    if not _is_number(max_p95_latency_ms) or max_p95_latency_ms < 0:
        raise ValueError("max_p95_latency_ms must be finite and non-negative.")
    if max_mean_cost_usd is not None and (
        not _is_number(max_mean_cost_usd) or max_mean_cost_usd < 0
    ):
        raise ValueError("max_mean_cost_usd must be finite and non-negative.")


def regression_gate(
    summary: ExperimentSummary,
    *,
    min_pass_rate: float = 0.9,
    min_relevance: float = 0.75,
    max_p95_latency_ms: float = 5000.0,
    max_mean_cost_usd: float | None = None,
) -> GateResult:
    _validate_thresholds(min_pass_rate, min_relevance, max_p95_latency_ms, max_mean_cost_usd)
    reasons: list[str] = []
    if not summary.cases or any(
        not isfinite(v)
        for v in (
            summary.pass_rate,
            summary.mean_relevance,
            summary.mean_citation_coverage,
            summary.latency_p95_ms,
            summary.mean_cost_usd,
        )
    ):
        return GateResult(passed=False, reasons=["Empty or non-finite evaluation summary."])

    if summary.pass_rate < min_pass_rate:
        reasons.append(f"pass_rate {summary.pass_rate:.3f} < required {min_pass_rate:.3f}")
    if summary.mean_relevance < min_relevance:
        reasons.append(
            f"mean_relevance {summary.mean_relevance:.3f} < required {min_relevance:.3f}"
        )
    if summary.latency_p95_ms > max_p95_latency_ms:
        reasons.append(f"p95 latency {summary.latency_p95_ms:.1f}ms > {max_p95_latency_ms:.1f}ms")
    if max_mean_cost_usd is not None and summary.mean_cost_usd > max_mean_cost_usd:
        reasons.append(f"mean cost USD {summary.mean_cost_usd:.6f} > USD {max_mean_cost_usd:.6f}")

    return GateResult(passed=not reasons, reasons=reasons)
