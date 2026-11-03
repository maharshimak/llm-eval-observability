"""Per-metric regression budgets with a critical-regression release gate.

Compares baseline and candidate per-case metrics for the *same* cases. For each
budgeted metric it reports the mean delta, a fixed-seed paired bootstrap
interval for the delta and, for binary outcomes, an exact McNemar test. A
candidate is denied when any *critical* metric regresses by more than its
budget (and, if ``require_significance`` is set, the regression is also
statistically supported). Non-critical budget breaches are reported as warnings.

The gate is fail-closed: unknown metrics, invalid budgets, mismatched case sets
and non-finite or out-of-range measurements raise ``ValueError`` rather than
producing a decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import fsum, isfinite

from llm_eval.models import CaseMetrics
from llm_eval.statistics import paired_binary_mcnemar, paired_bootstrap_compare

# metric -> (higher_is_better, is_binary)
GATE_METRICS: dict[str, tuple[bool, bool]] = {
    "pass_rate": (True, True),
    "relevance": (True, False),
    "citation_coverage": (True, False),
    "forbidden_hit_rate": (False, True),
    "latency_ms": (False, False),
    "cost_usd": (False, False),
}


@dataclass(frozen=True, slots=True)
class MetricBudget:
    """Largest tolerated worsening of the metric's mean, in the metric's own units."""

    max_regression: float
    critical: bool = True

    def __post_init__(self) -> None:
        value = self.max_regression
        if type(value) not in (int, float) or not isfinite(value) or value < 0:
            raise ValueError("max_regression must be finite and non-negative")
        if type(self.critical) is not bool:
            raise ValueError("critical must be a boolean")


def default_budgets() -> dict[str, MetricBudget]:
    return {
        "pass_rate": MetricBudget(0.02, critical=True),
        "relevance": MetricBudget(0.03, critical=True),
        "citation_coverage": MetricBudget(0.03, critical=True),
        "forbidden_hit_rate": MetricBudget(0.0, critical=True),
        "latency_ms": MetricBudget(500.0, critical=False),
    }


@dataclass(frozen=True, slots=True)
class GatePolicy:
    budgets: Mapping[str, MetricBudget] = field(default_factory=default_budgets)
    require_significance: bool = False
    confidence: float = 0.95
    resamples: int = 2000
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.budgets:
            raise ValueError("at least one metric budget is required")
        unknown = sorted(set(self.budgets) - set(GATE_METRICS))
        if unknown:
            raise ValueError(f"unknown gate metrics: {unknown}")
        if any(not isinstance(budget, MetricBudget) for budget in self.budgets.values()):
            raise ValueError("budgets must be MetricBudget instances")
        if type(self.require_significance) is not bool:
            raise ValueError("require_significance must be a boolean")
        if type(self.confidence) is not float or not 0.5 <= self.confidence < 1:
            raise ValueError("confidence must be a float in [0.5, 1)")
        if type(self.resamples) is not int or not 100 <= self.resamples <= 20_000:
            raise ValueError("resamples must be an integer in [100, 20000]")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


@dataclass(frozen=True, slots=True)
class MetricComparison:
    metric: str
    higher_is_better: bool
    baseline_mean: float
    candidate_mean: float
    delta: float  # mean of per-case (candidate - baseline), in the metric's units
    regression: float  # how much worse the candidate is (0 when equal or better)
    ci_lower: float  # paired bootstrap interval for ``delta``
    ci_upper: float
    significantly_worse: bool
    mcnemar_worsened: int | None
    mcnemar_improved: int | None
    mcnemar_p_value: float | None
    max_regression: float
    critical: bool
    exceeds_budget: bool
    blocking: bool


@dataclass(frozen=True, slots=True)
class CaseTransitions:
    new_failures: tuple[str, ...]
    fixed: tuple[str, ...]
    still_failing: tuple[str, ...]
    still_passing: int


@dataclass(frozen=True, slots=True)
class GateDecision:
    decision: str  # "allow" or "deny"
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    case_count: int
    metrics: tuple[MetricComparison, ...]
    transitions: CaseTransitions


def _validated_cases(cases: Sequence[CaseMetrics], side: str) -> dict[str, CaseMetrics]:
    if not cases:
        raise ValueError(f"{side} must contain at least one case")
    mapped: dict[str, CaseMetrics] = {}
    for case in cases:
        if not isinstance(case, CaseMetrics):
            raise TypeError(f"{side} cases must be CaseMetrics")
        if case.case_id in mapped:
            raise ValueError(f"{side} case IDs must be unique")
        for name in ("relevance", "citation_coverage"):
            value = getattr(case, name)
            if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{side} {name} must be finite and within [0, 1]")
        for name in ("latency_ms", "estimated_cost_usd"):
            value = getattr(case, name)
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{side} {name} must be finite and non-negative")
        if type(case.passed) is not bool or type(case.forbidden_hit) is not bool:
            raise ValueError(f"{side} passed/forbidden_hit must be booleans")
        mapped[case.case_id] = case
    return mapped


def _values(case: CaseMetrics, metric: str) -> float:
    if metric == "pass_rate":
        return 1.0 if case.passed else 0.0
    if metric == "forbidden_hit_rate":
        return 1.0 if case.forbidden_hit else 0.0
    if metric == "latency_ms":
        return float(case.latency_ms)
    if metric == "cost_usd":
        return float(case.estimated_cost_usd)
    return float(getattr(case, metric))


def evaluate_release_gate(
    baseline: Sequence[CaseMetrics],
    candidate: Sequence[CaseMetrics],
    policy: GatePolicy | None = None,
) -> GateDecision:
    policy = policy or GatePolicy()
    base = _validated_cases(baseline, "baseline")
    cand = _validated_cases(candidate, "candidate")
    if set(base) != set(cand):
        raise ValueError("baseline and candidate must contain the same case IDs")
    case_ids = sorted(base)
    alpha = 1.0 - policy.confidence

    comparisons: list[MetricComparison] = []
    blocking: list[str] = []
    warnings: list[str] = []
    for metric in sorted(policy.budgets):
        budget = policy.budgets[metric]
        higher_is_better, is_binary = GATE_METRICS[metric]
        old = [_values(base[case_id], metric) for case_id in case_ids]
        new = [_values(cand[case_id], metric) for case_id in case_ids]
        baseline_mean = fsum(old) / len(old)
        candidate_mean = fsum(new) / len(new)
        # Mean of per-case differences: exact for 0/1 outcomes, so a pass-rate
        # drop of exactly the budget is not pushed over it by rounding.
        delta = fsum(b - a for a, b in zip(old, new, strict=True)) / len(old)
        regression = max(0.0, -delta if higher_is_better else delta)
        # Raw delta (candidate - baseline) interval; orientation handled below.
        bootstrap = paired_bootstrap_compare(
            old,
            new,
            higher_is_better=True,
            confidence=policy.confidence,
            resamples=policy.resamples,
            seed=policy.seed,
        )
        lower, upper = bootstrap.interval.lower, bootstrap.interval.upper
        worsened = improved = None
        p_value = None
        if is_binary:
            good_value = 1.0 if higher_is_better else 0.0
            test = paired_binary_mcnemar(
                [value == good_value for value in old], [value == good_value for value in new]
            )
            worsened, improved, p_value = test.worsened, test.improved, test.p_value
            significantly_worse = test.p_value < alpha and test.worsened > test.improved
        else:
            significantly_worse = upper < 0 if higher_is_better else lower > 0
        exceeds = regression > budget.max_regression
        blocks = (
            exceeds
            and budget.critical
            and (significantly_worse or not policy.require_significance)
        )
        message = (
            f"{metric} worsened by {regression:.6g} "
            f"(budget {budget.max_regression:.6g}, "
            f"{'critical' if budget.critical else 'non-critical'})"
        )
        if blocks:
            blocking.append(message)
        elif exceeds:
            suffix = "" if not budget.critical else " but is not statistically supported"
            warnings.append(message + suffix)
        comparisons.append(
            MetricComparison(
                metric=metric,
                higher_is_better=higher_is_better,
                baseline_mean=baseline_mean,
                candidate_mean=candidate_mean,
                delta=delta,
                regression=regression,
                ci_lower=lower,
                ci_upper=upper,
                significantly_worse=significantly_worse,
                mcnemar_worsened=worsened,
                mcnemar_improved=improved,
                mcnemar_p_value=p_value,
                max_regression=float(budget.max_regression),
                critical=budget.critical,
                exceeds_budget=exceeds,
                blocking=blocks,
            )
        )

    transitions = CaseTransitions(
        new_failures=tuple(i for i in case_ids if base[i].passed and not cand[i].passed),
        fixed=tuple(i for i in case_ids if not base[i].passed and cand[i].passed),
        still_failing=tuple(i for i in case_ids if not base[i].passed and not cand[i].passed),
        still_passing=sum(1 for i in case_ids if base[i].passed and cand[i].passed),
    )
    return GateDecision(
        decision="deny" if blocking else "allow",
        blocking_reasons=tuple(blocking),
        warnings=tuple(warnings),
        case_count=len(case_ids),
        metrics=tuple(comparisons),
        transitions=transitions,
    )
