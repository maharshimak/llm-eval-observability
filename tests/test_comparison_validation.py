from dataclasses import replace

import pytest

from llm_eval.comparison import compare, compare_paired_cases
from llm_eval.models import CaseMetrics, ExperimentSummary


def summary():
    case = CaseMetrics("case", 1.0, 1.0, False, 100.0, 0.001, True)
    return ExperimentSummary("test", 1.0, 1.0, 1.0, 100.0, 0.001, [case])


@pytest.mark.parametrize("field", [
    "pass_rate", "mean_relevance", "mean_citation_coverage", "latency_p95_ms", "mean_cost_usd",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_aggregate_comparison_rejects_nonfinite_measurements(field, value, side):
    valid = summary()
    invalid = replace(valid, **{field: value})
    with pytest.raises(ValueError, match="finite"):
        compare(invalid if side == "baseline" else valid, invalid if side == "candidate" else valid)


@pytest.mark.parametrize("budget", [
    "max_pass_rate_drop", "max_relevance_drop", "max_citation_drop",
    "max_latency_increase_ms", "max_cost_increase_usd",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "0.1"])
def test_paired_comparison_rejects_invalid_regression_budgets(budget, value):
    with pytest.raises(ValueError, match="budgets"):
        compare_paired_cases(summary(), summary(), resamples=100, **{budget: value})


@pytest.mark.parametrize("comparison", [compare, compare_paired_cases])
@pytest.mark.parametrize("budget", [
    "max_pass_rate_drop", "max_relevance_drop", "max_citation_drop", "max_latency_increase_ms",
])
def test_required_regression_budgets_cannot_be_disabled(comparison, budget):
    with pytest.raises(ValueError, match="budgets"):
        comparison(summary(), summary(), **{budget: None})


def test_paired_comparison_accepts_valid_zero_budgets_and_optional_cost_budget():
    assert compare_paired_cases(
        summary(), summary(), resamples=100,
        max_pass_rate_drop=0, max_relevance_drop=0, max_citation_drop=0,
        max_latency_increase_ms=0, max_cost_increase_usd=None,
    ).acceptable


def test_paired_comparison_still_blocks_a_regression_after_validation():
    baseline = summary()
    candidate = replace(baseline, cases=[replace(baseline.cases[0], relevance=0.0, passed=False)])
    assert not compare_paired_cases(baseline, candidate, resamples=100).acceptable
