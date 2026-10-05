import pytest

from llm_eval.comparison import compare, compare_paired_cases
from llm_eval.models import CaseMetrics, ExperimentSummary


def summary(citations=1):
    return ExperimentSummary("test", 1, 1, citations, 100, 0.001, [])


def test_citation_regression_blocks_release():
    assert not compare(summary(), summary(0.5)).acceptable


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_regression_budgets_rejected(value):
    with pytest.raises(ValueError):
        compare(summary(), summary(), max_pass_rate_drop=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_paired_regression_budgets_rejected(value):
    case = CaseMetrics(
        case_id="1",
        relevance=1.0,
        citation_coverage=1.0,
        forbidden_hit=False,
        latency_ms=100.0,
        estimated_cost_usd=0.001,
        passed=True,
    )
    baseline = ExperimentSummary("base", 1, 1, 1, 100, 0.001, [case])
    candidate = ExperimentSummary("candidate", 1, 1, 1, 100, 0.001, [case])

    with pytest.raises(ValueError):
        compare_paired_cases(
            baseline,
            candidate,
            max_pass_rate_drop=value,
        )
