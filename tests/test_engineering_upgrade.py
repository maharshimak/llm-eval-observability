import pytest

from llm_eval.comparison import compare
from llm_eval.models import ExperimentSummary


def summary(citations=1):
    return ExperimentSummary("test", 1, 1, citations, 100, 0.001, [])


def test_citation_regression_blocks_release():
    assert not compare(summary(), summary(0.5)).acceptable


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_regression_budgets_rejected(value):
    with pytest.raises(ValueError):
        compare(summary(), summary(), max_pass_rate_drop=value)
