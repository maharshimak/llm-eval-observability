from llm_eval.comparison import compare
from llm_eval.dataset import dataset_fingerprint
from llm_eval.models import EvalCase, ExperimentSummary


def summary(
    name: str,
    pass_rate: float,
    relevance: float,
    latency: float,
) -> ExperimentSummary:
    return ExperimentSummary(
        name=name,
        pass_rate=pass_rate,
        mean_relevance=relevance,
        mean_citation_coverage=1.0,
        latency_p95_ms=latency,
        mean_cost_usd=0.001,
        cases=[],
    )


def test_regression_comparison_blocks_bad_candidate() -> None:
    baseline = summary("baseline", 1.0, 0.95, 300)
    candidate = summary("candidate", 0.8, 0.70, 1200)

    decision = compare(baseline, candidate)

    assert not decision.acceptable
    assert len(decision.reasons) >= 2


def test_dataset_fingerprint_is_order_stable_for_sets() -> None:
    first = EvalCase(
        id="1",
        prompt="q",
        expected_terms={"a", "b"},
    )
    second = EvalCase(
        id="1",
        prompt="q",
        expected_terms={"b", "a"},
    )
    assert dataset_fingerprint([first]) == dataset_fingerprint([second])
