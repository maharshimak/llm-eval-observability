from llm_eval.comparison import compare, compare_paired_cases
from llm_eval.dataset import dataset_fingerprint
from llm_eval.models import CaseMetrics, EvalCase, ExperimentSummary


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


def paired_summary(name: str, relevance_values: list[float], latencies: list[float]) -> ExperimentSummary:
    cases = [
        CaseMetrics(
            case_id=str(index),
            relevance=relevance,
            citation_coverage=1.0,
            forbidden_hit=False,
            latency_ms=latency,
            estimated_cost_usd=0.001,
            passed=relevance >= 0.7,
        )
        for index, (relevance, latency) in enumerate(
            zip(relevance_values, latencies, strict=True)
        )
    ]
    return ExperimentSummary(
        name=name,
        pass_rate=sum(case.passed for case in cases) / len(cases),
        mean_relevance=sum(case.relevance for case in cases) / len(cases),
        mean_citation_coverage=1.0,
        latency_p95_ms=max(latencies),
        mean_cost_usd=0.001,
        cases=cases,
    )


def test_paired_comparison_blocks_supported_relevance_regression() -> None:
    baseline = paired_summary(
        "baseline",
        [0.95, 0.94, 0.96, 0.93, 0.97, 0.95],
        [100, 110, 105, 108, 102, 107],
    )
    candidate = paired_summary(
        "candidate",
        [0.70, 0.69, 0.71, 0.68, 0.72, 0.70],
        [105, 112, 109, 110, 107, 111],
    )

    decision = compare_paired_cases(
        baseline,
        candidate,
        resamples=500,
        seed=3,
    )

    assert not decision.acceptable
    assert any("relevance regression" in reason for reason in decision.reasons)
    relevance = next(item for item in decision.metrics if item.metric == "relevance")
    assert relevance.significant_regression
