import random

from llm_eval.models import EvalCase, ModelOutput
from llm_eval.runner import ExperimentRunner
from llm_eval.taxonomy import FAILURE_CATEGORIES, classify_case, count_failures

WORDS = ["alpha", "beta", "gamma", "delta", "unsafe"]


def test_categories_are_empty_exactly_when_the_runner_passes_a_case():
    rng = random.Random(5)
    for _ in range(50):
        runner = ExperimentRunner(
            min_relevance=rng.choice([0.0, 0.5, 1.0]),
            min_citation_coverage=rng.choice([0.0, 0.5, 1.0]),
            max_latency_ms=rng.choice([0.0, 100.0]),
        )
        cases = [
            EvalCase(
                str(i),
                "q",
                expected_terms=set(rng.sample(WORDS[:4], rng.randint(0, 3))),
                expected_citations=set(rng.sample(["d1", "d2"], rng.randint(0, 2))),
                forbidden_phrases={"unsafe"} if rng.random() < 0.5 else set(),
            )
            for i in range(20)
        ]
        outputs = {
            case.id: ModelOutput(
                " ".join(rng.sample(WORDS, rng.randint(0, 5))),
                rng.sample(["d1", "d2", "d3"], rng.randint(0, 3)),
                rng.choice([0.0, 50.0, 100.0, 150.0]),
                0,
                0,
            )
            for case in cases
        }
        summary = runner.run("prop", cases, lambda c, o=outputs: o[c.id])
        for metrics in summary.cases:
            categories = classify_case(
                metrics,
                min_relevance=runner.min_relevance,
                min_citation_coverage=runner.min_citation_coverage,
                max_latency_ms=runner.max_latency_ms,
            )
            assert (not categories) == metrics.passed
            assert list(categories) == [c for c in FAILURE_CATEGORIES if c in categories]


def test_count_failures_includes_every_category():
    counts = count_failures([("low_relevance",), (), ("low_relevance", "forbidden_content")])
    assert counts == {
        "forbidden_content": 1,
        "low_relevance": 2,
        "missing_expected_citation": 0,
        "latency_budget_exceeded": 0,
    }
