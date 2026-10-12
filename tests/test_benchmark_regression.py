"""Keep offline scoring and token-cost arithmetic testable in CI."""

from scripts.benchmark_offline_eval import benchmark


def test_synthetic_evaluation_baseline() -> None:
    report = benchmark()
    assert report["cases"] == 3
    assert report["pass_rate"] == 1.0
    assert report["mean_relevance"] == 1.0
    assert report["mean_citation_coverage"] == 1.0
    assert report["latency_p95_ms"] == 120.0
    assert report["mean_cost_usd"] > 0
