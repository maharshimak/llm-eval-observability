from llm_eval.statistics import bootstrap_mean_interval, paired_bootstrap_compare


def test_bootstrap_interval_is_deterministic_and_contains_estimate():
    interval = bootstrap_mean_interval([0.7, 0.8, 0.9, 1.0], resamples=500, seed=7)
    assert interval.lower <= interval.estimate <= interval.upper
    assert interval.samples == 4


def test_paired_comparison_requires_statistically_positive_delta():
    comparison = paired_bootstrap_compare(
        [0.4, 0.5, 0.45, 0.55, 0.5],
        [0.7, 0.8, 0.75, 0.85, 0.8],
        resamples=500,
        seed=3,
    )
    assert comparison.mean_delta > 0
    assert comparison.improved
