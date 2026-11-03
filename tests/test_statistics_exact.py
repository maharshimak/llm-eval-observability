"""Hand-computed checks for the exact McNemar test and nearest-rank percentile."""

import random

import pytest

from llm_eval.runner import percentile
from llm_eval.slo import TraceSample, _p95
from llm_eval.statistics import mcnemar_exact, nearest_rank_percentile, paired_binary_mcnemar


@pytest.mark.parametrize(
    ("worsened", "improved", "expected"),
    [
        (0, 0, 1.0),  # no discordant pairs
        (3, 0, 0.25),  # 2 * C(3,0) / 2**3 = 2/8
        (5, 1, 0.21875),  # 2 * (C(6,0) + C(6,1)) / 2**6 = 14/64
        (1, 5, 0.21875),  # symmetric
        (10, 0, 0.001953125),  # 2 / 1024
        (2, 2, 1.0),  # 2 * 11/16 capped at 1
        (7, 3, 0.34375),  # 2 * (1 + 10 + 45 + 120) / 1024 = 352/1024
    ],
)
def test_mcnemar_exact_matches_hand_computation(worsened, improved, expected):
    assert mcnemar_exact(worsened, improved).p_value == expected


@pytest.mark.parametrize("bad", [-1, 1.0, True, None])
def test_mcnemar_rejects_invalid_counts(bad):
    with pytest.raises(ValueError):
        mcnemar_exact(bad, 0)


def test_paired_binary_mcnemar_counts_discordant_pairs():
    result = paired_binary_mcnemar(
        [True, True, True, False, False], [False, False, True, True, False]
    )
    assert (result.worsened, result.improved) == (2, 1)
    assert result.p_value == 1.0  # 2 * (1 + 3) / 8 capped


def test_paired_binary_mcnemar_rejects_non_bool_and_unpaired():
    with pytest.raises(ValueError):
        paired_binary_mcnemar([1, 0], [0, 1])
    with pytest.raises(ValueError):
        paired_binary_mcnemar([True], [True, False])


@pytest.mark.parametrize(
    ("values", "percent", "expected"),
    [
        (list(range(1, 21)), 95, 19),  # rank ceil(0.95 * 20) = 19
        (list(range(1, 21)), 50, 10),
        (list(range(1, 11)), 50, 5),
        ([3, 1, 2], 50, 2),
        (list(range(1, 101)), 95, 95),
        ([5, 9, 1], 100, 9),
        ([7], 95, 7),
        ([10, 20], 1, 10),
    ],
)
def test_nearest_rank_percentile(values, percent, expected):
    assert nearest_rank_percentile(values, percent) == expected


@pytest.mark.parametrize("percent", [0, 101, 0.95, True])
def test_nearest_rank_percentile_rejects_invalid_percent(percent):
    with pytest.raises(ValueError):
        nearest_rank_percentile([1.0], percent)


def test_nearest_rank_percentile_rejects_empty_and_non_finite():
    with pytest.raises(ValueError):
        nearest_rank_percentile([], 50)
    with pytest.raises(ValueError):
        nearest_rank_percentile([1.0, float("nan")], 50)


def test_p95_definitions_agree_across_modules():
    rng = random.Random(11)
    for size in range(1, 301):
        values = [rng.uniform(0, 1000) for _ in range(size)]
        expected = nearest_rank_percentile(values, 95)
        assert percentile(values, 0.95) == expected
        samples = [TraceSample(latency_ms=value, cost_usd=0.0, success=True) for value in values]
        assert _p95(samples) == expected
