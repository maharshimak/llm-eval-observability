from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb, isfinite
from statistics import mean


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int


@dataclass(frozen=True, slots=True)
class PairedComparison:
    mean_delta: float
    interval: ConfidenceInterval
    improved: bool


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 42,
) -> ConfidenceInterval:
    if not values:
        raise ValueError("values must be non-empty")
    numeric = [float(value) for value in values]
    if any(not isfinite(value) for value in numeric):
        raise ValueError("values must be finite")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("resamples must be an integer >= 100")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    rng = random.Random(seed)
    size = len(numeric)
    means = sorted(
        mean(numeric[rng.randrange(size)] for _ in range(size))
        for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, min(resamples - 1, int(alpha * resamples)))
    upper_index = max(
        0,
        min(resamples - 1, int((1.0 - alpha) * resamples) - 1),
    )
    return ConfidenceInterval(
        estimate=mean(numeric),
        lower=means[lower_index],
        upper=means[upper_index],
        confidence=confidence,
        samples=size,
    )


def paired_bootstrap_compare(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    higher_is_better: bool = True,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 42,
) -> PairedComparison:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("baseline and candidate must be paired non-empty sequences")
    deltas = [
        float(new) - float(old) if higher_is_better else float(old) - float(new)
        for old, new in zip(baseline, candidate, strict=True)
    ]
    interval = bootstrap_mean_interval(
        deltas,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    return PairedComparison(
        mean_delta=interval.estimate,
        interval=interval,
        improved=interval.lower > 0,
    )


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Exact (binomial) McNemar test on paired binary outcomes.

    ``worsened`` counts pairs that were good in the baseline and bad in the
    candidate; ``improved`` counts the reverse. Concordant pairs carry no
    information about a difference and are ignored by the test.
    """

    worsened: int
    improved: int
    p_value: float


def mcnemar_exact(worsened: int, improved: int) -> McNemarResult:
    """Two-sided exact McNemar p-value: min(1, 2 * P[X <= min(b, c)]), X ~ Bin(b + c, 1/2).

    Computed with integer arithmetic, so the result is exact up to the final
    float conversion. With no discordant pairs the p-value is 1.
    """
    for value in (worsened, improved):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("discordant counts must be non-negative integers")
    total = worsened + improved
    if total == 0:
        return McNemarResult(worsened=0, improved=0, p_value=1.0)
    tail = sum(comb(total, k) for k in range(min(worsened, improved) + 1))
    p_value = Fraction(min(2 * tail, 2**total), 2**total)
    return McNemarResult(worsened=worsened, improved=improved, p_value=float(p_value))


def paired_binary_mcnemar(baseline: Sequence[bool], candidate: Sequence[bool]) -> McNemarResult:
    """McNemar test where True is the good outcome (for example, a passed case)."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("baseline and candidate must be paired non-empty sequences")
    if any(type(value) is not bool for value in (*baseline, *candidate)):
        raise ValueError("outcomes must be booleans")
    worsened = sum(1 for old, new in zip(baseline, candidate, strict=True) if old and not new)
    improved = sum(1 for old, new in zip(baseline, candidate, strict=True) if new and not old)
    return mcnemar_exact(worsened, improved)


def nearest_rank_percentile(values: Sequence[float], percent: int) -> float:
    """Nearest-rank percentile: the smallest value with at least ``percent``% of values <= it.

    The rank ``ceil(percent * n / 100)`` is computed with integer arithmetic so
    it is never perturbed by binary floating point.
    """
    if isinstance(percent, bool) or not isinstance(percent, int) or not 1 <= percent <= 100:
        raise ValueError("percent must be an integer in [1, 100]")
    if not values:
        raise ValueError("values must be non-empty")
    numeric = [float(value) for value in values]
    if any(not isfinite(value) for value in numeric):
        raise ValueError("values must be finite")
    ordered = sorted(numeric)
    rank = -(-percent * len(ordered) // 100)
    return ordered[rank - 1]
