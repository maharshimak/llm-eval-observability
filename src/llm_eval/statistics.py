from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
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
