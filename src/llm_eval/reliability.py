from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float
    sample_size: int
    successes: int


@dataclass(frozen=True, slots=True)
class ConfidenceGateResult:
    passed: bool
    reasons: tuple[str, ...]
    interval: ConfidenceInterval


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> ConfidenceInterval:
    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or not isinstance(successes, int)
        or not isinstance(total, int)
    ):
        raise TypeError("successes and total must be integers")
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Require 0 <= successes <= total and total > 0")
    if not isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")

    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    margin = z * sqrt((p * (1 - p) / total) + z2 / (4 * total * total)) / denominator
    return ConfidenceInterval(
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        sample_size=total,
        successes=successes,
    )


def confidence_gate(
    outcomes: Sequence[bool],
    *,
    min_lower_bound: float = 0.8,
    min_samples: int = 30,
    z: float = 1.96,
) -> ConfidenceGateResult:
    if not outcomes or any(type(value) is not bool for value in outcomes):
        raise ValueError("outcomes must be a non-empty sequence of booleans")
    if not 0 <= min_lower_bound <= 1:
        raise ValueError("min_lower_bound must be between 0 and 1")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples <= 0:
        raise ValueError("min_samples must be a positive integer")

    interval = wilson_interval(sum(outcomes), len(outcomes), z=z)
    reasons: list[str] = []
    if len(outcomes) < min_samples:
        reasons.append(f"sample size {len(outcomes)} < required {min_samples}")
    if interval.lower < min_lower_bound:
        reasons.append(
            f"Wilson lower bound {interval.lower:.3f} < required {min_lower_bound:.3f}"
        )
    return ConfidenceGateResult(passed=not reasons, reasons=tuple(reasons), interval=interval)
