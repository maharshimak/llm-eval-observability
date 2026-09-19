from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite


@dataclass(frozen=True, slots=True)
class TraceSample:
    latency_ms: float
    cost_usd: float
    success: bool

    def __post_init__(self) -> None:
        for name, value in (("latency_ms", self.latency_ms), ("cost_usd", self.cost_usd)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if type(self.success) is not bool:
            raise TypeError("success must be a boolean")


@dataclass(frozen=True, slots=True)
class SLOPolicy:
    max_p95_latency_ms: float
    max_average_cost_usd: float
    min_success_rate: float
    min_samples: int = 20

    def __post_init__(self) -> None:
        if not isfinite(self.max_p95_latency_ms) or self.max_p95_latency_ms <= 0:
            raise ValueError("max_p95_latency_ms must be finite and positive")
        if not isfinite(self.max_average_cost_usd) or self.max_average_cost_usd < 0:
            raise ValueError("max_average_cost_usd must be finite and non-negative")
        if not isfinite(self.min_success_rate) or not 0 <= self.min_success_rate <= 1:
            raise ValueError("min_success_rate must be between 0 and 1")
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int):
            raise TypeError("min_samples must be an integer")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")


@dataclass(frozen=True, slots=True)
class SLOResult:
    passed: bool
    reasons: tuple[str, ...]
    sample_count: int
    p95_latency_ms: float
    average_cost_usd: float
    success_rate: float


def _p95(samples: Sequence[TraceSample]) -> float:
    ordered = sorted(sample.latency_ms for sample in samples)
    rank = max(1, ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def evaluate_slo(samples: Sequence[TraceSample], policy: SLOPolicy) -> SLOResult:
    if not samples:
        raise ValueError("samples cannot be empty")
    p95_latency = _p95(samples)
    average_cost = sum(sample.cost_usd for sample in samples) / len(samples)
    success_rate = sum(sample.success for sample in samples) / len(samples)

    reasons: list[str] = []
    if len(samples) < policy.min_samples:
        reasons.append(f"sample count {len(samples)} < required {policy.min_samples}")
    if p95_latency > policy.max_p95_latency_ms:
        reasons.append(
            f"p95 latency {p95_latency:.3f}ms > maximum {policy.max_p95_latency_ms:.3f}ms"
        )
    if average_cost > policy.max_average_cost_usd:
        reasons.append(
            f"average cost {average_cost:.6f} > maximum "
            f"{policy.max_average_cost_usd:.6f}"
        )
    if success_rate < policy.min_success_rate:
        reasons.append(
            f"success rate {success_rate:.3f} < minimum {policy.min_success_rate:.3f}"
        )

    return SLOResult(
        passed=not reasons,
        reasons=tuple(reasons),
        sample_count=len(samples),
        p95_latency_ms=p95_latency,
        average_cost_usd=average_cost,
        success_rate=success_rate,
    )
