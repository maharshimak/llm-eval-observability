import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import fsum, isfinite
from pathlib import Path

from llm_eval.models import CaseMetrics
from llm_eval.statistics import nearest_rank_percentile


class JsonlTraceStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, experiment: str, metric: CaseMetrics) -> None:
        record = {"experiment": experiment, **asdict(metric)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """One completed model/tool call as reported by a tracing source."""

    span_id: str
    name: str
    latency_ms: float
    success: bool
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.span_id, str) or not self.span_id:
            raise ValueError("span_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.model is not None and (not isinstance(self.model, str) or not self.model):
            raise ValueError("model must be a non-empty string or None")
        if type(self.success) is not bool:
            raise ValueError("success must be a boolean")
        for field_name in ("latency_ms", "cost_usd"):
            value = getattr(self, field_name)
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class TraceAggregate:
    count: int
    success_count: int
    error_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    latency_mean_ms: float
    total_cost_usd: float
    mean_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int


def aggregate_traces(records: Sequence[TraceRecord]) -> TraceAggregate:
    """Aggregate latency/cost/tokens. Percentiles use the nearest-rank method.

    Sums use ``math.fsum`` (correctly rounded), so the result does not depend
    on record order.
    """
    if not records:
        raise ValueError("records must be non-empty")
    latencies = [float(record.latency_ms) for record in records]
    count = len(records)
    successes = sum(1 for record in records if record.success)
    total_cost = fsum(float(record.cost_usd) for record in records)
    return TraceAggregate(
        count=count,
        success_count=successes,
        error_rate=(count - successes) / count,
        latency_p50_ms=nearest_rank_percentile(latencies, 50),
        latency_p95_ms=nearest_rank_percentile(latencies, 95),
        latency_max_ms=max(latencies),
        latency_mean_ms=fsum(latencies) / count,
        total_cost_usd=total_cost,
        mean_cost_usd=total_cost / count,
        total_input_tokens=sum(record.input_tokens for record in records),
        total_output_tokens=sum(record.output_tokens for record in records),
    )


def ingest_traces(
    records: Sequence[TraceRecord],
    *,
    group_by: str | None = None,
) -> tuple[TraceAggregate, dict[str, TraceAggregate]]:
    """Validate span uniqueness and aggregate overall and per ``model`` or ``name``.

    Records without a model are grouped under ``"unknown"`` when grouping by model.
    """
    span_ids = [record.span_id for record in records]
    if len(set(span_ids)) != len(span_ids):
        raise ValueError("span_id values must be unique")
    if group_by not in (None, "model", "name"):
        raise ValueError("group_by must be None, 'model' or 'name'")
    overall = aggregate_traces(records)
    groups: dict[str, list[TraceRecord]] = {}
    if group_by is not None:
        for record in records:
            key = getattr(record, group_by) or "unknown"
            groups.setdefault(key, []).append(record)
    return overall, {key: aggregate_traces(groups[key]) for key in sorted(groups)}
