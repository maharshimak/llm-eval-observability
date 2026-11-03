import pytest

from llm_eval.traces import TraceRecord, aggregate_traces, ingest_traces


def record(index: int, **overrides) -> TraceRecord:
    values = {
        "span_id": f"s{index}",
        "name": "generate",
        "latency_ms": float(index),
        "success": index % 5 != 0,
        "cost_usd": 0.1,
        "input_tokens": 10,
        "output_tokens": 2,
        "model": "m-a" if index % 2 else "m-b",
    }
    values.update(overrides)
    return TraceRecord(**values)


def test_aggregate_hand_computed():
    records = [record(i) for i in range(1, 21)]
    aggregate = aggregate_traces(records)
    assert aggregate.count == 20
    assert aggregate.success_count == 16  # 5, 10, 15, 20 failed
    assert aggregate.error_rate == 0.2
    assert aggregate.latency_p50_ms == 10.0  # rank ceil(0.5 * 20) = 10
    assert aggregate.latency_p95_ms == 19.0  # rank ceil(0.95 * 20) = 19
    assert aggregate.latency_max_ms == 20.0
    assert aggregate.latency_mean_ms == 10.5
    assert aggregate.total_cost_usd == 2.0  # fsum is correctly rounded
    assert aggregate.mean_cost_usd == 0.1
    assert aggregate.total_input_tokens == 200
    assert aggregate.total_output_tokens == 40


def test_aggregate_is_independent_of_record_order():
    records = [record(i, cost_usd=0.1 * i) for i in range(1, 40)]
    assert aggregate_traces(records) == aggregate_traces(list(reversed(records)))


def test_group_by_model_with_unknown_bucket():
    records = [record(1), record(2), record(3, model=None)]
    overall, groups = ingest_traces(records, group_by="model")
    assert overall.count == 3
    assert sorted(groups) == ["m-a", "m-b", "unknown"]
    assert groups["m-a"].count == 1
    assert groups["unknown"].latency_p95_ms == 3.0


def test_duplicate_span_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        ingest_traces([record(1), record(1)])


def test_invalid_group_by_rejected():
    with pytest.raises(ValueError):
        ingest_traces([record(1)], group_by="span_id")


@pytest.mark.parametrize(
    "overrides",
    [
        {"latency_ms": float("nan")},
        {"latency_ms": -1.0},
        {"latency_ms": True},
        {"cost_usd": float("inf")},
        {"input_tokens": 1.5},
        {"output_tokens": -1},
        {"success": 1},
        {"span_id": ""},
        {"model": ""},
    ],
)
def test_invalid_records_rejected(overrides):
    with pytest.raises(ValueError):
        record(1, **overrides)


def test_empty_records_rejected():
    with pytest.raises(ValueError):
        aggregate_traces([])
