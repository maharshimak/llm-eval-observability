from llm_eval.slo import SLOPolicy, TraceSample, evaluate_slo


def test_slo_gate_passes_healthy_trace_window() -> None:
    samples = [
        TraceSample(latency_ms=100 + index, cost_usd=0.01, success=True)
        for index in range(20)
    ]
    result = evaluate_slo(
        samples,
        SLOPolicy(
            max_p95_latency_ms=150,
            max_average_cost_usd=0.02,
            min_success_rate=0.99,
        ),
    )

    assert result.passed
    assert result.reasons == ()
    assert result.success_rate == 1.0


def test_slo_gate_explains_multiple_failures() -> None:
    samples = [
        TraceSample(latency_ms=500, cost_usd=0.10, success=False),
        TraceSample(latency_ms=600, cost_usd=0.10, success=True),
    ]
    result = evaluate_slo(
        samples,
        SLOPolicy(
            max_p95_latency_ms=200,
            max_average_cost_usd=0.02,
            min_success_rate=0.9,
            min_samples=10,
        ),
    )

    assert not result.passed
    assert len(result.reasons) == 4
    assert result.p95_latency_ms == 600
    assert result.success_rate == 0.5
