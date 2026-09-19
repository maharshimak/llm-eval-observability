from llm_eval.reliability import confidence_gate, wilson_interval


def test_confidence_gate_rewards_large_consistently_passing_samples() -> None:
    result = confidence_gate([True] * 50, min_lower_bound=0.9, min_samples=30)
    assert result.passed is True
    assert result.interval.lower > 0.9


def test_confidence_gate_rejects_small_or_unreliable_samples() -> None:
    result = confidence_gate([True] * 10, min_lower_bound=0.8, min_samples=30)
    assert result.passed is False
    assert any("sample size" in reason for reason in result.reasons)

    mixed = wilson_interval(40, 50)
    assert mixed.lower < 0.8
