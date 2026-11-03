from dataclasses import replace

import pytest

from llm_eval.models import CaseMetrics
from llm_eval.release_gate import GatePolicy, MetricBudget, evaluate_release_gate


def cases(count: int = 50) -> list[CaseMetrics]:
    return [CaseMetrics(f"c{i:02d}", 1.0, 1.0, False, 100.0, 0.001, True) for i in range(count)]


def fail_first(items: list[CaseMetrics], how_many: int, **changes) -> list[CaseMetrics]:
    changes = {"relevance": 0.0, "passed": False, **changes}
    return [replace(c, **changes) if i < how_many else c for i, c in enumerate(items)]


def metric(decision, name):
    return next(item for item in decision.metrics if item.metric == name)


def test_identical_runs_are_allowed():
    decision = evaluate_release_gate(cases(), cases())
    assert decision.decision == "allow"
    assert decision.blocking_reasons == ()
    assert decision.transitions.still_passing == 50


def test_pass_rate_drop_exactly_at_budget_is_allowed():
    # 1 new failure in 50 cases = 0.02 drop, equal to the default 0.02 budget.
    # Naive mean subtraction gives 1.0 - 0.98 = 0.020000000000000018 > 0.02.
    candidate = fail_first(cases(), 1, relevance=0.5)
    decision = evaluate_release_gate(cases(), candidate)
    pass_rate = metric(decision, "pass_rate")
    assert pass_rate.regression == 0.02
    assert not pass_rate.exceeds_budget
    assert decision.decision == "allow"


def test_critical_regression_above_budget_denies_with_hand_checked_statistics():
    candidate = fail_first(cases(), 5)
    decision = evaluate_release_gate(cases(), candidate)
    assert decision.decision == "deny"
    pass_rate = metric(decision, "pass_rate")
    assert pass_rate.delta == -0.1
    assert (pass_rate.mcnemar_worsened, pass_rate.mcnemar_improved) == (5, 0)
    assert pass_rate.mcnemar_p_value == 0.0625  # 2 / 2**5
    assert pass_rate.blocking
    assert any(reason.startswith("pass_rate worsened by 0.1") for reason in decision.blocking_reasons)
    assert decision.transitions.new_failures == ("c00", "c01", "c02", "c03", "c04")


def test_require_significance_only_blocks_supported_regressions():
    candidate = fail_first(cases(), 3)
    budgets = {"pass_rate": MetricBudget(0.0)}
    strict = evaluate_release_gate(cases(), candidate, GatePolicy(budgets=budgets))
    lenient = evaluate_release_gate(
        cases(), candidate, GatePolicy(budgets=budgets, require_significance=True)
    )
    assert strict.decision == "deny"
    # McNemar p = 2 / 2**3 = 0.25, not < 0.05.
    assert metric(lenient, "pass_rate").mcnemar_p_value == 0.25
    assert lenient.decision == "allow"
    assert lenient.warnings and "not statistically supported" in lenient.warnings[0]


def test_non_critical_breach_is_a_warning():
    candidate = [replace(c, latency_ms=900.0) for c in cases()]
    decision = evaluate_release_gate(cases(), candidate)
    latency = metric(decision, "latency_ms")
    assert latency.regression == 800.0
    assert latency.exceeds_budget and not latency.blocking
    assert decision.decision == "allow"
    assert decision.warnings == ("latency_ms worsened by 800 (budget 500, non-critical)",)


def test_new_forbidden_content_blocks_by_default():
    candidate = fail_first(cases(), 1, relevance=1.0, forbidden_hit=True)
    decision = evaluate_release_gate(cases(), candidate)
    assert decision.decision == "deny"
    assert metric(decision, "forbidden_hit_rate").blocking


def test_improvement_is_never_a_regression():
    baseline = fail_first(cases(), 10)
    decision = evaluate_release_gate(baseline, cases())
    assert decision.decision == "allow"
    assert metric(decision, "pass_rate").regression == 0.0
    assert len(decision.transitions.fixed) == 10


def test_gate_is_deterministic():
    candidate = fail_first(cases(), 4, latency_ms=250.0)
    assert evaluate_release_gate(cases(), candidate) == evaluate_release_gate(cases(), candidate)


@pytest.mark.parametrize(
    "candidate",
    [
        cases(49),
        [*cases(49), CaseMetrics("other", 1.0, 1.0, False, 100.0, 0.001, True)],
        [*cases(50), cases(1)[0]],
        fail_first(cases(), 1, relevance=float("nan")),
        fail_first(cases(), 1, relevance=1.5),
        fail_first(cases(), 1, latency_ms=-1.0),
        fail_first(cases(), 1, passed=1),
        [],
    ],
)
def test_invalid_case_sets_rejected(candidate):
    with pytest.raises(ValueError):
        evaluate_release_gate(cases(), candidate)


@pytest.mark.parametrize(
    "build",
    [
        lambda: GatePolicy(budgets={"accuracy": MetricBudget(0.1)}),
        lambda: GatePolicy(budgets={}),
        lambda: MetricBudget(float("nan")),
        lambda: MetricBudget(-0.1),
        lambda: MetricBudget(True),
        lambda: GatePolicy(confidence=1.0),
        lambda: GatePolicy(resamples=10),
    ],
)
def test_invalid_policy_rejected(build):
    with pytest.raises(ValueError):
        build()
