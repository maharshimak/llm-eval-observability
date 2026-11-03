"""Regression tests for defects confirmed on main (each failed before its fix)."""

import json
from hashlib import sha256

import pytest

from llm_eval.dataset import dataset_fingerprint
from llm_eval.gates import regression_gate
from llm_eval.metrics import relevance
from llm_eval.models import CaseMetrics, EvalCase, ExperimentSummary, ModelOutput
from llm_eval.runner import ExperimentRunner
from llm_eval.versioning import build_versioned_dataset, run_versioned_experiment

NAN = float("nan")


def failing_summary() -> ExperimentSummary:
    case = CaseMetrics("a", 0.0, 0.0, True, 1e9, 1.0, False)
    return ExperimentSummary("bad", 0.0, 0.0, 0.0, 1e9, 1.0, [case])


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_pass_rate": NAN},
        {"min_relevance": NAN},
        {"max_p95_latency_ms": NAN},
        {"max_mean_cost_usd": NAN},
        {"min_pass_rate": -0.1},
        {"min_relevance": 1.5},
        {"max_p95_latency_ms": -1.0},
        {"max_mean_cost_usd": float("inf")},
        {"min_pass_rate": True},
    ],
)
def test_regression_gate_rejects_invalid_thresholds_instead_of_passing(overrides):
    # On main, NaN thresholds made every comparison False, so this completely
    # failing summary passed the gate.
    with pytest.raises(ValueError):
        regression_gate(failing_summary(), **overrides)


def test_regression_gate_still_blocks_with_valid_thresholds():
    result = regression_gate(failing_summary(), max_mean_cost_usd=0.5)
    assert not result.passed
    assert len(result.reasons) == 4


def test_manifest_reports_mixed_provider_when_first_output_is_unreported():
    dataset = build_versioned_dataset("d", "1", [EvalCase("a", "q"), EvalCase("b", "q")])
    outputs = {
        "a": ModelOutput("x", [], 1, 0, 0, provider=None, model=None),
        "b": ModelOutput("x", [], 1, 0, 0, provider="vendor", model="m1"),
    }
    run = run_versioned_experiment(
        ExperimentRunner(), name="n", dataset=dataset, candidate=lambda c: outputs[c.id]
    )
    # On main this was recorded as provider="vendor", model="m1".
    assert run.manifest.provider == "mixed"
    assert run.manifest.model == "mixed"


def test_manifest_keeps_single_unreported_provider_as_none():
    dataset = build_versioned_dataset("d", "1", [EvalCase("a", "q"), EvalCase("b", "q")])
    run = run_versioned_experiment(
        ExperimentRunner(),
        name="n",
        dataset=dataset,
        candidate=lambda c: ModelOutput("x", [], 1, 0, 0),
    )
    assert run.manifest.provider is None
    assert run.manifest.model is None


def test_both_dataset_fingerprint_functions_agree_for_non_ascii_text():
    cases = [EvalCase("a", "Qu'est-ce qu'un café ?", expected_terms={"café", "über"})]
    # On main dataset_fingerprint escaped non-ASCII while VersionedDataset did not.
    assert dataset_fingerprint(cases) == build_versioned_dataset("d", "1", cases).fingerprint


def test_ascii_dataset_fingerprint_is_unchanged_from_previous_encoding():
    cases = [EvalCase("a", "What is RAG?", {"retrieval", "rag"}, {"d1"}, {"never"})]
    legacy = json.dumps(
        [
            {
                "id": "a",
                "prompt": "What is RAG?",
                "expected_terms": ["rag", "retrieval"],
                "expected_citations": ["d1"],
                "forbidden_phrases": ["never"],
            }
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert dataset_fingerprint(cases) == sha256(legacy.encode("utf-8")).hexdigest()


def output(text: str) -> ModelOutput:
    return ModelOutput(text, [], 0, 0, 0)


def test_multi_word_expected_term_can_match():
    case = EvalCase("a", "q", expected_terms={"machine learning"})
    # On main a multi-word term could never match, scoring an exact answer 0.0.
    assert relevance(case, output("Machine learning is useful.")) == 1.0


def test_multi_word_term_requires_contiguous_tokens():
    case = EvalCase("a", "q", expected_terms={"machine learning"})
    assert relevance(case, output("learning about a machine")) == 0.0


def test_single_word_terms_and_punctuated_terms():
    case = EvalCase("a", "q", expected_terms={"RAG", "rag.", "canary"})
    # "RAG" and "rag." normalise to the same term, so there are two distinct terms.
    assert relevance(case, output("rag only")) == 0.5


def test_term_without_word_characters_never_matches():
    case = EvalCase("a", "q", expected_terms={"!!!", "ok"})
    assert relevance(case, output("ok !!!")) == 0.5
