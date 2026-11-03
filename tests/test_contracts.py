"""makma.v1 contract tests: real run_operation outputs validated against the vendored schema."""

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from llm_eval.canonical import canonical_json_bytes
from llm_eval.contracts import capabilities, run_operation
from llm_eval.dataset import dataset_fingerprint
from llm_eval.models import EvalCase

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "contracts" / "makma.v1.schema.json").read_text(
        encoding="utf-8"
    )
)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())

DATASET = {
    "id": "synthetic-rag",
    "version": "2026.11",
    "cases": [
        {
            "id": "rag",
            "prompt": "What is retrieval augmented generation?",
            "expected_terms": ["retrieval", "generation"],
            "expected_citations": ["doc-rag"],
        },
        {
            "id": "ops",
            "prompt": "What makes a canary release safe?",
            "expected_terms": ["canary", "rollback"],
            "forbidden_phrases": ["guaranteed"],
        },
        {"id": "privacy", "prompt": "How are columns protected?", "expected_terms": ["masking"]},
    ],
}

GOOD = [
    {
        "case_id": "rag",
        "output": "Retrieval feeds generation [doc-rag].",
        "citations": ["doc-rag"],
        "latency_ms": 120,
        "input_tokens": 30,
        "output_tokens": 8,
    },
    {"case_id": "ops", "output": "A canary with automatic rollback.", "latency_ms": 95.5},
    {"case_id": "privacy", "output": "Column masking.", "latency_ms": 80},
]

BAD = [
    {"case_id": "rag", "output": "It is a database.", "latency_ms": 130},
    {"case_id": "ops", "output": "Canary rollback is guaranteed.", "latency_ms": 99},
    {"case_id": "privacy", "output": "Column masking.", "latency_ms": 82},
]

EMPTY_CITATIONS = {"cited_ids": [], "evidence_ids": []}


def valid(envelope: dict) -> dict:
    VALIDATOR.validate(envelope)
    assert envelope["system"] == "llm-eval-observability"
    assert envelope["execution_mode"] == "deterministic_offline"
    started = envelope["started_at"]
    assert started.endswith("Z")
    datetime.fromisoformat(started)
    return envelope


def evaluate(predictions, **extra):
    payload = {"dataset": DATASET, "predictions": predictions, **extra}
    return valid(run_operation("evaluate", payload))


def test_vendored_schema_is_a_valid_draft_2020_12_schema():
    Draft202012Validator.check_schema(SCHEMA)


def test_capabilities_list_real_operations_with_input_schemas():
    caps = capabilities()
    assert caps["contract_version"] == "makma.v1"
    assert caps["system"] == "llm-eval-observability"
    names = [op["name"] for op in caps["operations"]]
    assert names == [
        "evaluate",
        "compare",
        "validate_citations",
        "score_retrieval",
        "ingest_traces",
    ]
    for op in caps["operations"]:
        Draft202012Validator.check_schema(op["input_schema"])
        assert op["side_effects"] is False
        assert op["requires_approval"] is False


def test_evaluate_ok_with_fingerprint_taxonomy_and_provenance():
    envelope = evaluate(BAD)
    assert envelope["status"] == "ok"
    assert envelope["error"] is None
    data = envelope["data"]
    expected_fp = dataset_fingerprint(
        [
            EvalCase(
                c["id"],
                c["prompt"],
                set(c.get("expected_terms", [])),
                set(c.get("expected_citations", [])),
                set(c.get("forbidden_phrases", [])),
            )
            for c in DATASET["cases"]
        ]
    )
    assert data["dataset"]["fingerprint"] == expected_fp
    assert data["scoring"]["model_judge"] == "not_used"
    by_id = {case["case_id"]: case for case in data["cases"]}
    assert by_id["rag"]["failure_categories"] == ["low_relevance", "missing_expected_citation"]
    assert by_id["ops"]["failure_categories"] == ["forbidden_content"]
    assert by_id["privacy"]["failure_categories"] == []
    assert data["failure_counts"]["low_relevance"] == 1
    assert data["summary"]["latency_p50_ms"] == 99
    refs = {(ref["kind"], ref["id"]): ref["sha256"] for ref in envelope["provenance"]}
    assert refs[("dataset", "synthetic-rag@2026.11")] == expected_fp
    assert refs[("artifact", "evaluate-report")] == sha256(canonical_json_bytes(data)).hexdigest()


def test_reports_are_byte_identical_when_rerun_from_stored_inputs():
    first, second = evaluate(GOOD), evaluate(GOOD)
    assert first["run_id"] != second["run_id"]
    assert canonical_json_bytes(first["data"]) == canonical_json_bytes(second["data"])
    baseline, candidate = first["data"], evaluate(BAD)["data"]
    one = run_operation("compare", {"baseline": baseline, "candidate": candidate})
    two = run_operation("compare", {"baseline": baseline, "candidate": candidate})
    assert canonical_json_bytes(one["data"]) == canonical_json_bytes(two["data"])


def test_compare_denies_candidate_with_critical_regression():
    baseline, candidate = evaluate(GOOD)["data"], evaluate(BAD)["data"]
    envelope = valid(run_operation("compare", {"baseline": baseline, "candidate": candidate}))
    assert envelope["status"] == "ok"
    assert envelope["data"]["decision"] == "deny"
    assert envelope["policy"]["decision"] == "deny"
    assert envelope["policy"]["reasons"] == envelope["data"]["blocking_reasons"]
    assert envelope["data"]["transitions"]["new_failures"] == ["ops", "rag"]


def test_compare_allows_identical_candidate():
    report = evaluate(GOOD)["data"]
    envelope = valid(run_operation("compare", {"baseline": report, "candidate": report}))
    assert envelope["policy"]["decision"] == "allow"


def test_compare_rejects_different_datasets():
    baseline = evaluate(GOOD)["data"]
    changed_case = {**DATASET["cases"][2], "prompt": "How are rows protected?"}
    changed = {**DATASET, "cases": [*DATASET["cases"][:2], changed_case]}
    candidate = valid(run_operation("evaluate", {"dataset": changed, "predictions": GOOD}))
    envelope = valid(
        run_operation("compare", {"baseline": baseline, "candidate": candidate["data"]})
    )
    assert envelope["status"] == "error"
    assert envelope["data"] is None
    assert envelope["error"]["code"] == "DATASET_MISMATCH"


def test_fingerprint_identifies_content_not_version_label():
    relabeled = {**DATASET, "version": "2026.12"}
    first = evaluate(GOOD)["data"]["dataset"]
    second = valid(run_operation("evaluate", {"dataset": relabeled, "predictions": GOOD}))
    assert second["data"]["dataset"]["fingerprint"] == first["fingerprint"]
    assert second["data"]["dataset"]["version"] == "2026.12"


def test_compare_rejects_different_thresholds():
    baseline = evaluate(GOOD)["data"]
    candidate = evaluate(GOOD, thresholds={"min_relevance": 0.5})["data"]
    envelope = valid(run_operation("compare", {"baseline": baseline, "candidate": candidate}))
    assert envelope["error"]["code"] == "THRESHOLD_MISMATCH"


def test_compare_rejects_tampered_pass_flag():
    baseline = evaluate(GOOD)["data"]
    tampered = evaluate(BAD)["data"]
    tampered["cases"][0]["passed"] = True
    envelope = valid(run_operation("compare", {"baseline": baseline, "candidate": tampered}))
    assert envelope["status"] == "error"
    assert envelope["error"]["code"] == "INVALID_INPUT"


def test_compare_rejects_unknown_budget_metric():
    report = evaluate(GOOD)["data"]
    policy = {"budgets": {"accuracy": {"max_regression": 0.1}}}
    envelope = valid(
        run_operation("compare", {"baseline": report, "candidate": report, "policy": policy})
    )
    assert envelope["error"]["code"] == "INVALID_INPUT"


def test_validate_citations_operation():
    payload = {"cited_ids": ["d1", "dx"], "evidence_ids": ["d1"]}
    envelope = valid(run_operation("validate_citations", payload))
    assert envelope["data"]["valid"] is False
    assert envelope["data"]["unknown_ids"] == ["dx"]
    assert envelope["data"]["check"] == "identifier_subset_only_no_entailment"


def test_score_retrieval_operation_macro_averages():
    payload = {
        "k": 2,
        "queries": [
            {"query_id": "q1", "retrieved_ids": ["a", "b"], "relevant_ids": ["a"]},
            {"query_id": "q2", "retrieved_ids": ["x", "y", "c"], "relevant_ids": ["c"]},
        ],
    }
    mean = valid(run_operation("score_retrieval", payload))["data"]["mean"]
    assert mean["precision_at_k"] == 0.25  # (1/2 + 0/2) / 2
    assert mean["recall_at_k"] == 0.5
    assert mean["mrr"] == pytest.approx((1 + 1 / 3) / 2)


def test_ingest_traces_operation_with_slo_policy():
    records = [
        {"span_id": f"s{i}", "name": "generate", "latency_ms": i, "success": True, "cost_usd": 0.01}
        for i in range(1, 21)
    ]
    slo = {"max_p95_latency_ms": 18, "max_average_cost_usd": 1, "min_success_rate": 1}
    envelope = valid(run_operation("ingest_traces", {"records": records, "slo": slo}))
    data = envelope["data"]
    assert data["overall"]["latency_p50_ms"] == 10
    assert data["overall"]["latency_p95_ms"] == 19
    assert data["slo"]["passed"] is False
    assert envelope["policy"]["decision"] == "deny"


TRACE = {"span_id": "a", "name": "n", "latency_ms": 1, "success": True}
REST = GOOD[1:]
NAN = float("nan")
DUPLICATE_RETRIEVAL = {"query_id": "q", "retrieved_ids": ["a", "a"], "relevant_ids": ["a"]}


@pytest.mark.parametrize(
    ("operation", "payload"),
    [
        ("evaluate", {"dataset": DATASET, "predictions": GOOD[:2]}),
        ("evaluate", {"dataset": DATASET, "predictions": [*GOOD, {"case_id": "x", "output": ""}]}),
        ("evaluate", {"dataset": DATASET, "predictions": GOOD, "unexpected": 1}),
        ("evaluate", {"dataset": DATASET, "predictions": [{**GOOD[0], "input_tokens": True}, *REST]}),
        ("evaluate", {"dataset": DATASET, "predictions": [{**GOOD[0], "latency_ms": NAN}, *REST]}),
        ("evaluate", {"dataset": DATASET, "predictions": [{**GOOD[0], "latency_ms": -1}, *REST]}),
        ("validate_citations", {"cited_ids": "d1", "evidence_ids": []}),
        ("score_retrieval", {"k": 1, "queries": [DUPLICATE_RETRIEVAL]}),
        ("ingest_traces", {"records": [TRACE, TRACE]}),
        ("ingest_traces", {"records": []}),
    ],
)
def test_invalid_input_returns_schema_valid_error_envelope(operation, payload):
    envelope = valid(run_operation(operation, payload))
    assert envelope["status"] == "error"
    assert envelope["data"] is None
    assert envelope["error"]["code"] == "INVALID_INPUT"
    assert envelope["error"]["retryable"] is False


@pytest.mark.parametrize("operation", ["rollback_model", "Bad Op!", ""])
def test_unknown_operation_returns_schema_valid_error(operation):
    envelope = valid(run_operation(operation, {}))
    assert envelope["status"] == "error"
    assert envelope["error"]["code"] == "UNKNOWN_OPERATION"


def test_run_and_correlation_ids_are_propagated():
    envelope = valid(
        run_operation("validate_citations", EMPTY_CITATIONS, run_id="run-1", correlation_id="c-1")
    )
    assert (envelope["run_id"], envelope["correlation_id"]) == ("run-1", "c-1")
    defaulted = valid(run_operation("validate_citations", EMPTY_CITATIONS, run_id="run-2"))
    assert defaulted["correlation_id"] == "run-2"


def test_oversized_correlation_id_is_rejected():
    envelope = valid(
        run_operation("validate_citations", EMPTY_CITATIONS, correlation_id="x" * 200)
    )
    assert envelope["error"]["code"] == "INVALID_INPUT"
