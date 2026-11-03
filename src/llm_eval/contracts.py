"""MAK'MA shared execution contract ``makma.v1`` for this evaluation system.

``run_operation`` runs one domain operation in-process and returns the
envelope as a plain dict that validates against
``contracts/makma.v1.schema.json``. Every operation here is deterministic and
offline (lexical scoring, identifier checks, arithmetic and seeded
resampling); none calls a model, so ``execution_mode`` is always
``deterministic_offline``. The optional model judge (``/v1/judge``) is not
exposed through this contract.

The ``data`` of every successful operation contains no timestamps or random
identifiers, so re-running an operation on the same input produces
byte-identical canonical JSON (see ``llm_eval.canonical``); its SHA-256 is
recorded as an ``artifact`` provenance entry.
"""

from __future__ import annotations

import json
import secrets
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from math import fsum
from time import perf_counter
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
    model_validator,
)

from llm_eval.canonical import sha256_hex
from llm_eval.citations import retrieval_metrics, validate_citations
from llm_eval.models import CaseMetrics, EvalCase, ModelOutput
from llm_eval.release_gate import GatePolicy, MetricBudget, evaluate_release_gate
from llm_eval.runner import ExperimentRunner
from llm_eval.slo import SLOPolicy, TraceSample, evaluate_slo
from llm_eval.statistics import nearest_rank_percentile
from llm_eval.taxonomy import FAILURE_CATEGORIES, classify_case, count_failures
from llm_eval.traces import TraceRecord, ingest_traces
from llm_eval.versioning import build_versioned_dataset

CONTRACT_VERSION = "makma.v1"
SYSTEM = "llm-eval-observability"
EXECUTION_MODE = "deterministic_offline"
_OPERATION_PATTERN_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")

Number = StrictInt | StrictFloat


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


# --------------------------------------------------------------------------- envelope


class ProvenanceRef(_Strict):
    kind: Literal["document", "chunk", "artifact", "dataset", "model", "query", "record"]
    id: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    span: tuple[int, int] | None = None
    uri: str | None = None


class ContractError(_Strict):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    message: str = Field(max_length=2000)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class PolicyDecision(_Strict):
    decision: Literal["allow", "deny", "requires_approval"]
    reasons: list[str] = Field(default_factory=list)


class Envelope(_Strict):
    contract_version: Literal["makma.v1"] = CONTRACT_VERSION
    system: Literal["llm-eval-observability"] = SYSTEM
    operation: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    run_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    status: Literal["ok", "error", "blocked", "pending_approval"]
    execution_mode: Literal["deterministic_offline", "local_model", "remote_model"]
    data: Any = None
    error: ContractError | None = None
    provenance: list[ProvenanceRef] = Field(default_factory=list)
    policy: PolicyDecision | None = None
    started_at: str
    duration_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def _status_rules(self) -> Envelope:
        if (self.status == "ok") != (self.error is None):
            raise ValueError("status 'ok' if and only if error is null")
        if self.status in {"error", "blocked"} and self.data is not None:
            raise ValueError("data must be null when status is error or blocked")
        if self.status == "pending_approval" and (
            self.policy is None or self.policy.decision != "requires_approval"
        ):
            raise ValueError("pending_approval requires policy.decision 'requires_approval'")
        return self


# --------------------------------------------------------------------------- inputs


class Thresholds(_Strict):
    min_relevance: Number = Field(default=0.7, ge=0, le=1)
    min_citation_coverage: Number = Field(default=0.8, ge=0, le=1)
    max_latency_ms: Number = Field(default=5000.0, ge=0)


class Pricing(_Strict):
    input_per_million: Number = Field(default=0.0, ge=0)
    output_per_million: Number = Field(default=0.0, ge=0)


class DatasetCase(_Strict):
    id: str = Field(min_length=1, max_length=300)
    prompt: str = Field(min_length=1, max_length=20_000)
    expected_terms: list[str] = Field(default_factory=list, max_length=200)
    expected_citations: list[str] = Field(default_factory=list, max_length=200)
    forbidden_phrases: list[str] = Field(default_factory=list, max_length=200)


class Dataset(_Strict):
    id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=100)
    cases: list[DatasetCase] = Field(min_length=1, max_length=1000)


class Prediction(_Strict):
    case_id: str = Field(min_length=1, max_length=300)
    output: str = Field(max_length=100_000)
    citations: list[str] = Field(default_factory=list, max_length=200)
    latency_ms: Number = Field(default=0.0, ge=0)
    input_tokens: StrictInt = Field(default=0, ge=0)
    output_tokens: StrictInt = Field(default=0, ge=0)


class EvaluateInput(_Strict):
    dataset: Dataset
    predictions: list[Prediction] = Field(min_length=1, max_length=1000)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    pricing: Pricing = Field(default_factory=Pricing)

    @model_validator(mode="after")
    def _aligned(self) -> EvaluateInput:
        case_ids = [case.id for case in self.dataset.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("dataset case ids must be unique")
        predicted = [prediction.case_id for prediction in self.predictions]
        if len(set(predicted)) != len(predicted):
            raise ValueError("prediction case_ids must be unique")
        missing = sorted(set(case_ids) - set(predicted))
        extra = sorted(set(predicted) - set(case_ids))
        if missing or extra:
            raise ValueError(
                f"predictions must cover exactly the dataset cases "
                f"(missing={missing[:10]}, unexpected={extra[:10]})"
            )
        return self


class ReportDataset(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    id: str | None = None
    version: str | None = None


class ReportCase(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False, frozen=True)

    case_id: str = Field(min_length=1, max_length=300)
    relevance: Number = Field(ge=0, le=1)
    citation_coverage: Number = Field(ge=0, le=1)
    forbidden_hit: StrictBool
    latency_ms: Number = Field(ge=0)
    estimated_cost_usd: Number = Field(ge=0)
    passed: StrictBool


class StoredReport(BaseModel):
    """The ``data`` returned by ``evaluate`` (extra fields are ignored)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    dataset: ReportDataset
    thresholds: Thresholds
    cases: list[ReportCase] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def _consistent(self) -> StoredReport:
        # A stored report is untrusted input: its pass flags must follow from its
        # own metrics and thresholds, otherwise the comparison would be meaningless.
        for case in self.cases:
            categories = classify_case(
                _case_metrics(case),
                min_relevance=self.thresholds.min_relevance,
                min_citation_coverage=self.thresholds.min_citation_coverage,
                max_latency_ms=self.thresholds.max_latency_ms,
            )
            if case.passed != (not categories):
                raise ValueError(f"case {case.case_id!r}: passed flag contradicts its metrics")
        return self


class BudgetInput(_Strict):
    max_regression: Number = Field(ge=0)
    critical: StrictBool = True


class PolicyInput(_Strict):
    budgets: dict[str, BudgetInput] | None = Field(default=None, max_length=10)
    require_significance: StrictBool = False
    confidence: StrictFloat = Field(default=0.95, ge=0.5, lt=1)
    resamples: StrictInt = Field(default=2000, ge=100, le=5000)
    seed: StrictInt = 42


class CompareInput(_Strict):
    baseline: StoredReport
    candidate: StoredReport
    policy: PolicyInput = Field(default_factory=PolicyInput)


class ValidateCitationsInput(_Strict):
    cited_ids: list[str] = Field(max_length=1000)
    evidence_ids: list[str] = Field(max_length=10_000)
    require_citation: StrictBool = False


class RetrievalQuery(_Strict):
    query_id: str = Field(min_length=1, max_length=300)
    retrieved_ids: list[str] = Field(max_length=1000)
    relevant_ids: list[str] = Field(min_length=1, max_length=1000)


class ScoreRetrievalInput(_Strict):
    k: StrictInt = Field(ge=1, le=1000)
    queries: list[RetrievalQuery] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def _unique(self) -> ScoreRetrievalInput:
        ids = [query.query_id for query in self.queries]
        if len(set(ids)) != len(ids):
            raise ValueError("query_id values must be unique")
        return self


class TraceInput(_Strict):
    span_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    latency_ms: Number = Field(ge=0)
    success: StrictBool
    cost_usd: Number = Field(default=0.0, ge=0)
    input_tokens: StrictInt = Field(default=0, ge=0)
    output_tokens: StrictInt = Field(default=0, ge=0)
    model: str | None = Field(default=None, min_length=1, max_length=200)


class SLOInput(_Strict):
    max_p95_latency_ms: Number = Field(gt=0)
    max_average_cost_usd: Number = Field(ge=0)
    min_success_rate: Number = Field(ge=0, le=1)
    min_samples: StrictInt = Field(default=20, ge=1)


class IngestTracesInput(_Strict):
    records: list[TraceInput] = Field(min_length=1, max_length=10_000)
    group_by: Literal["model", "name"] | None = None
    slo: SLOInput | None = None


# --------------------------------------------------------------------------- operations


class _OperationError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


_Result = tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]


def _ref(kind: str, ref_id: str, sha: str | None) -> dict[str, Any]:
    return {"kind": kind, "id": ref_id, "sha256": sha, "span": None, "uri": None}


def _case_metrics(case: ReportCase) -> CaseMetrics:
    return CaseMetrics(
        case_id=case.case_id,
        relevance=case.relevance,
        citation_coverage=case.citation_coverage,
        forbidden_hit=case.forbidden_hit,
        latency_ms=case.latency_ms,
        estimated_cost_usd=case.estimated_cost_usd,
        passed=case.passed,
    )


def _evaluate(payload: EvaluateInput) -> _Result:
    cases = [
        EvalCase(
            id=case.id,
            prompt=case.prompt,
            expected_terms=set(case.expected_terms),
            expected_citations=set(case.expected_citations),
            forbidden_phrases=set(case.forbidden_phrases),
        )
        for case in payload.dataset.cases
    ]
    dataset = build_versioned_dataset(payload.dataset.id, payload.dataset.version, cases)
    lookup = {prediction.case_id: prediction for prediction in payload.predictions}

    def candidate(case: EvalCase) -> ModelOutput:
        prediction = lookup[case.id]
        return ModelOutput(
            text=prediction.output,
            citations=list(prediction.citations),
            latency_ms=float(prediction.latency_ms),
            input_tokens=prediction.input_tokens,
            output_tokens=prediction.output_tokens,
        )

    thresholds = payload.thresholds
    runner = ExperimentRunner(
        min_relevance=thresholds.min_relevance,
        min_citation_coverage=thresholds.min_citation_coverage,
        max_latency_ms=thresholds.max_latency_ms,
        input_per_million=payload.pricing.input_per_million,
        output_per_million=payload.pricing.output_per_million,
    )
    summary = runner.run(f"{dataset.id}@{dataset.version}", list(dataset.cases), candidate)
    classified = [
        classify_case(
            metrics,
            min_relevance=thresholds.min_relevance,
            min_citation_coverage=thresholds.min_citation_coverage,
            max_latency_ms=thresholds.max_latency_ms,
        )
        for metrics in summary.cases
    ]
    costs = [metrics.estimated_cost_usd for metrics in summary.cases]
    latencies = [metrics.latency_ms for metrics in summary.cases]
    predictions_sha = sha256_hex(
        sorted(
            (prediction.model_dump(mode="json") for prediction in payload.predictions),
            key=lambda item: item["case_id"],
        )
    )
    data = {
        "dataset": {
            "id": dataset.id,
            "version": dataset.version,
            "fingerprint": dataset.fingerprint,
            "case_count": len(dataset.cases),
        },
        "predictions_sha256": predictions_sha,
        "thresholds": thresholds.model_dump(mode="json"),
        "pricing": payload.pricing.model_dump(mode="json"),
        "scoring": {
            "method": "deterministic_lexical",
            "model_judge": "not_used",
            "failure_categories": list(FAILURE_CATEGORIES),
        },
        "summary": {
            "pass_rate": summary.pass_rate,
            "mean_relevance": summary.mean_relevance,
            "mean_citation_coverage": summary.mean_citation_coverage,
            "latency_p50_ms": nearest_rank_percentile(latencies, 50),
            "latency_p95_ms": nearest_rank_percentile(latencies, 95),
            "mean_cost_usd": fsum(costs) / len(costs),
            "total_cost_usd": fsum(costs),
        },
        "failure_counts": count_failures(classified),
        "cases": [
            {**asdict(metrics), "failure_categories": list(categories)}
            for metrics, categories in zip(summary.cases, classified, strict=True)
        ],
    }
    provenance = [
        _ref("dataset", f"{dataset.id}@{dataset.version}", dataset.fingerprint),
        _ref("record", "predictions", predictions_sha),
    ]
    return data, provenance, None


def _compare(payload: CompareInput) -> _Result:
    baseline, candidate = payload.baseline, payload.candidate
    if baseline.dataset.fingerprint != candidate.dataset.fingerprint:
        raise _OperationError(
            "DATASET_MISMATCH",
            "baseline and candidate were evaluated on different datasets",
            {
                "baseline_fingerprint": baseline.dataset.fingerprint,
                "candidate_fingerprint": candidate.dataset.fingerprint,
            },
        )
    if baseline.thresholds != candidate.thresholds:
        raise _OperationError(
            "THRESHOLD_MISMATCH",
            "baseline and candidate were scored with different thresholds",
            {
                "baseline": baseline.thresholds.model_dump(mode="json"),
                "candidate": candidate.thresholds.model_dump(mode="json"),
            },
        )
    policy_input = payload.policy
    budgets = (
        None
        if policy_input.budgets is None
        else {
            metric: MetricBudget(float(budget.max_regression), critical=budget.critical)
            for metric, budget in policy_input.budgets.items()
        }
    )
    policy_kwargs: dict[str, Any] = {
        "require_significance": policy_input.require_significance,
        "confidence": policy_input.confidence,
        "resamples": policy_input.resamples,
        "seed": policy_input.seed,
    }
    if budgets is not None:
        policy_kwargs["budgets"] = budgets
    policy = GatePolicy(**policy_kwargs)
    decision = evaluate_release_gate(
        [_case_metrics(case) for case in baseline.cases],
        [_case_metrics(case) for case in candidate.cases],
        policy,
    )
    baseline_sha = sha256_hex(baseline.model_dump(mode="json"))
    candidate_sha = sha256_hex(candidate.model_dump(mode="json"))
    data = {
        "decision": decision.decision,
        "dataset_fingerprint": baseline.dataset.fingerprint,
        "baseline_sha256": baseline_sha,
        "candidate_sha256": candidate_sha,
        "policy": {
            "budgets": {
                metric: {"max_regression": float(budget.max_regression), "critical": budget.critical}
                for metric, budget in sorted(policy.budgets.items())
            },
            "require_significance": policy.require_significance,
            "confidence": policy.confidence,
            "resamples": policy.resamples,
            "seed": policy.seed,
            "statistics": "paired bootstrap percentile interval; exact McNemar for binary metrics",
        },
        "case_count": decision.case_count,
        "blocking_reasons": list(decision.blocking_reasons),
        "warnings": list(decision.warnings),
        "metrics": [asdict(metric) for metric in decision.metrics],
        "transitions": asdict(decision.transitions),
    }
    provenance = [
        _ref("dataset", "evaluated-dataset", baseline.dataset.fingerprint),
        _ref("artifact", "baseline-report", baseline_sha),
        _ref("artifact", "candidate-report", candidate_sha),
    ]
    reasons = list(decision.blocking_reasons) or ["no critical metric exceeded its budget"]
    return data, provenance, {"decision": decision.decision, "reasons": reasons}


def _validate_citations(payload: ValidateCitationsInput) -> _Result:
    result = validate_citations(
        payload.cited_ids, payload.evidence_ids, require_citation=payload.require_citation
    )
    data = {**asdict(result), "check": "identifier_subset_only_no_entailment"}
    return data, [_ref("record", "citation-input", sha256_hex(payload.model_dump(mode="json")))], None


def _score_retrieval(payload: ScoreRetrievalInput) -> _Result:
    per_query = []
    for query in payload.queries:
        metrics = retrieval_metrics(query.retrieved_ids, query.relevant_ids, k=payload.k)
        per_query.append({"query_id": query.query_id, **asdict(metrics)})
    count = len(per_query)

    def macro(name: str) -> float:
        return fsum(item[name] for item in per_query) / count

    data = {
        "k": payload.k,
        "query_count": count,
        "relevance": "binary",
        "mean": {
            "precision_at_k": macro("precision_at_k"),
            "recall_at_k": macro("recall_at_k"),
            "hit_rate_at_k": macro("hit_rate_at_k"),
            "mrr": macro("reciprocal_rank"),
            "ndcg_at_k": macro("ndcg_at_k"),
        },
        "queries": per_query,
    }
    return data, [_ref("record", "retrieval-input", sha256_hex(payload.model_dump(mode="json")))], None


def _ingest_traces(payload: IngestTracesInput) -> _Result:
    records = [TraceRecord(**record.model_dump()) for record in payload.records]
    overall, groups = ingest_traces(records, group_by=payload.group_by)
    data: dict[str, Any] = {
        "percentile_method": "nearest_rank",
        "group_by": payload.group_by,
        "overall": asdict(overall),
        "groups": {key: asdict(value) for key, value in groups.items()},
        "slo": None,
    }
    policy = None
    if payload.slo is not None:
        slo = evaluate_slo(
            [
                TraceSample(
                    latency_ms=record.latency_ms,
                    cost_usd=record.cost_usd,
                    success=record.success,
                )
                for record in records
            ],
            SLOPolicy(
                max_p95_latency_ms=float(payload.slo.max_p95_latency_ms),
                max_average_cost_usd=float(payload.slo.max_average_cost_usd),
                min_success_rate=float(payload.slo.min_success_rate),
                min_samples=payload.slo.min_samples,
            ),
        )
        data["slo"] = {**asdict(slo), "reasons": list(slo.reasons)}
        policy = {
            "decision": "allow" if slo.passed else "deny",
            "reasons": list(slo.reasons) or ["all SLO objectives met"],
        }
    records_sha = sha256_hex(
        sorted((record.model_dump(mode="json") for record in payload.records),
               key=lambda item: item["span_id"])
    )
    return data, [_ref("record", "trace-records", records_sha)], policy


_Operation = tuple[type[BaseModel], Callable[[Any], _Result], str]

OPERATIONS: dict[str, _Operation] = {
    "evaluate": (
        EvaluateInput,
        _evaluate,
        ("Score predictions against a versioned dataset (lexical relevance, expected "
        "citations, forbidden phrases, latency, cost) with a failure taxonomy."),
    ),
    "compare": (
        CompareInput,
        _compare,
        ("Gate a candidate evaluate report against a baseline on the same dataset with "
        "per-metric regression budgets; returns policy allow/deny."),
    ),
    "validate_citations": (
        ValidateCitationsInput,
        _validate_citations,
        "Check that cited ids are a subset of the provided evidence ids.",
    ),
    "score_retrieval": (
        ScoreRetrievalInput,
        _score_retrieval,
        "Binary-relevance precision@k, recall@k, hit rate, MRR and nDCG@k per query.",
    ),
    "ingest_traces": (
        IngestTracesInput,
        _ingest_traces,
        ("Validate trace records and aggregate latency (nearest-rank p50/p95), cost, "
        "tokens and error rate, optionally against an SLO."),
    ),
}


def _version() -> str:
    try:
        return package_version("llm-eval-observability")
    except PackageNotFoundError:  # pragma: no cover - source checkout without install
        return "0.0.0"


def capabilities() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "system": SYSTEM,
        "version": _version(),
        "operations": [
            {
                "name": name,
                "description": description,
                "side_effects": False,
                "requires_approval": False,
                "execution_mode": EXECUTION_MODE,
                "input_schema": model.model_json_schema(),
                "timeout_s": 30,
            }
            for name, (model, _, description) in OPERATIONS.items()
        ],
    }


def _safe_operation_name(operation: object) -> str:
    if (
        isinstance(operation, str)
        and 1 <= len(operation) <= 64
        and operation[0].isascii()
        and operation[0].islower()
        and set(operation) <= _OPERATION_PATTERN_CHARS
    ):
        return operation
    return "invalid_operation"


def _identifier(value: object) -> str | None:
    if isinstance(value, str) and 1 <= len(value) <= 128:
        return value
    return None


def run_operation(
    operation: str,
    payload: dict[str, Any],
    *,
    run_id: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    started = perf_counter()
    trace_id = uuid.uuid4().hex
    span_id = secrets.token_hex(8)
    resolved_run_id = _identifier(run_id) or uuid.uuid4().hex
    resolved_correlation = _identifier(correlation_id) or resolved_run_id

    status = "ok"
    data: Any = None
    error: dict[str, Any] | None = None
    provenance: list[dict[str, Any]] = []
    policy: dict[str, Any] | None = None

    def fail(code: str, message: str, details: dict[str, Any] | None = None) -> None:
        nonlocal status, error, data, provenance, policy
        status, data, provenance, policy = "error", None, [], None
        error = {"code": code, "message": message[:2000], "retryable": False,
                 "details": details or {}}

    spec = OPERATIONS.get(operation) if isinstance(operation, str) else None
    if (run_id is not None and _identifier(run_id) is None) or (
        correlation_id is not None and _identifier(correlation_id) is None
    ):
        fail("INVALID_INPUT", "run_id and correlation_id must be strings of 1-128 characters")
    elif spec is None:
        fail(
            "UNKNOWN_OPERATION",
            "unknown operation",
            {"operation": str(operation)[:100], "supported": sorted(OPERATIONS)},
        )
    elif not isinstance(payload, dict):
        fail("INVALID_INPUT", "payload must be a JSON object")
    else:
        model, handler, _ = spec
        try:
            parsed = model.model_validate(payload)
            data, provenance, policy = handler(parsed)
            provenance = [*provenance, _ref("artifact", f"{operation}-report", sha256_hex(data))]
        except ValidationError as exc:
            errors = json.loads(exc.json(include_url=False, include_input=False))
            fail("INVALID_INPUT", "payload failed validation", {"errors": errors[:50]})
        except _OperationError as exc:
            fail(exc.code, exc.message, exc.details)
        except (TypeError, ValueError) as exc:
            fail("INVALID_INPUT", str(exc))
        except Exception as exc:  # noqa: BLE001 - envelope must always be returned
            fail("INTERNAL_ERROR", f"operation failed: {type(exc).__name__}")

    envelope = Envelope(
        operation=_safe_operation_name(operation),
        run_id=resolved_run_id,
        correlation_id=resolved_correlation,
        trace_id=trace_id,
        span_id=span_id,
        status=status,
        execution_mode=EXECUTION_MODE,
        data=data,
        error=error,
        provenance=provenance,
        policy=policy,
        started_at=started_at,
        duration_ms=max(0.0, (perf_counter() - started) * 1000),
    )
    return envelope.model_dump(mode="json")
