"""Deterministic LLM evaluation baseline (no external model calls)."""

from __future__ import annotations

import json

from llm_eval.models import EvalCase, ModelOutput
from llm_eval.runner import ExperimentRunner

CASES = [
    EvalCase("rag", "What is retrieval augmented generation?", {"retrieval", "generation"}, {"rag-guide"}),
    EvalCase("ops", "What makes canary deployment safe?", {"canary", "rollback"}, {"ops-guide"}),
    EvalCase("privacy", "How are sensitive columns protected?", {"authorization", "columns"}, {"security-guide"}),
]

OUTPUTS = {
    "rag": ("Retrieval augmented generation uses retrieval and generation.", ["rag-guide"], 24, 12),
    "ops": ("A canary deployment uses gates and rollback.", ["ops-guide"], 31, 11),
    "privacy": ("Authorization restricts access to sensitive columns.", ["security-guide"], 22, 10),
}


def benchmark() -> dict[str, object]:
    def candidate(case: EvalCase) -> ModelOutput:
        text, citations, input_tokens, output_tokens = OUTPUTS[case.id]
        return ModelOutput(
            text=text,
            citations=citations,
            latency_ms=120.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider="offline-fixture",
            model="deterministic-baseline",
        )

    # Synthetic pricing inputs test cost arithmetic; they are not live billing.
    summary = ExperimentRunner(
        input_per_million=1.0, output_per_million=2.0
    ).run("offline-baseline", CASES, candidate)
    return {
        "dataset": "three synthetic exact-match cases; no API or model inference",
        "cases": len(summary.cases),
        "pass_rate": summary.pass_rate,
        "mean_relevance": summary.mean_relevance,
        "mean_citation_coverage": summary.mean_citation_coverage,
        "latency_p95_ms": summary.latency_p95_ms,
        "mean_cost_usd": summary.mean_cost_usd,
    }


def main() -> None:
    report = benchmark()
    print(json.dumps(report, sort_keys=True, indent=2))
    if report["pass_rate"] != 1.0 or report["mean_citation_coverage"] != 1.0:
        raise SystemExit("Synthetic offline evaluation regression gate failed")


if __name__ == "__main__":
    main()
