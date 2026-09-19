# Design and operating boundaries

Offline experiment evaluation for lexical relevance, expected citations, forbidden phrases, reported latency and estimated cost.

## Scope

Relevance is lexical overlap and citation coverage checks identifiers, not factuality or entailment. Candidate outputs supply latency and token counts; these are validated but not independently measured. Cost is calculated only when caller-provided rates are configured; the API defaults to zero rates. JSONL traces, dataset loading and comparison are library utilities, not an integrated dashboard. No provider calls, live telemetry collector or deployed release automation is included.

## Interfaces

Implementation lives in `src/llm_eval/`. Public examples in the README use its Python API. FastAPI exposes the same local capabilities; `/openapi.json` is the endpoint schema.

## Validation

Tests include synthetic regression fixtures. Package and container checks verify installation separately from source-tree imports. Tests do not certify general model quality, clinical correctness or multi-tenant isolation.

## Planned evolution

Measured provider adapters; citation faithfulness; paired dataset comparison constraints; versioned pricing input; OpenTelemetry export.
