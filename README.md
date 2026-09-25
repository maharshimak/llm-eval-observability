# 🔭 LLM Eval & Observability

**A MAK'MA Studio Product · MAK'MA Labs**

[Live Product Demo](https://maharshimak.github.io/makma-ai-os/projects/llm-eval-observability/) · [MAK'MA Labs](https://maharshimak.github.io/makma-ai-os/projects/)

A compact evaluation and observability platform for **LLM/RAG regressions**.


## Product contract — engineering upgrade

**Problem and audience:** A release evaluation workstation for engineers comparing LLM/RAG candidate outputs with a baseline.

**Live tool:** https://maharshimak.github.io/makma-ai-os/projects/llm-eval-observability/

**Implemented browser workflow:** Batch case JSON, relevance/citation/forbidden checks, supplied latency/tokens/pricing, per-case failures, nearest-rank p95, cost/pass-rate/sample SLOs, Wilson interval, baseline deltas and regression budgets. Separately reported reliability does not affect case-derived release metrics.

**Backend and parity contract:** Python comparison gates citation regressions and rejects invalid regression budgets. Browser relevance uses phrase containment; the legacy Python metric uses token membership. The two modes are identified explicitly. Manually supplied candidates remain supported, and Python now also provides an OpenAI-compatible candidate adapter that measures wall-clock latency and consumes provider-reported token usage when available.

**Architecture:** `makma-ai-os/demo` is the shared web product source and Pages deployment. This repository owns its Python domain package. The central `tests/e2e` suite exercises all nine products; `tests/fixtures/python-parity.json` plus `scripts/generate_parity.py` guard shared mathematical contracts. Backend revisions used for regeneration are pinned in the central `backend-lock.json`.

**Safety and limitations:** Optional live inference is available only through an explicitly configured OpenAI-compatible endpoint. There is still no semantic model judge or proof of factual grounding; citation checks verify identifiers rather than entailment. Wilson intervals assume the supplied cases represent the population, and synthetic samples are not production benchmarks. Inputs are validated, rendered user values are escaped, and deterministic results are not presented as model inference.

**Verification:** Run `python -m ruff check .` and `python -m pytest -q`. `tests/test_engineering_upgrade.py` protects the new rejection/correctness paths. Central web checks: `npm ci`, `npm test`, `npm run build`, `npx playwright install --with-deps chromium`, `npm run test:e2e`. CI gates publishing on browser interactions and validates all public URLs after deployment.

**Implemented semantic judging:** `OpenAICompatibleJudge` provides strict JSON rubric scoring with fail-closed output validation, and the protected `/v1/judge` API can run it against a configured provider. Deterministic lexical/citation metrics remain available for reproducible regression testing.

**Highest-value next work:** Paired statistical comparisons, semantic/faithfulness evaluators and provider trace ingestion.

**Provenance:** Independent MAK’MA Studio engineering implementation; examples are synthetic and no employer code or data is included. Existing MIT license applies.


## Implemented

- typed experiment cases
- latency and cost tracking
- lexical answer relevance
- citation coverage
- forbidden-phrase detection
- token/cost estimation plus provider-reported usage capture when available
- OpenAI-compatible live candidate execution with measured wall-clock latency and provider/model provenance
- versioned evaluation datasets with deterministic SHA-256 fingerprints
- reproducible experiment manifests recording dataset, provider/model, prompt version and git revision
- experiment aggregation
- regression quality gates
- JSONL trace store
- FastAPI evaluation endpoint
- tests
- Docker

## Why this project matters

AI systems can regress without throwing exceptions. A deployment may still return HTTP 200 while becoming slower, more expensive, less grounded or less relevant.

This project treats model quality as an engineering signal.

```text
test cases
   ↓
candidate system
   ↓
traces
   ↓
metrics
   ↓
quality gate
   ↓
PASS / BLOCK RELEASE
```

## Metrics

- relevance score
- citation coverage
- forbidden-output rate
- latency p95
- average estimated cost
- pass rate

## Example

```python
from llm_eval.models import EvalCase, ModelOutput
from llm_eval.runner import ExperimentRunner
from llm_eval.gates import regression_gate

cases = [
    EvalCase(
        id="rag-1",
        prompt="What is RAG?",
        expected_terms={"retrieval", "generation"},
        expected_citations={"doc-1"},
    )
]


def candidate(case):
    return ModelOutput(
        text="Retrieval augmented generation combines retrieval and generation.",
        citations=["doc-1"],
        latency_ms=180,
        input_tokens=100,
        output_tokens=28,
    )


summary = ExperimentRunner().run("candidate-v1", cases, candidate)
decision = regression_gate(summary)
print(decision)
```

## Roadmap

- prompt/version registry
- RAG faithfulness evaluator
- pairwise model comparison
- OpenTelemetry exporter
- dashboard
- CI deployment gate

## Scope and limitations

Relevance is lexical overlap and citation coverage checks identifiers, not factuality or entailment. Manual candidates can still supply latency/token observations; the OpenAI-compatible candidate instead measures wall-clock latency and uses provider-reported token usage when present, explicitly labeling a whitespace-token estimate fallback otherwise. Cost is calculated only when caller-provided rates are configured; the API defaults to zero rates. JSONL traces, dataset loading and comparison are library utilities, not an integrated dashboard. There is no continuous telemetry collector, semantic judge or deployed release automation.

## Installation and development

Requires Python 3.12 or newer. Run from this project directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
python -m pip wheel --no-deps . -w dist
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.

## Library usage

```python
from llm_eval.models import EvalCase, ModelOutput
from llm_eval.runner import ExperimentRunner
from llm_eval.gates import regression_gate
cases = [EvalCase("rag-1", "What is RAG?", expected_terms={"retrieval", "generation"})]
def candidate(case):
    return ModelOutput("retrieval augmented generation", [], 100, 20, 5)
summary = ExperimentRunner().run("offline-demo", cases, candidate)
print(regression_gate(summary))
```

## Configuration

Offline evaluation needs no credentials. Live API evaluation reads `LLM_EVAL_BASE_URL` and `LLM_EVAL_MODEL`; optional settings are `LLM_EVAL_API_KEY`, `LLM_EVAL_TIMEOUT_SECONDS`, `LLM_EVAL_TEMPERATURE`, and `LLM_EVAL_API_TOKEN`.

## Service and API schema

```bash
python -m uvicorn llm_eval.api:app --host 127.0.0.1 --port 8000
```

Interactive endpoint schemas are at `http://127.0.0.1:8000/docs`; machine-readable schemas are at `/openapi.json`. The API is local-only by default. Set `LLM_EVAL_API_TOKEN` for bearer-authenticated remote access. `/v1/evaluate/live` executes the configured OpenAI-compatible candidate using `LLM_EVAL_BASE_URL`, `LLM_EVAL_MODEL`, optional `LLM_EVAL_API_KEY`, `LLM_EVAL_TIMEOUT_SECONDS`, and `LLM_EVAL_TEMPERATURE`.

## Container

```bash
docker build -t llm-eval-observability .
docker run --rm -p 127.0.0.1:8000:8000 llm-eval-observability
```

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/llm_eval/` | Implementation |
| `tests/` | Offline unit and regression tests |
| `docs/DESIGN.md` | Architecture and trust boundaries |
| `.github/workflows/ci.yml` | Install, lint, tests, wheel and container build |
| `pyproject.toml` | Dependencies and package configuration |

## Next engineering work

Async/batched provider adapters; citation faithfulness; paired dataset comparison constraints; versioned datasets/pricing; OpenTelemetry export and CI release integration. These are planned work, not current capabilities.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). CI runs on every push and pull request through `.github/workflows/ci.yml`.

## License and provenance

[MIT](LICENSE), copyright 2026 Maharshi Patel. This public portfolio implementation is independent of employer systems and contains no confidential employer code or data. Examples and test fixtures are synthetic.
