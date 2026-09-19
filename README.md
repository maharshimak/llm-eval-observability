# 🔭 LLM Eval & Observability

A compact evaluation and observability platform for **LLM/RAG regressions**.

## Implemented

- typed experiment cases
- latency and cost tracking
- lexical answer relevance
- citation coverage
- forbidden-phrase detection
- token/cost estimation
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

- provider adapters
- prompt/version registry
- RAG faithfulness evaluator
- pairwise model comparison
- dataset versioning
- OpenTelemetry exporter
- dashboard
- CI deployment gate

## Scope and limitations

Relevance is lexical overlap and citation coverage checks identifiers, not factuality or entailment. Candidate outputs supply latency and token counts; these are validated but not independently measured. Cost is calculated only when caller-provided rates are configured; the API defaults to zero rates. JSONL traces, dataset loading and comparison are library utilities, not an integrated dashboard. No provider calls, live telemetry collector or deployed release automation is included.

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

Configuration is supplied through Python function/constructor arguments. No credentials or environment file are needed for the offline example.

## Service and API schema

```bash
python -m uvicorn llm_eval.api:app --host 127.0.0.1 --port 8000
```

Interactive endpoint schemas are at `http://127.0.0.1:8000/docs`; machine-readable schemas are at `/openapi.json`. These APIs have no built-in authentication. Use trusted local data and local access.

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

Measured provider adapters; citation faithfulness; paired dataset comparison constraints; versioned pricing input; OpenTelemetry export. These are planned work, not current capabilities.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). The standalone CI workflow runs after migration; while nested in the profile repository, the parent CI validates this project.

## License and provenance

[MIT](LICENSE), copyright 2026 Maharshi Patel. This public portfolio implementation is independent of employer systems and contains no confidential employer code or data. Examples and test fixtures are synthetic.
