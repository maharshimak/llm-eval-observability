from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

from llm_eval.gates import regression_gate
from llm_eval.models import EvalCase, ModelOutput
from llm_eval.runner import ExperimentRunner

app = FastAPI(title="LLM Eval & Observability", version="0.1.0")


class EvalItem(BaseModel):
    id: str
    prompt: str
    output: str
    citations: list[str] = Field(default_factory=list)
    expected_terms: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    latency_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class EvaluateRequest(BaseModel):
    experiment: str
    items: list[EvalItem] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Evaluation item IDs must be unique.")
        return self


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/evaluate")
def evaluate(request: EvaluateRequest) -> dict[str, object]:
    cases = [
        EvalCase(
            id=item.id,
            prompt=item.prompt,
            expected_terms=set(item.expected_terms),
            expected_citations=set(item.expected_citations),
            forbidden_phrases=set(item.forbidden_phrases),
        )
        for item in request.items
    ]
    lookup = {item.id: item for item in request.items}

    def candidate(case: EvalCase) -> ModelOutput:
        item = lookup[case.id]
        return ModelOutput(
            text=item.output,
            citations=item.citations,
            latency_ms=item.latency_ms,
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
        )

    summary = ExperimentRunner().run(request.experiment, cases, candidate)
    gate = regression_gate(summary)
    return {"summary": asdict(summary), "gate": asdict(gate)}
