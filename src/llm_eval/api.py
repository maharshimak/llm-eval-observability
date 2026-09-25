import os
import secrets
from dataclasses import asdict

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_eval.gates import regression_gate
from llm_eval.judges import OpenAICompatibleJudge
from llm_eval.models import EvalCase, ModelOutput
from llm_eval.providers import OpenAICompatibleCandidate
from llm_eval.runner import ExperimentRunner

app = FastAPI(
    title="LLM Eval & Observability",
    version="0.2.0",
    description=(
        "Versioned evaluation service for supplied outputs or configured live "
        "OpenAI-compatible candidates, with bearer-protected remote access."
    ),
)


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=300)
    prompt: str = Field(min_length=1, max_length=20_000)
    output: str = Field(max_length=100_000)
    citations: list[str] = Field(default_factory=list, max_length=200)
    expected_terms: list[str] = Field(default_factory=list, max_length=200)
    expected_citations: list[str] = Field(default_factory=list, max_length=200)
    forbidden_phrases: list[str] = Field(default_factory=list, max_length=200)
    latency_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class LiveEvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=300)
    prompt: str = Field(min_length=1, max_length=20_000)
    expected_terms: list[str] = Field(default_factory=list, max_length=200)
    expected_citations: list[str] = Field(default_factory=list, max_length=200)
    forbidden_phrases: list[str] = Field(default_factory=list, max_length=200)


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str = Field(min_length=1, max_length=300)
    items: list[EvalItem] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Evaluation item IDs must be unique.")
        return self


class LiveEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str = Field(min_length=1, max_length=300)
    items: list[LiveEvalItem] = Field(min_length=1, max_length=250)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Evaluation item IDs must be unique.")
        return self


class JudgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=20_000)
    output: str = Field(max_length=100_000)
    rubric: str = Field(min_length=1, max_length=20_000)
    threshold: float = Field(default=0.8, ge=0, le=1)
    citations: list[str] = Field(default_factory=list, max_length=200)
    expected_terms: list[str] = Field(default_factory=list, max_length=200)
    expected_citations: list[str] = Field(default_factory=list, max_length=200)


async def require_auth(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    token = os.environ.get("LLM_EVAL_API_TOKEN")
    if token:
        scheme, _, supplied = (authorization or "").partition(" ")
        if (
            scheme.lower() != "bearer"
            or not supplied
            or not secrets.compare_digest(supplied, token)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Valid bearer token required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return

    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Remote access requires LLM_EVAL_API_TOKEN.",
        )


def _case(
    item: EvalItem | LiveEvalItem,
) -> EvalCase:
    return EvalCase(
        id=item.id,
        prompt=item.prompt,
        expected_terms=set(item.expected_terms),
        expected_citations=set(item.expected_citations),
        forbidden_phrases=set(item.forbidden_phrases),
    )


def _live_candidate() -> OpenAICompatibleCandidate:
    base_url = os.environ.get("LLM_EVAL_BASE_URL", "").strip()
    model = os.environ.get("LLM_EVAL_MODEL", "").strip()
    if not base_url or not model:
        raise HTTPException(
            status_code=503,
            detail="LLM_EVAL_BASE_URL and LLM_EVAL_MODEL are required for live evaluation.",
        )
    try:
        timeout_seconds = float(os.environ.get("LLM_EVAL_TIMEOUT_SECONDS", "60"))
        temperature = float(os.environ.get("LLM_EVAL_TEMPERATURE", "0"))
    except ValueError as error:
        raise HTTPException(
            status_code=503,
            detail="Live evaluation timeout and temperature must be numeric.",
        ) from error
    return OpenAICompatibleCandidate(
        base_url=base_url,
        model=model,
        api_key=os.environ.get("LLM_EVAL_API_KEY", ""),
        timeout_seconds=timeout_seconds,
        temperature=temperature,
    )


protected = [Depends(require_auth)]


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "live_provider_configured": bool(
            os.environ.get("LLM_EVAL_BASE_URL") and os.environ.get("LLM_EVAL_MODEL")
        ),
    }


@app.post("/v1/evaluate", dependencies=protected)
def evaluate(request: EvaluateRequest) -> dict[str, object]:
    cases = [_case(item) for item in request.items]
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


@app.post("/v1/evaluate/live", dependencies=protected)
def evaluate_live(request: LiveEvaluateRequest) -> dict[str, object]:
    cases = [_case(item) for item in request.items]
    candidate = _live_candidate()
    try:
        summary = ExperimentRunner().run(request.experiment, cases, candidate)
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Live candidate execution failed: {type(error).__name__}",
        ) from error
    gate = regression_gate(summary)
    return {
        "summary": asdict(summary),
        "gate": asdict(gate),
        "provider": "openai-compatible",
        "model": candidate.model,
    }


@app.post("/v1/judge", dependencies=protected)
def judge(request: JudgeRequest) -> dict[str, object]:
    base_url = (
        os.environ.get("LLM_JUDGE_BASE_URL", "").strip()
        or os.environ.get("LLM_EVAL_BASE_URL", "").strip()
    )
    model = (
        os.environ.get("LLM_JUDGE_MODEL", "").strip()
        or os.environ.get("LLM_EVAL_MODEL", "").strip()
    )
    if not base_url or not model:
        raise HTTPException(
            status_code=503,
            detail="Configure LLM_JUDGE_BASE_URL/LLM_JUDGE_MODEL or the live eval provider.",
        )
    try:
        timeout_seconds = float(os.environ.get("LLM_JUDGE_TIMEOUT_SECONDS", "60"))
    except ValueError as error:
        raise HTTPException(status_code=503, detail="Judge timeout must be numeric.") from error

    evaluator = OpenAICompatibleJudge(
        base_url=base_url,
        model=model,
        rubric=request.rubric,
        api_key=(
            os.environ.get("LLM_JUDGE_API_KEY", "")
            or os.environ.get("LLM_EVAL_API_KEY", "")
        ),
        threshold=request.threshold,
        timeout_seconds=timeout_seconds,
    )
    case = EvalCase(
        id="judge",
        prompt=request.prompt,
        expected_terms=set(request.expected_terms),
        expected_citations=set(request.expected_citations),
        forbidden_phrases=set(),
    )
    output = ModelOutput(
        text=request.output,
        citations=request.citations,
        latency_ms=0,
        input_tokens=0,
        output_tokens=0,
    )
    try:
        result = evaluator.evaluate(case, output)
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Judge execution failed: {type(error).__name__}",
        ) from error
    return asdict(result)
