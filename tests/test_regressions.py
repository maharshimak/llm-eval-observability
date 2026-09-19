import pytest
from fastapi.testclient import TestClient

from llm_eval.api import app
from llm_eval.models import EvalCase, ModelOutput
from llm_eval.runner import ExperimentRunner


@pytest.mark.parametrize("cases", [[], [EvalCase("a", "q"), EvalCase("a", "q")]])
def test_empty_and_duplicate_experiments_rejected(cases):
    with pytest.raises(ValueError):
        ExperimentRunner().run("demo", cases, lambda c: ModelOutput("", [], 0, 0, 0))


@pytest.mark.parametrize("latency,tokens", [(float("nan"), 0), (-1, 0), (1, -2)])
def test_invalid_measurements_rejected(latency, tokens):
    with pytest.raises(ValueError):
        ExperimentRunner().run(
            "demo", [EvalCase("a", "q")], lambda c: ModelOutput("", [], latency, tokens, 0)
        )


def test_api_rejects_duplicate_ids():
    item = {"id": "same", "prompt": "q", "output": "a"}
    assert (
        TestClient(app)
        .post("/v1/evaluate", json={"experiment": "demo", "items": [item, item]})
        .status_code
        == 422
    )
