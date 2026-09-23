import json

from llm_eval.models import EvalCase
from llm_eval.providers import OpenAICompatibleCandidate


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [
                    {"message": {"content": "Grounded answer [doc-1] and [doc-2]."}}
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 9},
            }
        ).encode()


def test_openai_candidate_executes_provider_and_records_usage(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["authorization"] = req.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr("llm_eval.providers.request.urlopen", fake_urlopen)

    candidate = OpenAICompatibleCandidate(
        base_url="http://localhost:1234/v1",
        model="test-model",
        api_key="secret",
        timeout_seconds=12,
    )
    output = candidate(EvalCase(id="case-1", prompt="Answer with citations."))

    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["authorization"] == "Bearer secret"
    assert output.citations == ["doc-1", "doc-2"]
    assert output.input_tokens == 11
    assert output.output_tokens == 9
    assert output.usage_source == "reported"
    assert output.provider == "openai-compatible"
    assert output.model == "test-model"
