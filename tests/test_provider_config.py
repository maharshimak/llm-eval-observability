import pytest

from llm_eval.providers import OpenAICompatibleCandidate


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_candidate_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        OpenAICompatibleCandidate(
            base_url="https://example.test/v1",
            model="demo",
            timeout_seconds=timeout,
        )


@pytest.mark.parametrize("temperature", [float("inf"), float("-inf"), float("nan")])
def test_candidate_rejects_non_finite_temperature(temperature):
    with pytest.raises(ValueError):
        OpenAICompatibleCandidate(
            base_url="https://example.test/v1",
            model="demo",
            temperature=temperature,
        )
