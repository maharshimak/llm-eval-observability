import pytest

from llm_eval.judges import OpenAICompatibleJudge


def test_judge_parses_structured_score_and_threshold():
    judge = OpenAICompatibleJudge(
        base_url="http://localhost",
        model="judge",
        rubric="Answer must be grounded.",
        threshold=0.8,
    )
    result = judge.parse('{"score": 0.85, "reason": "Grounded in the supplied evidence."}')
    assert result.passed
    assert result.score == 0.85


@pytest.mark.parametrize(
    "raw",
    [
        '{"score": 2, "reason": "bad"}',
        '{"score": "0.8", "reason": "bad"}',
        '{"score": 0.8, "reason": ""}',
        "not json",
    ],
)
def test_judge_rejects_invalid_provider_output(raw):
    judge = OpenAICompatibleJudge(
        base_url="http://localhost",
        model="judge",
        rubric="Correctness",
    )
    with pytest.raises(ValueError):
        judge.parse(raw)
