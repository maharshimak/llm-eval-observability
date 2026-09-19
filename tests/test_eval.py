from llm_eval.gates import regression_gate
from llm_eval.models import EvalCase, ModelOutput
from llm_eval.runner import ExperimentRunner


def test_experiment_and_gate_pass() -> None:
    cases = [
        EvalCase(
            id="rag",
            prompt="What is RAG?",
            expected_terms={"retrieval", "generation"},
            expected_citations={"doc-1"},
        )
    ]

    def candidate(_: EvalCase) -> ModelOutput:
        return ModelOutput(
            text="Retrieval augmented generation combines retrieval and generation.",
            citations=["doc-1"],
            latency_ms=120,
            input_tokens=100,
            output_tokens=30,
        )

    summary = ExperimentRunner().run("candidate-v1", cases, candidate)

    assert summary.pass_rate == 1.0
    assert regression_gate(summary).passed


def test_forbidden_phrase_fails_case() -> None:
    case = EvalCase(
        id="safe",
        prompt="answer",
        forbidden_phrases={"internal secret"},
    )

    def candidate(_: EvalCase) -> ModelOutput:
        return ModelOutput(
            text="internal secret",
            citations=[],
            latency_ms=10,
            input_tokens=1,
            output_tokens=2,
        )

    summary = ExperimentRunner().run("unsafe", [case], candidate)
    assert summary.pass_rate == 0.0
