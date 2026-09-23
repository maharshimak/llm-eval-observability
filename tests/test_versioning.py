from llm_eval.models import EvalCase, ModelOutput
from llm_eval.runner import ExperimentRunner
from llm_eval.versioning import build_versioned_dataset, run_versioned_experiment


def test_versioned_dataset_is_deterministically_fingerprinted() -> None:
    cases = [EvalCase("a", "hello", expected_terms={"hello"})]
    first = build_versioned_dataset("demo", "1", cases)
    second = build_versioned_dataset("demo", "1", cases)
    assert first.fingerprint == second.fingerprint


def test_versioned_run_records_provider_model_and_dataset() -> None:
    dataset = build_versioned_dataset(
        "demo",
        "1",
        [EvalCase("a", "hello", expected_terms={"hello"})],
    )

    def candidate(case: EvalCase) -> ModelOutput:
        return ModelOutput(
            text="hello",
            citations=[],
            latency_ms=10,
            input_tokens=1,
            output_tokens=1,
            provider="openai-compatible",
            model="test-model",
        )

    result = run_versioned_experiment(
        ExperimentRunner(min_citation_coverage=0),
        name="candidate",
        dataset=dataset,
        candidate=candidate,
        prompt_version="p1",
        git_revision="abc123",
    )

    assert result.manifest.dataset_fingerprint == dataset.fingerprint
    assert result.manifest.provider == "openai-compatible"
    assert result.manifest.model == "test-model"
    assert result.manifest.prompt_version == "p1"
