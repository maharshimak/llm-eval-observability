from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256

from llm_eval.models import EvalCase, ExperimentSummary, ModelOutput
from llm_eval.runner import ExperimentRunner


@dataclass(frozen=True, slots=True)
class VersionedDataset:
    id: str
    version: str
    fingerprint: str
    cases: tuple[EvalCase, ...]


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    name: str
    dataset_id: str
    dataset_version: str
    dataset_fingerprint: str
    provider: str | None
    model: str | None
    prompt_version: str | None
    git_revision: str | None


@dataclass(frozen=True, slots=True)
class VersionedExperimentRun:
    manifest: ExperimentManifest
    summary: ExperimentSummary


def _case_payload(case: EvalCase) -> dict[str, object]:
    return {
        "id": case.id,
        "prompt": case.prompt,
        "expected_terms": sorted(case.expected_terms),
        "expected_citations": sorted(case.expected_citations),
        "forbidden_phrases": sorted(case.forbidden_phrases),
    }


def build_versioned_dataset(
    dataset_id: str,
    version: str,
    cases: list[EvalCase],
) -> VersionedDataset:
    if not dataset_id.strip() or not version.strip():
        raise ValueError("dataset_id and version are required")
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Dataset requires non-empty cases with unique IDs.")
    payload = json.dumps(
        [_case_payload(case) for case in cases],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return VersionedDataset(
        id=dataset_id,
        version=version,
        fingerprint=sha256(payload).hexdigest(),
        cases=tuple(cases),
    )


def run_versioned_experiment(
    runner: ExperimentRunner,
    *,
    name: str,
    dataset: VersionedDataset,
    candidate: Callable[[EvalCase], ModelOutput],
    prompt_version: str | None = None,
    git_revision: str | None = None,
) -> VersionedExperimentRun:
    observed_provider: str | None = None
    observed_model: str | None = None

    def observed(case: EvalCase) -> ModelOutput:
        nonlocal observed_provider, observed_model
        output = candidate(case)
        if observed_provider is None:
            observed_provider = output.provider
        elif output.provider != observed_provider:
            observed_provider = "mixed"
        if observed_model is None:
            observed_model = output.model
        elif output.model != observed_model:
            observed_model = "mixed"
        return output

    summary = runner.run(name, list(dataset.cases), observed)
    manifest = ExperimentManifest(
        name=name,
        dataset_id=dataset.id,
        dataset_version=dataset.version,
        dataset_fingerprint=dataset.fingerprint,
        provider=observed_provider,
        model=observed_model,
        prompt_version=prompt_version,
        git_revision=git_revision,
    )
    return VersionedExperimentRun(manifest=manifest, summary=summary)


def export_run_json(run: VersionedExperimentRun) -> str:
    return json.dumps(
        {
            "manifest": asdict(run.manifest),
            "summary": asdict(run.summary),
        },
        sort_keys=True,
        indent=2,
    )
