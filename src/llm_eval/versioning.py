from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

from llm_eval.dataset import dataset_fingerprint
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


def build_versioned_dataset(
    dataset_id: str,
    version: str,
    cases: list[EvalCase],
) -> VersionedDataset:
    if not dataset_id.strip() or not version.strip():
        raise ValueError("dataset_id and version are required")
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Dataset requires non-empty cases with unique IDs.")
    return VersionedDataset(
        id=dataset_id,
        version=version,
        fingerprint=dataset_fingerprint(cases),
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
    providers: list[str | None] = []
    models: list[str | None] = []

    def observed(case: EvalCase) -> ModelOutput:
        output = candidate(case)
        providers.append(output.provider)
        models.append(output.model)
        return output

    summary = runner.run(name, list(dataset.cases), observed)
    # None is a legitimate observation ("not reported"), so it cannot double as
    # the "nothing seen yet" sentinel: any disagreement, including None vs a
    # value, is reported as "mixed".
    observed_provider = providers[0] if len(set(providers)) == 1 else "mixed"
    observed_model = models[0] if len(set(models)) == 1 else "mixed"
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
