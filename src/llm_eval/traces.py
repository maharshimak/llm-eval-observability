import json
from dataclasses import asdict
from pathlib import Path

from llm_eval.models import CaseMetrics


class JsonlTraceStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, experiment: str, metric: CaseMetrics) -> None:
        record = {"experiment": experiment, **asdict(metric)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
