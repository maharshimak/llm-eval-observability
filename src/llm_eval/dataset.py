import hashlib
import json
from dataclasses import asdict

from llm_eval.models import EvalCase


def dataset_fingerprint(cases: list[EvalCase]) -> str:
    canonical = [
        {
            **asdict(case),
            "expected_terms": sorted(case.expected_terms),
            "expected_citations": sorted(case.expected_citations),
            "forbidden_phrases": sorted(case.forbidden_phrases),
        }
        for case in cases
    ]
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
