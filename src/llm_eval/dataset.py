from llm_eval.canonical import sha256_hex
from llm_eval.models import EvalCase


def case_payload(case: EvalCase) -> dict[str, object]:
    """Order-independent JSON form of a case (set fields are sorted)."""
    return {
        "id": case.id,
        "prompt": case.prompt,
        "expected_terms": sorted(case.expected_terms),
        "expected_citations": sorted(case.expected_citations),
        "forbidden_phrases": sorted(case.forbidden_phrases),
    }


def dataset_fingerprint(cases: list[EvalCase]) -> str:
    """SHA-256 of the canonical JSON list of cases (case order is significant)."""
    return sha256_hex([case_payload(case) for case in cases])
