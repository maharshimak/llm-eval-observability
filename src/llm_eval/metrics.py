import re

from llm_eval.models import EvalCase, ModelOutput


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_'-]+", text.lower()))


def relevance(case: EvalCase, output: ModelOutput) -> float:
    if not case.expected_terms:
        return 1.0
    output_terms = tokenize(output.text)
    wanted = {term.lower() for term in case.expected_terms}
    return len(output_terms.intersection(wanted)) / len(wanted)


def citation_coverage(case: EvalCase, output: ModelOutput) -> float:
    if not case.expected_citations:
        return 1.0
    seen = set(output.citations)
    return len(seen.intersection(case.expected_citations)) / len(case.expected_citations)


def contains_forbidden(case: EvalCase, output: ModelOutput) -> bool:
    text = output.text.lower()
    return any(phrase.lower() in text for phrase in case.forbidden_phrases)


def estimated_cost(
    output: ModelOutput,
    input_per_million: float,
    output_per_million: float,
) -> float:
    return (
        output.input_tokens * input_per_million + output.output_tokens * output_per_million
    ) / 1_000_000
