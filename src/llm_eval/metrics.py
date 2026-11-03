import re

from llm_eval.models import EvalCase, ModelOutput

_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")


def token_sequence(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def tokenize(text: str) -> set[str]:
    return set(token_sequence(text))


def _contains_phrase(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return any(tuple(tokens[i : i + width]) == phrase for i in range(len(tokens) - width + 1))


def relevance(case: EvalCase, output: ModelOutput) -> float:
    """Fraction of expected terms present in the output.

    Each expected term is tokenized with the same rule as the output and must
    appear as a contiguous token sequence, so multi-word terms such as
    "machine learning" can match. Terms are de-duplicated after normalisation.
    A term with no word characters can never match.
    """
    if not case.expected_terms:
        return 1.0
    output_tokens = token_sequence(output.text)
    wanted = {tuple(token_sequence(term)) for term in case.expected_terms}
    matched = sum(1 for phrase in wanted if phrase and _contains_phrase(output_tokens, phrase))
    return matched / len(wanted)


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
