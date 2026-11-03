"""Deterministic citation validation and ranked-retrieval metrics.

These checks compare identifiers only. A valid citation set means every cited
id was among the evidence ids supplied to the generator; it does not show that
the cited evidence supports the answer (no entailment check is performed).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import log2


def _ids(values: Sequence[str], name: str) -> list[str]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of strings, not a string")
    items = list(values)
    if any(not isinstance(value, str) or not value for value in items):
        raise ValueError(f"{name} must contain non-empty strings")
    return items


@dataclass(frozen=True, slots=True)
class CitationValidation:
    valid: bool
    cited_count: int
    unknown_ids: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    uncited_evidence_ids: tuple[str, ...]
    citation_precision: float | None
    evidence_coverage: float | None


def validate_citations(
    cited_ids: Sequence[str],
    evidence_ids: Sequence[str],
    *,
    require_citation: bool = False,
) -> CitationValidation:
    """Check that cited ids are a subset of the evidence ids provided.

    ``citation_precision`` is |unique cited ∩ evidence| / |unique cited| (None
    when nothing is cited). ``evidence_coverage`` is |unique cited ∩ evidence|
    / |unique evidence| (None when no evidence was provided). The set is valid
    when it has no unknown ids and, if ``require_citation``, at least one
    citation. Duplicate citations are reported but do not invalidate the set.
    """
    cited = _ids(cited_ids, "cited_ids")
    evidence = _ids(evidence_ids, "evidence_ids")
    evidence_set = set(evidence)
    unique_cited = list(dict.fromkeys(cited))
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in cited:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    unknown = tuple(value for value in unique_cited if value not in evidence_set)
    supported = len(unique_cited) - len(unknown)
    uncited = tuple(value for value in dict.fromkeys(evidence) if value not in seen)
    valid = not unknown and (bool(unique_cited) or not require_citation)
    return CitationValidation(
        valid=valid,
        cited_count=len(cited),
        unknown_ids=unknown,
        duplicate_ids=tuple(duplicates),
        uncited_evidence_ids=uncited,
        citation_precision=supported / len(unique_cited) if unique_cited else None,
        evidence_coverage=supported / len(evidence_set) if evidence_set else None,
    )


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    k: int
    retrieved_count: int
    relevant_count: int
    hits_at_k: int
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    *,
    k: int,
) -> RetrievalMetrics:
    """Binary-relevance ranking metrics for one query.

    - precision@k = hits in top k / k (positions past the end of a short list
      count as misses, the standard fixed-denominator definition)
    - recall@k = hits in top k / |relevant|
    - reciprocal rank = 1 / rank of the first relevant id in the full list, 0 if none
    - nDCG@k = DCG@k / IDCG@k with gain 1 and discount 1 / log2(rank + 1)

    Duplicate retrieved ids are rejected because they would inflate hits.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    retrieved = _ids(retrieved_ids, "retrieved_ids")
    relevant = set(_ids(relevant_ids, "relevant_ids"))
    if not relevant:
        raise ValueError("relevant_ids must be non-empty")
    if len(set(retrieved)) != len(retrieved):
        raise ValueError("retrieved_ids must not contain duplicates")
    top = retrieved[:k]
    flags = [doc in relevant for doc in top]
    hits = sum(flags)
    dcg = sum(1.0 / log2(rank + 2) for rank, hit in enumerate(flags) if hit)
    idcg = sum(1.0 / log2(rank + 2) for rank in range(min(k, len(relevant))))
    first = next((rank for rank, doc in enumerate(retrieved, start=1) if doc in relevant), None)
    return RetrievalMetrics(
        k=k,
        retrieved_count=len(retrieved),
        relevant_count=len(relevant),
        hits_at_k=hits,
        precision_at_k=hits / k,
        recall_at_k=hits / len(relevant),
        hit_rate_at_k=1.0 if hits else 0.0,
        reciprocal_rank=1.0 / first if first else 0.0,
        ndcg_at_k=dcg / idcg,
    )
