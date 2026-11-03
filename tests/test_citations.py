"""Hand-computed citation validation and retrieval metric checks."""

from math import log2

import pytest

from llm_eval.citations import retrieval_metrics, validate_citations


def test_citation_validation_reports_unknown_duplicate_and_uncited_ids():
    result = validate_citations(["d1", "d3", "d1"], ["d1", "d2"])
    assert not result.valid
    assert result.cited_count == 3
    assert result.unknown_ids == ("d3",)
    assert result.duplicate_ids == ("d1",)
    assert result.uncited_evidence_ids == ("d2",)
    assert result.citation_precision == 0.5  # {d1} / {d1, d3}
    assert result.evidence_coverage == 0.5  # {d1} / {d1, d2}


def test_citation_subset_is_valid():
    result = validate_citations(["d2", "d1"], ["d1", "d2", "d3"])
    assert result.valid
    assert result.unknown_ids == ()
    assert result.citation_precision == 1.0
    assert result.evidence_coverage == 2 / 3


def test_empty_citations_valid_only_when_not_required():
    assert validate_citations([], ["d1"]).valid
    assert validate_citations([], ["d1"]).citation_precision is None
    assert not validate_citations([], ["d1"], require_citation=True).valid


def test_citations_without_evidence_are_invalid():
    result = validate_citations(["d1"], [])
    assert not result.valid
    assert result.evidence_coverage is None


@pytest.mark.parametrize(
    ("cited", "evidence"),
    [("d1", ["d1"]), (["d1"], "d1"), ([""], ["d1"]), ([1], ["d1"])],
)
def test_citation_validation_rejects_malformed_ids(cited, evidence):
    with pytest.raises((TypeError, ValueError)):
        validate_citations(cited, evidence)


def test_retrieval_metrics_hand_computed():
    result = retrieval_metrics(["a", "x", "b", "y"], ["a", "b", "c"], k=3)
    assert result.hits_at_k == 2
    assert result.precision_at_k == 2 / 3
    assert result.recall_at_k == 2 / 3
    assert result.hit_rate_at_k == 1.0
    assert result.reciprocal_rank == 1.0
    dcg = 1 / log2(2) + 1 / log2(4)  # ranks 1 and 3
    idcg = 1 / log2(2) + 1 / log2(3) + 1 / log2(4)
    assert result.ndcg_at_k == pytest.approx(dcg / idcg)
    assert result.ndcg_at_k == pytest.approx(0.703918, abs=1e-6)


def test_reciprocal_rank_uses_full_list_beyond_k():
    result = retrieval_metrics(["x", "y", "a"], ["a"], k=2)
    assert result.hits_at_k == 0
    assert result.reciprocal_rank == pytest.approx(1 / 3)
    assert result.ndcg_at_k == 0.0


def test_short_ranked_list_uses_fixed_k_denominator():
    result = retrieval_metrics(["a"], ["a"], k=5)
    assert result.precision_at_k == 0.2
    assert result.recall_at_k == 1.0
    assert result.ndcg_at_k == 1.0


def test_no_results_scores_zero():
    result = retrieval_metrics([], ["a"], k=3)
    assert (result.precision_at_k, result.recall_at_k, result.reciprocal_rank) == (0, 0, 0)


@pytest.mark.parametrize(
    ("retrieved", "relevant", "k"),
    [(["a", "a"], ["a"], 2), (["a"], [], 1), (["a"], ["a"], 0), (["a"], ["a"], True)],
)
def test_retrieval_metrics_reject_invalid_input(retrieved, relevant, k):
    with pytest.raises(ValueError):
        retrieval_metrics(retrieved, relevant, k=k)
