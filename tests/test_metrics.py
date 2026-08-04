"""Unit tests for per-impression ranking metrics (Milestone 5).

Expected values are either simple exact fractions (AUC, MRR) or computed via
an independent, obviously-correct formula inline (nDCG) -- never copied from
the function under test, and never hand-derived decimal literals for
anything non-trivial, since those are easy to get subtly wrong by hand.

Run with: pytest tests/test_metrics.py -v
"""

import math

import numpy as np
import pytest

from news_reco.evaluation.metrics import auc_score, ndcg_at_k, reciprocal_rank, recall_at_k

# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


def test_auc_perfect_ranking_scores_one():
    labels = np.array([1, 0, 0])
    scores = np.array([3, 2, 1])  # the one positive is ranked highest
    assert auc_score(labels, scores) == pytest.approx(1.0)


def test_auc_worst_ranking_scores_zero():
    labels = np.array([0, 0, 1])
    scores = np.array([3, 2, 1])  # the one positive is ranked lowest
    assert auc_score(labels, scores) == pytest.approx(0.0)


def test_auc_tied_scores_average_to_half():
    labels = np.array([1, 0])
    scores = np.array([1, 1])  # tied score -> "coin flip" -> 0.5
    assert auc_score(labels, scores) == pytest.approx(0.5)


def test_auc_partial_ordering():
    # Positives at scores {4, 1}, negatives at scores {3, 2}.
    # Correctly-ordered (positive, negative) pairs: (4,3), (4,2) -> 2 correct.
    # Incorrectly-ordered: (1,3), (1,2) -> 2 incorrect. AUC = 2 / 4 = 0.5.
    labels = np.array([1, 1, 0, 0])
    scores = np.array([4, 1, 3, 2])
    assert auc_score(labels, scores) == pytest.approx(0.5)


def test_auc_undefined_without_both_classes():
    all_positive = np.array([1, 1, 1])
    all_negative = np.array([0, 0, 0])
    scores = np.array([3, 2, 1])
    assert auc_score(all_positive, scores) is None
    assert auc_score(all_negative, scores) is None


# ---------------------------------------------------------------------------
# Reciprocal rank
# ---------------------------------------------------------------------------


def test_reciprocal_rank_of_first_hit():
    labels = np.array([0, 1, 0, 0])
    scores = np.array([4, 3, 2, 1])  # descending -> ranked order == input order
    # First clicked item is at rank 2 -> RR = 1/2.
    assert reciprocal_rank(labels, scores) == pytest.approx(0.5)


def test_reciprocal_rank_perfect_hit_at_rank_one():
    labels = np.array([1, 0, 0])
    scores = np.array([3, 2, 1])
    assert reciprocal_rank(labels, scores) == pytest.approx(1.0)


def test_reciprocal_rank_undefined_with_no_clicks():
    labels = np.array([0, 0, 0])
    scores = np.array([3, 2, 1])
    assert reciprocal_rank(labels, scores) is None


# ---------------------------------------------------------------------------
# nDCG@k
# ---------------------------------------------------------------------------


def test_ndcg_at_k_distinguishes_shallow_from_deep_cutoff():
    # Relevant items at rank 1 and rank 7 (scores already descending, so
    # rank == position + 1). k=5 only sees the rank-1 hit; k=10 sees both.
    labels = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    scores = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)

    # Independent, obviously-correct computation -- not calling ndcg_at_k.
    expected_idcg = 1 / math.log2(1 + 1) + 1 / math.log2(2 + 1)  # ideal: both hits at rank 1,2
    expected_dcg_at_5 = 1 / math.log2(1 + 1)  # only the rank-1 hit falls within top-5
    expected_dcg_at_10 = 1 / math.log2(1 + 1) + 1 / math.log2(7 + 1)  # both hits within top-10

    assert ndcg_at_k(labels, scores, k=5) == pytest.approx(expected_dcg_at_5 / expected_idcg)
    assert ndcg_at_k(labels, scores, k=10) == pytest.approx(expected_dcg_at_10 / expected_idcg)
    # And they must actually differ -- otherwise this test wouldn't be
    # exercising the k-cutoff logic at all.
    assert ndcg_at_k(labels, scores, k=5) < ndcg_at_k(labels, scores, k=10)


def test_ndcg_at_k_perfect_ranking_scores_one():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([4, 3, 2, 1])  # both relevant items already at the top
    assert ndcg_at_k(labels, scores, k=10) == pytest.approx(1.0)


def test_ndcg_at_k_undefined_with_no_clicks():
    labels = np.array([0, 0, 0])
    scores = np.array([3, 2, 1])
    assert ndcg_at_k(labels, scores, k=10) is None


# ---------------------------------------------------------------------------
# recall@k
# ---------------------------------------------------------------------------


def test_recall_at_k_counts_hits_within_top_k():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"c", "z"}  # "c" is in top-5, "z" was never retrieved at all
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(0.5)


def test_recall_at_k_ignores_hits_beyond_k():
    retrieved = ["a", "b", "c"]
    relevant = {"c"}
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_recall_at_k_undefined_with_no_relevant_items():
    assert recall_at_k(["a", "b"], set(), k=5) is None
