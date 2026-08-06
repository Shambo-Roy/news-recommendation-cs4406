"""Unit tests for beyond-accuracy metrics: diversity, novelty, coverage (Milestone 8).

Run with: pytest tests/test_beyond_accuracy.py -v
"""

import math

import numpy as np
import pytest

from news_reco.evaluation.beyond_accuracy import CoverageTracker, NoveltyModel, intra_list_diversity

# ---------------------------------------------------------------------------
# Intra-list diversity
# ---------------------------------------------------------------------------

EMBEDDINGS = {
    "a": np.array([1.0, 0.0]),
    "b": np.array([0.0, 1.0]),  # orthogonal to a -> similarity 0, diversity 1
    "c": np.array([1.0, 0.0]),  # identical to a -> similarity 1, diversity 0
}


def _get_embedding(article_id: str) -> np.ndarray:
    return EMBEDDINGS[article_id]


def test_diversity_of_orthogonal_items_is_one():
    assert intra_list_diversity(["a", "b"], _get_embedding) == pytest.approx(1.0)


def test_diversity_of_identical_items_is_zero():
    assert intra_list_diversity(["a", "c"], _get_embedding) == pytest.approx(0.0)


def test_diversity_undefined_for_fewer_than_two_items():
    assert intra_list_diversity(["a"], _get_embedding) is None
    assert intra_list_diversity([], _get_embedding) is None


def test_diversity_averages_all_pairs():
    # Pairs: (a,b)=0 similarity, (a,c)=1 similarity, (b,c)=0 similarity.
    # Mean similarity = 1/3 -> diversity = 1 - 1/3 = 2/3.
    result = intra_list_diversity(["a", "b", "c"], _get_embedding)
    assert result == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# NoveltyModel
# ---------------------------------------------------------------------------


def test_novelty_scores_match_hand_computed_formula():
    model = NoveltyModel(train_click_counts={"popular": 10, "rare": 0}, n_articles=2)

    expected_popular = -math.log2((10 + 1) / (10 + 2))
    expected_rare = -math.log2((0 + 1) / (10 + 2))

    assert model.score("popular") == pytest.approx(expected_popular)
    assert model.score("rare") == pytest.approx(expected_rare)


def test_novelty_rarer_items_score_higher():
    model = NoveltyModel(train_click_counts={"popular": 10, "rare": 0}, n_articles=2)
    assert model.score("rare") > model.score("popular")


def test_novelty_unseen_article_matches_zero_count():
    model = NoveltyModel(train_click_counts={"popular": 10}, n_articles=2)
    # "never_in_train" has no entry at all -- must be treated identically
    # to an explicit count of 0, not crash or return something different.
    expected = -math.log2((0 + 1) / (10 + 2))
    assert model.score("never_in_train") == pytest.approx(expected)


def test_mean_novelty_averages_the_list():
    model = NoveltyModel(train_click_counts={"a": 5, "b": 1}, n_articles=2)
    expected = (model.score("a") + model.score("b")) / 2
    assert model.mean_score(["a", "b"]) == pytest.approx(expected)


def test_mean_novelty_undefined_for_empty_list():
    model = NoveltyModel(train_click_counts={}, n_articles=2)
    assert model.mean_score([]) is None


# ---------------------------------------------------------------------------
# CoverageTracker
# ---------------------------------------------------------------------------


def test_coverage_tracks_union_across_multiple_lists():
    tracker = CoverageTracker(n_articles=10)
    tracker.add(["a", "b"])
    tracker.add(["b", "c"])  # "b" repeated -- must not double count
    assert tracker.coverage == pytest.approx(3 / 10)


def test_coverage_zero_articles_does_not_divide_by_zero():
    tracker = CoverageTracker(n_articles=0)
    assert tracker.coverage == 0.0
