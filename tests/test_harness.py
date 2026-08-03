"""Unit tests for the evaluation harness (Milestone 5).

Run with: pytest tests/test_harness.py -v
"""

import numpy as np
import pytest

from news_reco.evaluation.harness import aggregate, evaluate_ranking


def test_evaluate_ranking_returns_all_four_metrics():
    labels = np.array([1, 0, 0])
    scores = np.array([3, 2, 1])
    result = evaluate_ranking(labels, scores)
    assert set(result.keys()) == {"auc", "mrr", "ndcg@5", "ndcg@10"}
    assert result["auc"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)


def test_evaluate_ranking_returns_none_for_undefined_metrics_on_no_click_impression():
    labels = np.array([0, 0, 0])
    scores = np.array([3, 2, 1])
    result = evaluate_ranking(labels, scores)
    assert result["auc"] is None
    assert result["mrr"] is None
    assert result["ndcg@5"] is None
    assert result["ndcg@10"] is None


def test_aggregate_computes_correct_mean_and_exclusion_counts():
    per_impression = [
        {"auc": 1.0, "mrr": 1.0, "ndcg@5": 1.0, "ndcg@10": 1.0},
        {"auc": 0.0, "mrr": 0.5, "ndcg@5": 0.5, "ndcg@10": 0.5},
        # A no-click impression: every metric undefined, like evaluate_ranking
        # would actually return.
        {"auc": None, "mrr": None, "ndcg@5": None, "ndcg@10": None},
    ]

    summary = aggregate(per_impression)

    assert summary.loc["auc", "n_included"] == 2
    assert summary.loc["auc", "n_excluded"] == 1
    assert summary.loc["auc", "mean"] == pytest.approx(0.5)
    assert summary.loc["mrr", "mean"] == pytest.approx(0.75)
    # The CI must bracket the mean it's describing.
    assert summary.loc["auc", "ci_lower"] <= summary.loc["auc", "mean"] <= summary.loc["auc", "ci_upper"]
