"""Aggregate per-impression metrics into dataset-level results with bootstrap CIs.

news_reco.evaluation.metrics scores one impression at a time; this module is
the "evaluation harness" the assignment spine asks for: run every impression
through the same metric functions, then report the mean -- and its
uncertainty -- across whichever impressions each metric was actually
defined for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from news_reco.evaluation.bootstrap import bootstrap_ci
from news_reco.evaluation.metrics import auc_score, ndcg_at_k, reciprocal_rank

METRIC_NAMES = ("auc", "mrr", "ndcg@5", "ndcg@10")


def evaluate_ranking(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """Compute all four metrics for one impression's ranked candidates."""
    return {
        "auc": auc_score(labels, scores),
        "mrr": reciprocal_rank(labels, scores),
        "ndcg@5": ndcg_at_k(labels, scores, k=5),
        "ndcg@10": ndcg_at_k(labels, scores, k=10),
    }


def aggregate(
    per_impression_results: list[dict[str, float | None]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize per-impression metric dicts into mean + 95% bootstrap CI per metric.

    Returns a DataFrame indexed by metric name, with columns mean, ci_lower,
    ci_upper, n_included (impressions where the metric was defined), and
    n_excluded (where it wasn't -- e.g. an impression with no click at all).
    """
    results_df = pd.DataFrame(per_impression_results)
    total = len(results_df)

    rows = []
    for metric_name in METRIC_NAMES:
        values = results_df[metric_name].dropna().to_numpy(dtype=float)
        n_included = len(values)
        lower, upper = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
        rows.append(
            {
                "metric": metric_name,
                "mean": float(values.mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "n_included": n_included,
                "n_excluded": total - n_included,
            }
        )

    return pd.DataFrame(rows).set_index("metric")
