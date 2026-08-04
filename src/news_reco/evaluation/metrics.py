"""Per-impression ranking metrics: AUC, reciprocal rank, and nDCG@k.

Each function scores *one impression* -- a ranked list of candidate
articles with binary relevance labels (1 = clicked, 0 = not clicked) -- and
returns a float, or None when the metric is undefined for that impression
(e.g. AUC needs at least one click and one non-click to form a pair; MRR and
nDCG need at least one click). Undefined results must be excluded from an
average, not silently treated as 0 -- news_reco.evaluation.harness handles
that aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Probability a random clicked article outranks a random non-clicked one.

    Computed via the Mann-Whitney U / rank-sum identity rather than counting
    pairs directly: AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) /
    (n_pos * n_neg), where ranks are ascending (1 = lowest score) with ties
    given the average rank. This is O(n log n) instead of O(n^2) and,
    critically, handles tied scores correctly without extra code -- two
    candidates in this project's data very often tie (integer-ish BM25/count
    scores), so tie handling isn't an edge case, it's routine.

    Returns None if every label is the same (no positive/negative pair to
    compare -- an impression with zero clicks, or where everything was
    clicked, has no notion of "ranked correctly").
    """
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = pd.Series(scores).rank(method="average", ascending=True)
    sum_positive_ranks = ranks[labels == 1].sum()
    return (sum_positive_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def reciprocal_rank(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """1 / (rank of the first clicked article), 1-indexed by descending score.

    Returns None if the impression has no clicked article at all.
    """
    if labels.sum() == 0:
        return None

    order = np.argsort(-scores, kind="stable")  # stable: ties keep input order
    ranked_labels = labels[order]
    first_hit_rank = int(np.argmax(ranked_labels)) + 1  # argmax finds the first 1
    return 1.0 / first_hit_rank


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float | None:
    """Normalized discounted cumulative gain of the top-k ranked candidates.

    DCG@k sums 1/log2(rank+1) for each clicked article within the top k
    (rank 1-indexed); nDCG@k divides that by the best possible DCG@k -- the
    value if every clicked article were placed first. Binary relevance makes
    the usual (2^rel - 1) gain term collapse to just `rel`, so it's omitted.

    Returns None if the impression has no clicked article (IDCG@k would be
    zero, making the ratio undefined).
    """
    n_relevant = int(labels.sum())
    if n_relevant == 0:
        return None

    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order][:k]
    ranks = np.arange(1, len(ranked_labels) + 1)
    discounts = 1.0 / np.log2(ranks + 1)
    dcg = float((ranked_labels * discounts).sum())

    ideal_hits = min(n_relevant, k)
    ideal_ranks = np.arange(1, ideal_hits + 1)
    idcg = float((1.0 / np.log2(ideal_ranks + 1)).sum())

    return dcg / idcg


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float | None:
    """Fraction of relevant (ground-truth clicked) articles retrieved in the top-k.

    This evaluates a *candidate generator* (BM25, embeddings), not a final
    ranking: out of everything the user actually clicked, how much of it
    did retrieval even manage to surface in its top-k, before any
    reranking happens? Method-agnostic by design -- it takes a plain list
    of retrieved IDs, so the same function evaluates BM25 (Milestone 6) and
    embedding-based retrieval (Milestone 7) identically.

    Returns None if there are no relevant items to find (nothing to
    compute recall against).
    """
    if len(relevant_ids) == 0:
        return None
    retrieved_top_k = set(retrieved_ids[:k])
    hits = len(retrieved_top_k & relevant_ids)
    return hits / len(relevant_ids)
