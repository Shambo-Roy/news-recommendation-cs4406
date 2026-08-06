"""Beyond-accuracy metrics: intra-list diversity, novelty, and coverage.

Accuracy metrics (AUC, MRR, nDCG, recall@K) all ask "did we rank/retrieve
the item the user actually clicked?" They say nothing about *what kind* of
recommendations a system produces -- a system that always recommends the
same handful of extremely popular articles could score well on accuracy
while being useless as a discovery tool and unfair to the long tail of the
catalog. These three metrics measure that different axis:

- Diversity: how different are the items *within one list* from each other?
- Novelty: how *unexpected* (non-popular) are the recommended items?
- Coverage: across *many* users' recommendations, how much of the whole
  catalog ever gets shown at all?
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def intra_list_diversity(item_ids: list[str], get_embedding: Callable[[str], np.ndarray]) -> float | None:
    """Mean pairwise dissimilarity (1 - cosine similarity) within a recommended list.

    Uses embeddings as a topical-similarity proxy regardless of which
    method (BM25 or semantic) produced the list -- "how different are
    these items" is a question embeddings can answer even for a
    BM25-retrieved list, since diversity is about the *output*, not how it
    was generated.

    Returns None for a list with fewer than 2 items (no pair to compare).
    """
    if len(item_ids) < 2:
        return None

    vectors = np.stack([get_embedding(article_id) for article_id in item_ids])
    # Rows are L2-normalized (see compute_embeddings), so this one matrix
    # multiply gives every pairwise cosine similarity at once.
    similarity_matrix = vectors @ vectors.T

    n = len(item_ids)
    upper_triangle = np.triu_indices(n, k=1)  # exclude self-pairs and duplicate (i,j)/(j,i) pairs
    pairwise_similarities = similarity_matrix[upper_triangle]

    return float(1.0 - pairwise_similarities.mean())


class NoveltyModel:
    """Self-information novelty scoring, from article popularity in train clicks.

    novelty(item) = -log2(p(item)), where p(item) is how often the item was
    clicked in the *training* split specifically (never val/test -- using
    future popularity to score novelty would be a leakage-flavored
    circularity). Popular articles get low novelty (unsurprising to
    recommend); articles rarely or never clicked in train get high novelty.
    Laplace smoothing (+1 count, +n_articles in the denominator) keeps this
    finite for an article absent from train entirely, rather than
    -log2(0) = infinity.
    """

    def __init__(self, train_click_counts: dict[str, int], n_articles: int) -> None:
        self._counts = train_click_counts
        self._total_clicks = sum(train_click_counts.values())
        self._n_articles = n_articles

    def score(self, article_id: str) -> float:
        count = self._counts.get(article_id, 0)
        probability = (count + 1) / (self._total_clicks + self._n_articles)
        return -math.log2(probability)

    def mean_score(self, item_ids: list[str]) -> float | None:
        """Average novelty across a recommended list; None for an empty list."""
        if len(item_ids) == 0:
            return None
        return sum(self.score(article_id) for article_id in item_ids) / len(item_ids)


class CoverageTracker:
    """Tracks catalog coverage across many recommended lists.

    Unlike diversity/novelty (a property of one list), coverage is
    inherently a property of an entire evaluation run: it asks what
    fraction of the whole catalog is *ever* shown to *anyone*, across every
    impression evaluated. Call add() once per impression's recommended
    list, then read .coverage once at the end.
    """

    def __init__(self, n_articles: int) -> None:
        self._n_articles = n_articles
        self._seen: set[str] = set()

    def add(self, item_ids: list[str]) -> None:
        self._seen.update(item_ids)

    @property
    def coverage(self) -> float:
        if self._n_articles == 0:
            return 0.0
        return len(self._seen) / self._n_articles
