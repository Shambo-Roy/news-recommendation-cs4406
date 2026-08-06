"""Reciprocal Rank Fusion: combine two ranked candidate lists into one.

RRF combines *rank positions* rather than raw scores -- BM25 scores are
unbounded and corpus-scale-dependent, cosine similarity is bounded
[-1, 1]; averaging them directly would implicitly weight one method more
than the other depending on arbitrary scale, with no principled way to
correct for it. RRF sidesteps the problem entirely: a document's fused
score is 1/(k + rank) summed across every list it appears in, so only
*where* a document ranks in each list matters, never *how much* one
method's raw scores happen to spread out.
"""

from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of doc_ids into one, by rank position.

    Args:
        ranked_lists: One or more lists of doc_ids, each already sorted
            best-first by that method's own scoring.
        k: RRF's smoothing constant. Higher k flattens the influence of
            rank differences (rank 1 vs rank 50 matters less); 60 is the
            standard default from the original RRF paper (Cormack et al.,
            2009) and from most production hybrid-search implementations.

    Returns:
        (doc_id, fused_score) pairs, sorted descending by fused score. A
        document appearing in more than one list accumulates a
        contribution from each -- that consensus effect is the point of
        fusion, not just an averaging convenience.
    """
    fused_scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)


def fuse_score_dicts(score_dicts: list[dict[str, float]], k: int = 60) -> dict[str, float]:
    """RRF-fuse multiple methods' scores over the *same* fixed candidate set.

    Converts each method's {doc_id: score} into a locally-ranked list (best
    first, among just this candidate set), then fuses by rank via
    reciprocal_rank_fusion(). This is what makes a hybrid ranking over an
    impression's actual shown candidates well-defined and cheap: it never
    needs each method's rank across the *whole* catalog, only among the
    handful of candidates actually being compared. For open-corpus hybrid
    retrieval (recall@K, beyond-accuracy over a top-N list), call
    reciprocal_rank_fusion() directly on each method's search() results
    instead.
    """
    ranked_lists = [
        [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        for scores in score_dicts
    ]
    return dict(reciprocal_rank_fusion(ranked_lists, k=k))
