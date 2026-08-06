"""Unit tests for EmbeddingIndex and query construction (Milestone 7).

Uses tiny, hand-constructed 2D vectors where cosine similarity is
obviously computable by hand (parallel = 1.0, orthogonal = 0.0, opposite =
-1.0) -- no dependency on sentence-transformers or a real model here,
since these test the search/pooling logic, not embedding quality.

Run with: pytest tests/test_embeddings.py -v
"""

import numpy as np
import pytest

from news_reco.retrieval.semantic.embeddings import EmbeddingIndex, build_query_embedding

DOC_IDS = ["parallel", "orthogonal", "opposite"]
EMBEDDINGS = np.array(
    [
        [1.0, 0.0],  # parallel to the query below
        [0.0, 1.0],  # orthogonal to the query below
        [-1.0, 0.0],  # opposite of the query below
    ]
)


@pytest.fixture
def index() -> EmbeddingIndex:
    idx = EmbeddingIndex()
    idx.fit(DOC_IDS, EMBEDDINGS)
    return idx


def test_search_ranks_by_cosine_similarity(index: EmbeddingIndex):
    query = np.array([1.0, 0.0])
    results = dict(index.search(query, top_k=3))

    assert results["parallel"] == pytest.approx(1.0)
    assert results["orthogonal"] == pytest.approx(0.0)
    assert results["opposite"] == pytest.approx(-1.0)

    ranked_ids = [doc_id for doc_id, _ in index.search(query, top_k=3)]
    assert ranked_ids == ["parallel", "orthogonal", "opposite"]


def test_search_normalizes_a_non_unit_query(index: EmbeddingIndex):
    # A query of [5, 0] points the same direction as [1, 0] -- cosine
    # similarity only cares about direction, so the result must be
    # identical to the unit-length query case.
    query = np.array([5.0, 0.0])
    results = dict(index.search(query, top_k=3))
    assert results["parallel"] == pytest.approx(1.0)


def test_search_respects_top_k(index: EmbeddingIndex):
    query = np.array([1.0, 0.0])
    results = index.search(query, top_k=1)
    assert len(results) == 1
    assert results[0][0] == "parallel"


def test_search_before_fit_raises():
    idx = EmbeddingIndex()
    with pytest.raises(RuntimeError):
        idx.search(np.array([1.0, 0.0]), top_k=1)


def test_score_documents_matches_search_for_a_fixed_candidate_set(index: EmbeddingIndex):
    query = np.array([1.0, 0.0])
    scored = index.score_documents(query, ["parallel", "orthogonal", "opposite"])
    assert scored["parallel"] == pytest.approx(1.0)
    assert scored["orthogonal"] == pytest.approx(0.0)
    assert scored["opposite"] == pytest.approx(-1.0)


def test_score_documents_handles_unknown_doc_id(index: EmbeddingIndex):
    scored = index.score_documents(np.array([1.0, 0.0]), ["parallel", "not_a_real_doc"])
    assert scored["not_a_real_doc"] == pytest.approx(0.0)


def test_score_documents_none_query_scores_everything_zero(index: EmbeddingIndex):
    scored = index.score_documents(None, ["parallel", "orthogonal"])
    assert scored == {"parallel": 0.0, "orthogonal": 0.0}


def test_build_query_embedding_mean_pools_history():
    embeddings = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    result = build_query_embedding(["a", "b"], get_embedding=lambda aid: embeddings[aid])
    assert np.allclose(result, [0.5, 0.5])


def test_build_query_embedding_empty_history_returns_none():
    result = build_query_embedding([], get_embedding=lambda aid: np.array([1.0, 0.0]))
    assert result is None
