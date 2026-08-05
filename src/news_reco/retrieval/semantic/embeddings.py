"""Semantic retrieval: article embeddings + brute-force nearest-neighbor search.

Encodes article text into dense vectors with a pretrained sentence encoder,
then finds nearest neighbors to a query vector by exact cosine similarity.
At this corpus size (tens of thousands of articles) a brute-force matrix
multiply is both simpler and faster than standing up an approximate index
would be -- the assignment explicitly permits this for small-scale corpora.
"""

from __future__ import annotations

import os

# Harmless extra safety net for the classic Intel-OpenMP duplicate-runtime
# crash. It did NOT turn out to be the actual cause of this project's DLL
# failure on Windows (see below), but costs nothing to also set.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from collections.abc import Callable  # noqa: E402

import numpy as np  # noqa: E402

# The real fix, found by isolating the failure empirically rather than
# trusting the first plausible-sounding explanation: torch's bundled native
# libraries conflict with pyarrow's (both ship their own C++ runtime
# components) on Windows, crashing with "DLL initialization routine
# failed" -- but only if pyarrow/pandas gets imported into the process
# *first*. Importing torch (via sentence_transformers) before anything that
# pulls in pandas/pyarrow avoids the conflict entirely; reversed, no known
# env var fixes it. This module only *lazily* imports sentence_transformers
# inside compute_embeddings() (so EmbeddingIndex/build_query_embedding stay
# usable without that dependency in search-only contexts) -- which means
# **the entry-point script is responsible for importing sentence_transformers
# (or torch) before importing anything that touches pandas/pyarrow**, e.g.
# FeatureStore. See scripts/compute_article_embeddings.py and
# scripts/evaluate_semantic_recall.py for the required import order.


def compute_embeddings(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts into L2-normalized embedding vectors.

    L2-normalizing here (rather than at search time) means cosine
    similarity between two vectors reduces to a plain dot product later --
    cheaper to compute and simpler to reason about at search time.

    Imports sentence_transformers lazily so that everything else in this
    module (EmbeddingIndex, build_query_embedding) stays usable in
    search-only contexts without requiring that dependency to be installed
    or a model to be downloaded.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard against a degenerate all-zero embedding
    return embeddings / norms


class EmbeddingIndex:
    """Brute-force nearest-neighbor search over precomputed article embeddings."""

    def __init__(self) -> None:
        self._doc_ids: list[str] = []
        self._embeddings: np.ndarray | None = None  # shape (n_docs, dim), L2-normalized rows

    def fit(self, doc_ids: list[str], embeddings: np.ndarray) -> None:
        """Store a precomputed, already-normalized embedding matrix for search.

        Takes embeddings rather than raw text: computing embeddings (model
        inference over the whole catalog) is the expensive, one-time step,
        done once by compute_embeddings() and cached to disk. fit() here is
        deliberately cheap, so search-only runs never need the encoder
        model loaded at all -- mirroring how BM25Index.fit() tokenizes text
        directly, except embeddings are expensive enough to compute that
        splitting "compute" from "index" is worth the extra step.
        """
        self._doc_ids = list(doc_ids)
        self._embeddings = embeddings

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return up to top_k (doc_id, cosine_similarity) pairs, descending."""
        if self._embeddings is None:
            raise RuntimeError("EmbeddingIndex.fit() must be called before search().")
        if top_k <= 0 or len(self._doc_ids) == 0:
            return []

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        normalized_query = query_vector / query_norm

        # Cosine similarity to every document at once: rows are already
        # unit-length (from compute_embeddings), so this dot product *is*
        # cosine similarity, not just proportional to it.
        scores = self._embeddings @ normalized_query

        k = min(top_k, len(scores))
        # argpartition finds the top-k in O(n) without fully sorting
        # everything; only that small slice then gets sorted.
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(self._doc_ids[i], float(scores[i])) for i in top_indices]


def build_query_embedding(
    history: list[str], get_embedding: Callable[[str], np.ndarray]
) -> np.ndarray | None:
    """Mean-pool a user's history article embeddings into one profile vector.

    This is the query-construction method the assignment specifies for
    semantic retrieval (Q3.3): "compute a user representation (e.g.,
    mean-pooled embeddings of clicked articles)". Returns None for an
    empty history -- there's nothing to pool, mirroring how BM25's
    build_query_from_history() returns an empty (unsearchable) string for
    the same cold-start case.
    """
    if len(history) == 0:
        return None
    vectors = np.stack([get_embedding(article_id) for article_id in history])
    return vectors.mean(axis=0)
