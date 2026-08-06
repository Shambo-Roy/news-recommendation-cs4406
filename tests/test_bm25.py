"""Unit tests for BM25 (Milestone 6).

Expected scores are computed independently via math.log and the BM25
formula written out by hand in each test, never by calling the
implementation and trusting its own output.

Run with: pytest tests/test_bm25.py -v
"""

import math

import pytest

from news_reco.retrieval.lexical.bm25 import BM25Index, build_query_from_history, tokenize

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_drops_stopwords():
    assert tokenize("the cat and the dog") == ["cat", "dog"]


# ---------------------------------------------------------------------------
# BM25Index -- hand-computed scores
# ---------------------------------------------------------------------------
#
# doc1: "apple banana apple"          -> length 3, apple:2, banana:1
# doc2: "banana cherry"                -> length 2, banana:1, cherry:1
# doc3: "apple cherry cherry cherry"   -> length 4, apple:1, cherry:3
#
# avgdl = (3+2+4)/3 = 3.0; N=3; every term's doc_freq = 2 (each term is in
# exactly 2 of the 3 documents), so all three terms share one IDF value.

DOC_IDS = ["doc1", "doc2", "doc3"]
TEXTS = ["apple banana apple", "banana cherry", "apple cherry cherry cherry"]
K1, B = 1.5, 0.75
AVGDL = 3.0
IDF = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)  # == math.log(1.6), independent of the module


def _bm25_term_score(term_freq: int, doc_length: int) -> float:
    """The BM25 formula, written independently of BM25Index.search()."""
    denom = term_freq + K1 * (1 - B + B * doc_length / AVGDL)
    return IDF * (term_freq * (K1 + 1)) / denom


@pytest.fixture
def index() -> BM25Index:
    idx = BM25Index(k1=K1, b=B)
    idx.fit(DOC_IDS, TEXTS)
    return idx


def test_single_term_query_matches_hand_computed_scores(index: BM25Index):
    results = dict(index.search("apple", top_k=10))

    expected_doc1 = _bm25_term_score(term_freq=2, doc_length=3)
    expected_doc3 = _bm25_term_score(term_freq=1, doc_length=4)

    assert results["doc1"] == pytest.approx(expected_doc1)
    assert results["doc3"] == pytest.approx(expected_doc3)
    assert "doc2" not in results  # doc2 has zero occurrences of "apple"


def test_higher_term_frequency_ranks_higher_here(index: BM25Index):
    # doc1 has "apple" twice (in a short document); doc3 has it once (in a
    # longer document) -- both saturation and length norm push doc1 above doc3.
    results = index.search("apple", top_k=10)
    ranked_ids = [doc_id for doc_id, _ in results]
    assert ranked_ids == ["doc1", "doc3"]


def test_multi_term_query_sums_per_term_scores(index: BM25Index):
    results = dict(index.search("apple cherry", top_k=10))

    expected_doc3 = _bm25_term_score(term_freq=1, doc_length=4) + _bm25_term_score(term_freq=3, doc_length=4)
    assert results["doc3"] == pytest.approx(expected_doc3)


def test_score_documents_matches_search_for_a_fixed_candidate_set(index: BM25Index):
    # score_documents() and search() compute the same formula -- for a
    # candidate set search() would return, the scores must agree exactly.
    search_results = dict(index.search("apple", top_k=10))
    scored = index.score_documents("apple", ["doc1", "doc2", "doc3"])

    assert scored["doc1"] == pytest.approx(search_results["doc1"])
    assert scored["doc3"] == pytest.approx(search_results["doc3"])
    assert scored["doc2"] == pytest.approx(0.0)  # doc2 has zero occurrences of "apple"


def test_score_documents_handles_unknown_doc_id(index: BM25Index):
    scored = index.score_documents("apple", ["doc1", "not_a_real_doc"])
    assert scored["not_a_real_doc"] == pytest.approx(0.0)
    assert scored["doc1"] > 0.0


def test_search_respects_top_k():
    idx = BM25Index()
    idx.fit(DOC_IDS, TEXTS)
    results = idx.search("apple banana cherry", top_k=2)
    assert len(results) <= 2


def test_search_with_no_matching_terms_returns_empty():
    idx = BM25Index()
    idx.fit(DOC_IDS, TEXTS)
    assert idx.search("zebra giraffe", top_k=10) == []


def test_search_with_empty_query_returns_empty():
    idx = BM25Index()
    idx.fit(DOC_IDS, TEXTS)
    assert idx.search("", top_k=10) == []


# ---------------------------------------------------------------------------
# Query construction from history
# ---------------------------------------------------------------------------


def test_build_query_from_history_concatenates_titles():
    titles = {"N1": "Queen Elizabeth News", "N2": "Weight Loss Tips"}
    query = build_query_from_history(["N1", "N2"], get_title=lambda aid: titles[aid])
    assert query == "Queen Elizabeth News Weight Loss Tips"


def test_build_query_from_history_empty_history_yields_empty_query():
    query = build_query_from_history([], get_title=lambda aid: "unused")
    assert query == ""
