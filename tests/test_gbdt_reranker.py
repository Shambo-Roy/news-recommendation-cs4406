"""Unit tests for the GBDT re-ranker's feature-extraction module
(news_reco.reranking.gbdt) -- Component 2, entirely additive to Component 1.

Every expected value below is hand-computed in the test itself (or in a
comment next to it), not just re-derived from the implementation -- the
same discipline used throughout this project's existing test suite (see
tests/test_ebnerd_adapter.py, tests/test_submission.py).

Run with: pytest tests/test_gbdt_reranker.py -v
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from news_reco.reranking.gbdt import (
    ArticleLookup,
    COMMON_FEATURE_NAMES,
    DWELL_FEATURE_NAME,
    RECENCY_FEATURE_NAME,
    TEMPORAL_FEATURE_NAME,
    ExpandingTrainPopularity,
    _ages_hours_strictly_before,
    batch_max_jaccard_overlap,
    batch_recency_weighted_semantic_similarity,
    batch_semantic_history_similarity,
    build_feature_table,
    build_frozen_train_click_counts,
    build_impression_rows,
    label_match_with_history,
    label_match_with_history_union,
    load_ebnerd_history_detail,
    max_jaccard_overlap,
    mean_history_read_time,
    retrieval_rank_features,
    semantic_history_similarity,
    semantic_history_similarity_matrix,
    title_token_overlap_with_history,
)
from news_reco.retrieval.availability import AvailabilityGate
from news_reco.retrieval.lexical.bm25 import BM25Index
from news_reco.retrieval.semantic.embeddings import EmbeddingIndex

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Pure feature functions
# --------------------------------------------------------------------------


def test_title_token_overlap_with_history_max_over_history_jaccard():
    # candidate tokens: {manchester, united, wins, match} (no stopwords dropped here)
    # hist1 tokens: {manchester, city, loses, game} -> intersection={manchester}(1), union has 7 -> 1/7
    # hist2 tokens: {random, topic, here} -> intersection=0 -> 0
    # max-over-history = 1/7
    overlap = title_token_overlap_with_history(
        "Manchester United wins match",
        ["Manchester City loses game", "Random topic here"],
    )
    assert overlap == pytest.approx(1 / 7)


def test_title_token_overlap_with_history_empty_history_is_zero_not_nan():
    assert title_token_overlap_with_history("Some title here", []) == 0.0


def test_title_token_overlap_with_history_empty_candidate_title_is_zero():
    assert title_token_overlap_with_history("", ["Some history title"]) == 0.0


def test_max_jaccard_overlap_matches_title_token_overlap_with_history():
    """The precomputed-sets fast path (used at scale for EB-NeRD's long
    histories) must agree exactly with the reference, tokenize-per-call
    implementation -- same 1/7 hand-computed case as the reference test."""
    from news_reco.retrieval.lexical.bm25 import tokenize

    candidate_title = "Manchester United wins match"
    history_titles = ["Manchester City loses game", "Random topic here"]
    reference = title_token_overlap_with_history(candidate_title, history_titles)

    candidate_tokens = frozenset(tokenize(candidate_title))
    history_token_sets = [frozenset(tokenize(t)) for t in history_titles]
    fast_path = max_jaccard_overlap(candidate_tokens, history_token_sets)

    assert fast_path == reference == pytest.approx(1 / 7)


def test_max_jaccard_overlap_empty_inputs_are_zero():
    assert max_jaccard_overlap(frozenset(), [frozenset({"a"})]) == 0.0
    assert max_jaccard_overlap(frozenset({"a"}), []) == 0.0


def test_batch_max_jaccard_overlap_matches_per_candidate_reference():
    """The fully-vectorized (all candidates at once) path must agree
    exactly with calling max_jaccard_overlap once per candidate -- including
    edge cases: an empty-token candidate, and a history article whose title
    tokenized to nothing."""
    candidate_token_sets = [
        frozenset({"manchester", "united", "wins", "match"}),  # overlaps history[0]
        frozenset(),  # empty candidate -- must be 0.0 regardless of history
        frozenset({"random", "topic"}),  # no overlap with any history
    ]
    history_token_sets = [
        frozenset({"manchester", "city", "loses", "game"}),
        frozenset(),  # history title tokenized to nothing
    ]

    expected = [max_jaccard_overlap(c, history_token_sets) for c in candidate_token_sets]
    actual = batch_max_jaccard_overlap(candidate_token_sets, history_token_sets)

    assert list(actual) == pytest.approx(expected)
    assert expected[0] == pytest.approx(1 / 7)  # same hand-computed value as the reference test
    assert expected[1] == 0.0
    assert expected[2] == 0.0


def test_batch_max_jaccard_overlap_empty_candidates_or_history():
    assert list(batch_max_jaccard_overlap([], [frozenset({"a"})])) == []
    assert list(batch_max_jaccard_overlap([frozenset({"a"}), frozenset({"b"})], [])) == [0.0, 0.0]


def test_batch_semantic_history_similarity_matches_per_candidate_reference():
    candidate_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    history = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
    history_matrix = np.stack(history)

    expected_max, expected_mean = zip(*[semantic_history_similarity(c, history) for c in candidate_embeddings])
    actual_max, actual_mean = batch_semantic_history_similarity(candidate_embeddings, history_matrix)

    assert list(actual_max) == pytest.approx(list(expected_max))
    assert list(actual_mean) == pytest.approx(list(expected_mean))


def test_batch_semantic_history_similarity_empty_history_is_zero_zero():
    candidate_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    max_sims, mean_sims = batch_semantic_history_similarity(candidate_embeddings, None)
    assert list(max_sims) == [0.0, 0.0]
    assert list(mean_sims) == [0.0, 0.0]


def test_label_match_with_history_scalar_category_match_and_no_match():
    assert label_match_with_history("sport", ["news", "sport"]) == 1.0
    assert label_match_with_history("tech", ["news", "sport"]) == 0.0
    assert label_match_with_history("sport", []) == 0.0  # empty history -> no match


def test_label_match_with_history_multilabel_subcategory_list():
    # EB-NeRD-style multi-label subcategory: candidate [1, 2] vs history union {3, 2, 5} -> shares 2
    assert label_match_with_history([1, 2], [[3], [2, 5]]) == 1.0
    assert label_match_with_history([1, 9], [[3], [2, 5]]) == 0.0
    assert label_match_with_history([], [[1, 2]]) == 0.0  # candidate has no label at all


def test_label_match_with_history_union_matches_label_match_with_history():
    """Same fast-path-vs-reference consistency check for the category/
    subcategory match feature."""
    history_values = ["news", "sport"]
    history_union = {"news", "sport"}
    assert label_match_with_history_union("sport", history_union) == label_match_with_history("sport", history_values) == 1.0
    assert label_match_with_history_union("tech", history_union) == label_match_with_history("tech", history_values) == 0.0
    assert label_match_with_history_union("sport", set()) == label_match_with_history("sport", []) == 0.0


def test_semantic_history_similarity_matrix_matches_list_version():
    """The precomputed-matrix fast path (used at scale for EB-NeRD's long
    histories) must agree exactly with the reference, per-vector-loop
    implementation."""
    candidate = np.array([1.0, 0.0])
    history = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
    reference = semantic_history_similarity(candidate, history)

    history_matrix = np.stack(history)
    fast_path = semantic_history_similarity_matrix(candidate, history_matrix)

    assert fast_path == pytest.approx(reference)


def test_semantic_history_similarity_matrix_empty_or_none_is_zero_zero():
    candidate = np.array([1.0, 0.0])
    assert semantic_history_similarity_matrix(candidate, None) == (0.0, 0.0)
    assert semantic_history_similarity_matrix(candidate, np.zeros((0, 2))) == (0.0, 0.0)


def test_semantic_history_similarity_max_and_mean():
    candidate = np.array([1.0, 0.0])
    history = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    # similarities = [1.0, 0.0] -> max=1.0, mean=0.5
    max_sim, mean_sim = semantic_history_similarity(candidate, history)
    assert max_sim == pytest.approx(1.0)
    assert mean_sim == pytest.approx(0.5)


def test_semantic_history_similarity_empty_history_is_zero_zero():
    assert semantic_history_similarity(np.array([1.0, 0.0]), []) == (0.0, 0.0)


def test_retrieval_rank_features_matches_hand_computed_ranks():
    candidate_ids = ["x", "y", "z"]
    bm25_scores = {"x": 5.0, "y": 10.0, "z": 1.0}  # best->worst: y, x, z
    semantic_scores = {"x": 0.9, "y": 0.1, "z": 0.5}  # best->worst: x, z, y
    bm25_ranks, semantic_ranks = retrieval_rank_features(candidate_ids, bm25_scores, semantic_scores)
    assert bm25_ranks == [2, 1, 3]
    assert semantic_ranks == [1, 3, 2]


def test_retrieval_rank_features_tied_scores_produce_no_duplicate_ranks():
    candidate_ids = ["a", "b", "c", "d"]
    tied = {c: 1.0 for c in candidate_ids}
    ranks_a, ranks_b = retrieval_rank_features(candidate_ids, tied, tied)
    for ranks in (ranks_a, ranks_b):
        assert sorted(ranks) == [1, 2, 3, 4]  # a permutation -- no duplicate rank values


# --------------------------------------------------------------------------
# Popularity features (behavioral) -- this is the leakage-critical one
# --------------------------------------------------------------------------


def test_build_frozen_train_click_counts_counts_clicks_only():
    train_interactions = pd.DataFrame(
        {
            "impression_id": [1, 1, 2, 2, 3],
            "article_id": ["A", "B", "A", "A", "C"],
            "clicked": [1, 0, 1, 1, 0],
        }
    )
    counts = build_frozen_train_click_counts(train_interactions)
    assert counts == {"A": 3}  # B and C never have clicked=1, so they're absent, not zero-valued


def test_expanding_train_popularity_future_click_does_not_affect_earlier_impression():
    """The exact scenario requested: impression 1 (earliest) must not be
    influenced by a click that happens in impression 2 or 3 (later in train)."""
    train_impressions = pd.DataFrame(
        {
            "impression_id": [1, 2, 3],
            "time": pd.to_datetime(["2023-01-01 10:00", "2023-01-01 11:00", "2023-01-01 12:00"]),
        }
    )
    train_interactions = pd.DataFrame(
        {
            "impression_id": [1, 2, 3, 3],
            "article_id": ["A", "A", "A", "B"],
            "clicked": [1, 1, 0, 1],  # A clicked at imp1 (10:00) and imp2 (11:00); B clicked at imp3 (12:00)
        }
    )
    popularity = ExpandingTrainPopularity(train_impressions, train_interactions)

    # Impression 1 (10:00): no click on A has happened yet (the imp1 click IS at 10:00, not before it)
    assert popularity.count_before("A", pd.Timestamp("2023-01-01 10:00")) == 0
    # Impression 2 (11:00): only the imp1 click (10:00) is strictly before -> 1.
    # Critically, this must NOT be influenced by anything at or after 11:00.
    assert popularity.count_before("A", pd.Timestamp("2023-01-01 11:00")) == 1
    # Impression 3 (12:00): imp1 (10:00) and imp2 (11:00) clicks both count -> 2.
    assert popularity.count_before("A", pd.Timestamp("2023-01-01 12:00")) == 2
    # B was never clicked before imp3's own time (its only click IS at imp3's time) -> 0.
    assert popularity.count_before("B", pd.Timestamp("2023-01-01 12:00")) == 0


def test_expanding_train_popularity_same_timestamp_ties_are_excluded():
    """Two impressions at the identical timestamp must not count towards each other."""
    train_impressions = pd.DataFrame(
        {"impression_id": [1, 2], "time": pd.to_datetime(["2023-01-01 10:00", "2023-01-01 10:00"])}
    )
    train_interactions = pd.DataFrame({"impression_id": [1], "article_id": ["A"], "clicked": [1]})
    popularity = ExpandingTrainPopularity(train_impressions, train_interactions)
    # impression 2's click on A is at the SAME instant as impression 1's -- not strictly before.
    assert popularity.count_before("A", pd.Timestamp("2023-01-01 10:00")) == 0


def test_expanding_train_popularity_unknown_article_is_zero():
    train_impressions = pd.DataFrame({"impression_id": [1], "time": pd.to_datetime(["2023-01-01 10:00"])})
    train_interactions = pd.DataFrame({"impression_id": [1], "article_id": ["A"], "clicked": [1]})
    popularity = ExpandingTrainPopularity(train_impressions, train_interactions)
    assert popularity.count_before("NEVER_CLICKED", pd.Timestamp("2023-01-02")) == 0


# --------------------------------------------------------------------------
# Recency-decay + dwell-time features (EB-NeRD only, Assignment 2 Q1.1/Q1.2)
# --------------------------------------------------------------------------


def test_load_ebnerd_history_detail_reads_raw_parquet_arrays(tmp_path):
    history_df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "article_id_fixed": [np.array([10, 11]), np.array([20])],
            "impression_time_fixed": [
                np.array(pd.to_datetime(["2023-01-01 10:00", "2023-01-02 10:00"])),
                np.array(pd.to_datetime(["2023-01-01 08:00"])),
            ],
            "read_time_fixed": [np.array([12.5, np.nan]), np.array([30.0])],
        }
    )
    path = tmp_path / "history.parquet"
    history_df.to_parquet(path)

    detail = load_ebnerd_history_detail(path)

    # STRING keys, not the raw int64 user_id -- this is the actual bug this
    # test exists to catch. It slipped through once already: the original
    # version of this test compared detail.keys() against int literals
    # ({1, 2}), which a Python `==` happily treats as passing regardless of
    # whether the keys are int 1 or str "1" is NOT true (1 == "1" is False)
    # -- so that version WOULD have failed... except every key in this
    # fixture happens to be a single digit, and the bug was only found by
    # testing against real EB-NeRD data, where
    # news_reco.data.ebnerd_adapter's OWN common-schema impressions table
    # stores user_id as `str` (`.astype(str)`), while this function
    # originally left it as the raw parquet int -- every
    # history_detail_by_user.get(impression.user_id) lookup silently missed,
    # computing 0.0 everywhere with no crash. Asserting str keys explicitly
    # here, against the exact type real code will look them up with, is
    # what actually would have caught it.
    assert set(detail.keys()) == {"1", "2"}
    assert all(isinstance(k, str) for k in detail.keys())
    assert len(detail["1"]) == 2
    # article_id cast to str, NaN read_time preserved as NaN, not coerced to 0.0
    assert detail["1"][0] == ("10", pd.Timestamp("2023-01-01 10:00"), 12.5)
    assert detail["1"][1][0] == "11" and detail["1"][1][1] == pd.Timestamp("2023-01-02 10:00")
    assert np.isnan(detail["1"][1][2])
    assert detail["2"] == [("20", pd.Timestamp("2023-01-01 08:00"), 30.0)]


def test_ages_hours_strictly_before_excludes_ties_and_future_and_nan():
    impression_time = pd.Timestamp("2023-01-02 12:00")
    history_detail = [
        ("A", pd.Timestamp("2023-01-01 12:00"), 10.0),  # exactly 24h before -> kept, age=24.0
        ("B", pd.Timestamp("2023-01-02 12:00"), 5.0),  # exactly AT impression_time -> excluded (tie)
        ("C", pd.Timestamp("2023-01-03 12:00"), 7.0),  # AFTER impression_time -> excluded
        ("D", pd.NaT, 1.0),  # NaT click_time -> excluded
    ]
    kept = _ages_hours_strictly_before(history_detail, impression_time)
    assert len(kept) == 1
    article_id, age_hours, read_time = kept[0]
    assert article_id == "A"
    assert age_hours == pytest.approx(24.0)
    assert read_time == 10.0


def test_batch_recency_weighted_semantic_similarity_hand_computed():
    # Two candidates, two history articles with different ages -> different
    # decay weights at half_life=24h: weight = 0.5**(age/24).
    candidate_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])  # candidate 0 matches history 0; candidate 1 matches history 1
    history_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])  # history 0 (age=0h, weight=1.0), history 1 (age=24h, weight=0.5)
    history_weights = np.array([1.0, 0.5])

    result = batch_recency_weighted_semantic_similarity(candidate_matrix, history_matrix, history_weights)

    # candidate 0: sims=[1.0, 0.0] -> weighted_sum = 1.0*1.0 + 0.0*0.5 = 1.0 -> /1.5 = 0.6667
    # candidate 1: sims=[0.0, 1.0] -> weighted_sum = 0.0*1.0 + 1.0*0.5 = 0.5 -> /1.5 = 0.3333
    assert result[0] == pytest.approx(1.0 / 1.5)
    assert result[1] == pytest.approx(0.5 / 1.5)


def test_batch_recency_weighted_semantic_similarity_empty_history_is_zero():
    candidate_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert list(batch_recency_weighted_semantic_similarity(candidate_matrix, None, None)) == [0.0, 0.0]
    result = batch_recency_weighted_semantic_similarity(candidate_matrix, np.zeros((0, 2)), np.zeros(0))
    assert list(result) == [0.0, 0.0]


def test_mean_history_read_time_drops_nan_not_zero():
    # 10.0 and 30.0 average to 20.0; the NaN entry must be DROPPED, not
    # averaged in as 0.0 (which would give (10+30+0)/3 = 13.33, wrong).
    history_detail_before = [("A", 5.0, 10.0), ("B", 3.0, np.nan), ("C", 1.0, 30.0)]
    assert mean_history_read_time(history_detail_before) == pytest.approx(20.0)


def test_mean_history_read_time_empty_is_zero_not_nan():
    assert mean_history_read_time([]) == 0.0


# --------------------------------------------------------------------------
# Integration: build_impression_rows / build_feature_table with a tiny,
# fully hand-computable catalog + retrieval indexes.
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_catalog():
    articles = pd.DataFrame(
        {
            "article_id": ["1", "2", "3"],
            "category": ["sport", "news", "sport"],
            "subcategory": ["football", "weather", "football"],
            "title": ["Football match today", "Weather forecast rain", "Football transfer news"],
            "abstract": ["", "", ""],
            "text": ["Football match today", "Weather forecast rain", "Football transfer news"],
            "published_time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-05"]),
        }
    ).set_index("article_id", drop=False)

    bm25 = BM25Index()
    bm25.fit(articles["article_id"].tolist(), articles["text"].tolist())

    # Deliberately simple, not unit-normalized -- this test checks that
    # build_impression_rows correctly threads EmbeddingIndex's own scores
    # through, not EmbeddingIndex's cosine-similarity math (already covered
    # by tests/test_embeddings.py).
    embeddings = {"1": np.array([1.0, 0.0]), "2": np.array([0.0, 1.0]), "3": np.array([0.5, 0.5])}
    semantic = EmbeddingIndex()
    semantic.fit(list(embeddings.keys()), np.stack(list(embeddings.values())))

    zero_vector = np.zeros(2)

    def get_article_embedding(article_id: str) -> np.ndarray:
        return embeddings.get(article_id, zero_vector)

    return {
        "articles": articles,
        "article_lookup": ArticleLookup.from_dataframe(articles),
        "bm25": bm25,
        "semantic": semantic,
        "get_article_embedding": get_article_embedding,
    }


def test_build_impression_rows_hand_computed_values(tiny_catalog):
    availability_gate = AvailabilityGate(tiny_catalog["articles"])
    impression_time = pd.Timestamp("2023-01-03")  # after "1","2" published, before "3" published

    rows, n_violations = build_impression_rows(
        impression_id=100,
        candidate_ids=["1", "3"],
        history=["2"],
        impression_time=impression_time,
        article_lookup=tiny_catalog["article_lookup"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        bm25_query="Weather forecast rain",  # = title of history article "2"
        semantic_query_vector=np.array([0.0, 1.0]),  # mean-pooled history = embedding("2")
        popularity_lookup=lambda aid, t: {"1": 5, "3": 2}.get(aid, 0),
        availability_gate=availability_gate,
        gate_freshness=False,
        clicked_by_article=None,
    )

    assert [r["article_id"] for r in rows] == ["1", "3"]  # candidate order preserved exactly

    row1, row3 = rows

    # BM25: query tokens {weather, forecast, rain} share nothing with either
    # candidate's title tokens -> both score 0.0, and scores_to_ranks' stable
    # tie-break keeps candidate_ids' own order -> "1" gets rank 1, "3" rank 2.
    assert row1["bm25_score"] == 0.0 and row3["bm25_score"] == 0.0
    assert row1["bm25_rank_within_impression"] == 1
    assert row3["bm25_rank_within_impression"] == 2

    # Semantic: query=[0,1]; "1"=[1,0]->dot=0.0; "3"=[0.5,0.5]->dot=0.5 -> "3" ranks first.
    assert row1["semantic_score"] == pytest.approx(0.0)
    assert row3["semantic_score"] == pytest.approx(0.5)
    assert row1["semantic_rank_within_impression"] == 2
    assert row3["semantic_rank_within_impression"] == 1

    # Content features: neither candidate's title/category/subcategory overlaps
    # history article "2" (weather/news) at all.
    assert row1["title_token_overlap_with_history"] == 0.0
    assert row1["category_match_with_history"] == 0.0
    assert row1["subcategory_match_with_history"] == 0.0

    # Semantic-history: "1"=[1,0] vs history emb [0,1] -> 0.0/0.0; "3"=[0.5,0.5] -> 0.5/0.5.
    assert row1["max_cosine_similarity_to_any_history_article"] == pytest.approx(0.0)
    assert row3["max_cosine_similarity_to_any_history_article"] == pytest.approx(0.5)
    assert row3["mean_cosine_similarity_to_history_articles"] == pytest.approx(0.5)

    assert row1["history_length"] == 1 and row3["history_length"] == 1
    assert row1["article_training_click_count"] == 5
    assert row3["article_training_click_count"] == 2

    # Temporal: "1" published 2023-01-01, impression at 2023-01-03 -> 48.0 hours, never negative.
    assert row1["article_age_hours"] == pytest.approx(48.0)
    assert row1["article_age_hours"] >= 0

    # "3" published 2023-01-05, AFTER the 2023-01-03 impression -> genuine
    # temporal-availability violation. Ungated: kept, age is NaN (not
    # clipped to 0), and counted as a violation.
    assert np.isnan(row3["article_age_hours"])
    assert n_violations == 1

    # Position bias: 1-indexed position in candidate_ids' own order, "1"
    # first (position 1), "3" second (position 2) -- independent of score.
    assert row1["candidate_position_in_impression"] == 1
    assert row3["candidate_position_in_impression"] == 2

    # history_detail was not passed (None) -- recency/dwell columns must be
    # omitted entirely, not fabricated as 0.0/NaN.
    assert RECENCY_FEATURE_NAME not in row1
    assert DWELL_FEATURE_NAME not in row1


def test_build_impression_rows_recency_and_dwell_hand_computed(tiny_catalog):
    """Same fixture as the hand-computed test above, but with history_detail
    supplied -- verifies the recency-weighted-similarity and dwell-time
    columns end to end, against values worked out by hand."""
    impression_time = pd.Timestamp("2023-01-03 00:00")
    # History article "2" (embedding [0,1]) clicked 24h before the
    # impression, with a recorded read_time of 15.0 seconds.
    history_detail = [("2", pd.Timestamp("2023-01-02 00:00"), 15.0)]

    rows, _ = build_impression_rows(
        impression_id=100,
        candidate_ids=["1", "3"],
        history=["2"],
        impression_time=impression_time,
        article_lookup=tiny_catalog["article_lookup"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        bm25_query="Weather forecast rain",
        semantic_query_vector=np.array([0.0, 1.0]),
        popularity_lookup=lambda aid, t: 0,
        availability_gate=None,  # not needed for this test
        gate_freshness=False,
        clicked_by_article=None,
        history_detail=history_detail,
        recency_half_life_hours=24.0,
    )
    row1, row3 = rows

    # age = 24h = exactly one half-life -> weight = 0.5**(24/24) = 0.5.
    # Only one history article, so the weighted average IS that one
    # similarity value regardless of the weight's magnitude (weight cancels
    # out: weighted_sum / total_weight = (sim*0.5)/0.5 = sim).
    # "1"=[1,0] . "2"=[0,1] = 0.0; "3"=[0.5,0.5] . [0,1] = 0.5.
    assert row1[RECENCY_FEATURE_NAME] == pytest.approx(0.0)
    assert row3[RECENCY_FEATURE_NAME] == pytest.approx(0.5)

    # Dwell time is a per-IMPRESSION value (not per-candidate) -- both rows
    # get the same mean history read_time, 15.0 (only one entry).
    assert row1[DWELL_FEATURE_NAME] == pytest.approx(15.0)
    assert row3[DWELL_FEATURE_NAME] == pytest.approx(15.0)


def test_build_impression_rows_gate_freshness_drops_unpublished_candidate(tiny_catalog):
    availability_gate = AvailabilityGate(tiny_catalog["articles"])
    impression_time = pd.Timestamp("2023-01-03")

    rows, n_violations = build_impression_rows(
        impression_id=100,
        candidate_ids=["1", "3"],
        history=["2"],
        impression_time=impression_time,
        article_lookup=tiny_catalog["article_lookup"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        bm25_query="Weather forecast rain",
        semantic_query_vector=np.array([0.0, 1.0]),
        popularity_lookup=lambda aid, t: 0,
        availability_gate=availability_gate,
        gate_freshness=True,  # explicitly gated -- "3" must never reach the reranker
        clicked_by_article=None,
    )

    assert [r["article_id"] for r in rows] == ["1"]  # "3" dropped entirely, not NaN'd
    assert n_violations == 0  # a dropped candidate is not counted as a "kept with NaN" violation


def test_build_impression_rows_cold_start_defaults_to_zero_not_nan(tiny_catalog):
    rows, n_violations = build_impression_rows(
        impression_id=200,
        candidate_ids=["1", "3"],
        history=[],  # cold start
        impression_time=pd.Timestamp("2023-01-03"),
        article_lookup=tiny_catalog["article_lookup"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        bm25_query="",  # build_query_from_history([], ...) == ""
        semantic_query_vector=None,  # build_query_embedding([], ...) is None
        popularity_lookup=lambda aid, t: 0,
        availability_gate=None,  # MIND-like: no age feature at all
        gate_freshness=False,
        clicked_by_article=None,
    )
    for row in rows:
        assert row["bm25_score"] == 0.0
        assert row["semantic_score"] == 0.0
        assert row["title_token_overlap_with_history"] == 0.0
        assert row["category_match_with_history"] == 0.0
        assert row["max_cosine_similarity_to_any_history_article"] == 0.0
        assert row["mean_cosine_similarity_to_history_articles"] == 0.0
        assert row["history_length"] == 0
        assert TEMPORAL_FEATURE_NAME not in row  # MIND has no publish times -- feature omitted, not faked


def test_build_impression_rows_missing_article_handled_without_crashing(tiny_catalog):
    """A candidate absent from the catalog/embedding table (shouldn't happen
    normally, but must not crash) gets safe, documented defaults."""
    availability_gate = AvailabilityGate(tiny_catalog["articles"])
    rows, n_violations = build_impression_rows(
        impression_id=300,
        candidate_ids=["1", "UNKNOWN_ID"],
        history=["2"],
        impression_time=pd.Timestamp("2023-01-03"),
        article_lookup=tiny_catalog["article_lookup"],
        get_article_embedding=tiny_catalog["get_article_embedding"],  # falls back to zero-vector
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        bm25_query="Weather forecast rain",
        semantic_query_vector=np.array([0.0, 1.0]),
        popularity_lookup=lambda aid, t: 0,
        availability_gate=availability_gate,
        gate_freshness=False,
        clicked_by_article=None,
    )
    unknown_row = rows[1]
    assert unknown_row["article_id"] == "UNKNOWN_ID"
    assert unknown_row["semantic_score"] == 0.0  # zero-vector embedding -> dot product 0.0
    assert unknown_row["title_token_overlap_with_history"] == 0.0  # missing title -> "" -> no tokens
    assert unknown_row["category_match_with_history"] == 0.0  # missing category -> no label
    assert np.isnan(unknown_row["article_age_hours"])  # unknown article can't be confirmed published


# --------------------------------------------------------------------------
# build_feature_table: grouping, column names/shape, candidate-order, IDs
# --------------------------------------------------------------------------


def test_build_feature_table_shape_and_names(tiny_catalog):
    impressions = pd.DataFrame(
        {
            "impression_id": [1],
            "time": pd.to_datetime(["2023-01-03"]),
            "history": [["2"]],
        }
    )
    table, _ = build_feature_table(
        impressions,
        {1: ["1", "3"]},
        articles=tiny_catalog["articles"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        get_article_title=lambda aid: tiny_catalog["articles"].loc[aid, "title"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        popularity_lookup=lambda aid, t: 0,
        availability_gate=AvailabilityGate(tiny_catalog["articles"]),
    )
    expected_columns = (
        ["impression_id", "article_id"] + COMMON_FEATURE_NAMES + [TEMPORAL_FEATURE_NAME]
    )
    assert list(table.columns) == expected_columns
    assert len(table) == 2


def test_build_feature_table_impression_grouping_matches_candidate_counts(tiny_catalog):
    impressions = pd.DataFrame(
        {
            "impression_id": [10, 20],
            "time": pd.to_datetime(["2023-01-03", "2023-01-03"]),
            "history": [["2"], []],
        }
    )
    candidates_by_impression = {10: ["1", "3"], 20: ["1", "2", "3"]}
    table, _ = build_feature_table(
        impressions,
        candidates_by_impression,
        articles=tiny_catalog["articles"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        get_article_title=lambda aid: tiny_catalog["articles"].loc[aid, "title"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        popularity_lookup=lambda aid, t: 0,
    )
    group_sizes = table.groupby("impression_id", sort=False).size()
    # groupby(sort=False) preserves the table's own row order, which is what
    # LGBMRanker's `group` argument requires to line up with the feature matrix.
    assert list(group_sizes.index) == [10, 20]
    assert list(group_sizes.values) == [2, 3]


def test_build_feature_table_history_detail_missing_user_gets_zero_not_crash(tiny_catalog):
    """The exact bug caught before it shipped: history_detail_by_user
    provided for the TABLE, but one specific impression's user_id has no
    entry in it. Every row must still get RECENCY_FEATURE_NAME/
    DWELL_FEATURE_NAME (0.0, via the empty-history convention), not raise a
    KeyError from a fixed column list some rows don't have."""
    # user_id is a STRING here, matching news_reco.data.ebnerd_adapter's real
    # common-schema convention (`.astype(str)`) -- not an int, which is
    # exactly the type mismatch that let the real bug through undetected
    # (see test_load_ebnerd_history_detail_reads_raw_parquet_arrays).
    impressions = pd.DataFrame(
        {
            "impression_id": [1],
            "user_id": ["999"],  # NOT a key in history_detail_by_user below
            "time": pd.to_datetime(["2023-01-03"]),
            "history": [["2"]],
        }
    )
    table, _ = build_feature_table(
        impressions,
        {1: ["1", "3"]},
        articles=tiny_catalog["articles"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        get_article_title=lambda aid: tiny_catalog["articles"].loc[aid, "title"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        popularity_lookup=lambda aid, t: 0,
        history_detail_by_user={"1": [("2", pd.Timestamp("2023-01-01"), 10.0)]},  # keyed by user_id="1", not "999"
    )
    assert RECENCY_FEATURE_NAME in table.columns
    assert DWELL_FEATURE_NAME in table.columns
    assert (table[RECENCY_FEATURE_NAME] == 0.0).all()
    assert (table[DWELL_FEATURE_NAME] == 0.0).all()


def test_build_feature_table_history_detail_present_user_computes_correctly(tiny_catalog):
    impressions = pd.DataFrame(
        {
            "impression_id": [1],
            "user_id": ["1"],  # string, matching the real common-schema convention
            "time": pd.to_datetime(["2023-01-03"]),
            "history": [["2"]],
        }
    )
    table, _ = build_feature_table(
        impressions,
        {1: ["1", "3"]},
        articles=tiny_catalog["articles"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        get_article_title=lambda aid: tiny_catalog["articles"].loc[aid, "title"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        popularity_lookup=lambda aid, t: 0,
        history_detail_by_user={"1": [("2", pd.Timestamp("2023-01-02"), 20.0)]},
    )
    # Same math as test_build_impression_rows_recency_and_dwell_hand_computed:
    # one history article "2" ([0,1] embedding), read_time=20.0.
    assert table.set_index("article_id")[DWELL_FEATURE_NAME].to_dict() == {"1": 20.0, "3": 20.0}


def test_build_feature_table_candidate_order_preserved_within_impression(tiny_catalog):
    impressions = pd.DataFrame(
        {"impression_id": [1], "time": pd.to_datetime(["2023-01-03"]), "history": [[]]}
    )
    # Deliberately NOT catalog order ("3" before "1") -- output must match this order exactly.
    table, _ = build_feature_table(
        impressions,
        {1: ["3", "1"]},
        articles=tiny_catalog["articles"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        get_article_title=lambda aid: tiny_catalog["articles"].loc[aid, "title"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        popularity_lookup=lambda aid, t: 0,
    )
    assert list(table["article_id"]) == ["3", "1"]


def test_build_feature_table_output_ids_match_input_candidates_ungated(tiny_catalog):
    impressions = pd.DataFrame(
        {"impression_id": [1], "time": pd.to_datetime(["2023-01-03"]), "history": [["2"]]}
    )
    table, _ = build_feature_table(
        impressions,
        {1: ["1", "2", "3"]},
        articles=tiny_catalog["articles"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        get_article_title=lambda aid: tiny_catalog["articles"].loc[aid, "title"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        popularity_lookup=lambda aid, t: 0,
        availability_gate=AvailabilityGate(tiny_catalog["articles"]),
        gate_freshness=False,
    )
    assert set(table["article_id"]) == {"1", "2", "3"}  # exactly the input candidates, ungated
    assert table["article_id"].is_unique  # no duplicate rows/ranks for one impression


def test_build_feature_table_with_labels_emits_clicked_column(tiny_catalog):
    impressions = pd.DataFrame(
        {"impression_id": [1], "time": pd.to_datetime(["2023-01-03"]), "history": [[]]}
    )
    table, _ = build_feature_table(
        impressions,
        {1: ["1", "3"]},
        articles=tiny_catalog["articles"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        get_article_title=lambda aid: tiny_catalog["articles"].loc[aid, "title"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        popularity_lookup=lambda aid, t: 0,
        clicked_lookup={1: {"1": 0, "3": 1}},
    )
    assert "clicked" in table.columns
    assert table.set_index("article_id")["clicked"].to_dict() == {"1": 0, "3": 1}


def test_build_feature_table_unlabeled_has_no_clicked_column(tiny_catalog):
    """The official Codabench test files have no click column at all --
    build_feature_table must not fabricate one when clicked_lookup is None."""
    impressions = pd.DataFrame(
        {"impression_id": [1], "time": pd.to_datetime(["2023-01-03"]), "history": [[]]}
    )
    table, _ = build_feature_table(
        impressions,
        {1: ["1", "3"]},
        articles=tiny_catalog["articles"],
        get_article_embedding=tiny_catalog["get_article_embedding"],
        get_article_title=lambda aid: tiny_catalog["articles"].loc[aid, "title"],
        bm25_index=tiny_catalog["bm25"],
        semantic_index=tiny_catalog["semantic"],
        popularity_lookup=lambda aid, t: 0,
        clicked_lookup=None,
    )
    assert "clicked" not in table.columns


# --------------------------------------------------------------------------
# Guardrail: Component 1's submission scripts must remain byte-for-byte
# untouched by this work (the assignment's explicit constraint).
# --------------------------------------------------------------------------


def _strip_project_root_line(content: bytes) -> bytes:
    """Remove any line assigning PROJECT_ROOT via Path(__file__).resolve()....

    The ONE disclosed, functionally-necessary exception to "byte-identical":
    scripts/generate_mind_submission.py and scripts/generate_ebnerd_submission.py
    compute PROJECT_ROOT relative to their OWN file location, and moving them
    one level deeper (scripts/ -> scripts/submission/, during the Assignment 2
    project-structure reorganization) genuinely requires one more `.parent`
    call for that computation to still resolve to the same absolute path --
    not a behavior change, just keeping the same behavior correct after the
    move. Everything else in these files, and every other protected file, is
    still required to be 100% byte-identical.
    """
    lines = content.split(b"\n")
    kept = [line for line in lines if b"PROJECT_ROOT" not in line or b"Path(__file__)" not in line]
    return b"\n".join(kept)


@pytest.mark.parametrize(
    "head_path,current_path",
    [
        # (path as last committed at HEAD, path in the current working tree)
        # -- these differ for the three scripts moved into scripts/submission/
        # and scripts/evaluation/ during the project-structure reorganization
        # (Assignment 2 audit process). Comparing CONTENT directly via `git
        # show HEAD:<path>` (rather than `git diff HEAD -- <path>`) is
        # deliberate: `git diff` compares by PATH, so an uncommitted rename
        # shows as a full delete-at-old-path + add-at-new-path -- a false
        # positive here even when the content is byte-identical. Content
        # comparison sidesteps that entirely and stays correct through any
        # future reorganization too.
        # The scripts/ reorganization is now itself committed (HEAD already
        # reflects the new paths), so head_path == current_path for these
        # three -- the head_path/current_path split stays in the function
        # signature/design (not collapsed back to a single path) because
        # it's still the correct tool for the NEXT time a protected file
        # moves, not because it's doing anything for these three right now.
        ("scripts/submission/generate_mind_submission.py", "scripts/submission/generate_mind_submission.py"),
        ("scripts/submission/generate_ebnerd_submission.py", "scripts/submission/generate_ebnerd_submission.py"),
        ("scripts/evaluation/evaluate_full.py", "scripts/evaluation/evaluate_full.py"),
        ("src/news_reco/retrieval/lexical/bm25.py", "src/news_reco/retrieval/lexical/bm25.py"),
        ("src/news_reco/retrieval/semantic/embeddings.py", "src/news_reco/retrieval/semantic/embeddings.py"),
        ("src/news_reco/features/store.py", "src/news_reco/features/store.py"),
    ],
)
def test_component1_files_are_unmodified_relative_to_git_head(head_path, current_path):
    # core.autocrlf=true on this repo (confirmed) means `git show` (raw blob,
    # always LF) and the working-tree file (checked out with CRLF) will
    # legitimately differ in line endings even when content is otherwise
    # identical -- normalize CRLF->LF on both sides before comparing so this
    # test catches real content changes, not a checkout artifact.
    head_content = subprocess.run(
        ["git", "show", f"HEAD:{head_path}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    ).stdout.replace(b"\r\n", b"\n")
    current_content = (PROJECT_ROOT / current_path).read_bytes().replace(b"\r\n", b"\n")
    assert _strip_project_root_line(current_content) == _strip_project_root_line(head_content), (
        f"{current_path} (HEAD: {head_path}) has content differences relative to the last commit -- "
        "Component 1 files must remain untouched (moving/renaming is fine, and the PROJECT_ROOT path-depth "
        "line may differ if the file moved to a different nesting depth; no other line may change)."
    )
