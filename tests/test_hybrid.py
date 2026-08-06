"""Unit tests for Reciprocal Rank Fusion (Milestone 8).

Run with: pytest tests/test_hybrid.py -v
"""

import pytest

from news_reco.retrieval.hybrid import fuse_score_dicts, reciprocal_rank_fusion


def test_doc_ranked_first_in_both_lists_wins():
    list_a = ["x", "y", "z"]
    list_b = ["x", "z", "y"]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert fused[0][0] == "x"


def test_consensus_across_lists_beats_a_single_top_rank():
    # "consensus" is rank 2 in both lists: score = 2 * 1/(60+2) = 2/62.
    # "solo_winner" is rank 1 in only one list: score = 1/(60+1) = 1/61.
    # 2/62 (~0.03226) > 1/61 (~0.01639), so consensus should win despite
    # never being ranked first anywhere -- this is the actual point of
    # rank fusion, not just an implementation detail.
    list_a = ["solo_winner", "consensus", "other_a"]
    list_b = ["other_b", "consensus", "other_c"]
    fused = dict(reciprocal_rank_fusion([list_a, list_b]))
    assert fused["consensus"] > fused["solo_winner"]


def test_fused_scores_match_hand_computed_formula():
    list_a = ["x"]
    list_b = ["x"]
    fused = dict(reciprocal_rank_fusion([list_a, list_b], k=60))
    expected = 1 / (60 + 1) + 1 / (60 + 1)
    assert fused["x"] == pytest.approx(expected)


def test_doc_in_only_one_list_is_still_included():
    fused = dict(reciprocal_rank_fusion([["x", "y"], ["z"]]))
    assert set(fused.keys()) == {"x", "y", "z"}


def test_empty_lists_produce_empty_fusion():
    assert reciprocal_rank_fusion([[], []]) == []


def test_fuse_score_dicts_matches_manual_rank_and_fuse():
    bm25_scores = {"x": 5.0, "y": 3.0, "z": 1.0}
    semantic_scores = {"x": 0.2, "y": 0.9, "z": 0.5}

    fused = fuse_score_dicts([bm25_scores, semantic_scores])

    # Manually rank each dict best-first, then fuse the same way.
    bm25_ranked = ["x", "y", "z"]  # 5.0 > 3.0 > 1.0
    semantic_ranked = ["y", "z", "x"]  # 0.9 > 0.5 > 0.2
    expected = dict(reciprocal_rank_fusion([bm25_ranked, semantic_ranked]))

    assert fused == expected


def test_fuse_score_dicts_agreement_boosts_a_candidate():
    # "y" ranks 2nd in bm25 but 1st in semantic; "x" ranks 1st in bm25 but
    # last in semantic. Agreement (y) should beat a single-method top pick (x).
    bm25_scores = {"x": 5.0, "y": 3.0, "z": 1.0}
    semantic_scores = {"y": 0.9, "z": 0.5, "x": 0.2}
    fused = fuse_score_dicts([bm25_scores, semantic_scores])
    assert fused["y"] > fused["x"]
