"""Unit tests for temporal splitting and leakage checks (Milestone 3).

Uses small, hand-crafted DataFrames with known-correct expected outputs,
including deliberately-broken cases -- a leakage check that never fires on
broken input isn't proven to work.

Run with: pytest tests/test_temporal_split.py -v
"""

import pandas as pd
import pytest

from news_reco.splitting.temporal_split import (
    assign_split,
    check_no_future_click_leakage,
    check_split_boundaries,
)

VAL_START = pd.Timestamp("2019-11-14 00:00:00")
TEST_START = pd.Timestamp("2019-11-15 00:00:00")


def test_assign_split_handles_boundary_timestamps_correctly():
    impressions = pd.DataFrame(
        {
            "impression_id": [1, 2, 3, 4],
            "time": pd.to_datetime(
                [
                    "2019-11-13 23:59:59",  # just before val_start -> train
                    "2019-11-14 00:00:00",  # exactly val_start -> val
                    "2019-11-14 23:59:59",  # just before test_start -> val
                    "2019-11-15 00:00:00",  # exactly test_start -> test
                ]
            ),
        }
    )

    split = assign_split(impressions, VAL_START, TEST_START)

    assert list(split) == ["train", "val", "val", "test"]


def test_assign_split_rejects_inverted_cutoffs():
    impressions = pd.DataFrame({"impression_id": [1], "time": pd.to_datetime(["2019-11-14"])})
    with pytest.raises(ValueError):
        assign_split(impressions, val_start=TEST_START, test_start=VAL_START)


def test_check_split_boundaries_passes_for_a_correct_split():
    impressions = pd.DataFrame(
        {
            "impression_id": [1, 2, 3],
            "time": pd.to_datetime(["2019-11-13", "2019-11-14", "2019-11-15"]),
        }
    )
    split = assign_split(impressions, VAL_START, TEST_START)
    check_split_boundaries(impressions, split, VAL_START, TEST_START)  # should not raise


def test_check_split_boundaries_catches_a_mislabeled_impression():
    impressions = pd.DataFrame(
        {
            "impression_id": [1, 2],
            "time": pd.to_datetime(["2019-11-13", "2019-11-15"]),
        }
    )
    # Deliberately wrong: impression 2 (11-15, should be "test") mislabeled "train".
    broken_split = pd.Series(["train", "train"], index=impressions.index)

    with pytest.raises(AssertionError):
        check_split_boundaries(impressions, broken_split, VAL_START, TEST_START)


def test_check_no_future_click_leakage_passes_clean_history():
    impressions = pd.DataFrame(
        {
            "impression_id": [1, 2, 3],
            "user_id": ["U1", "U1", "U1"],
            "time": pd.to_datetime(
                ["2019-11-10 10:00:00", "2019-11-11 10:00:00", "2019-11-12 10:00:00"]
            ),
            "history": [[], ["N1"], ["N2", "N9"]],
            # impression 2 claims N1 was clicked before -> true (impression 1 clicked N1)
            # impression 3 claims N2 (clicked in impression 2, true) and N9 (never
            # observed as a click anywhere -> unverifiable, not a violation)
        }
    )
    interactions = pd.DataFrame(
        {
            "impression_id": [1, 1, 2, 2],
            "article_id": ["N1", "N5", "N2", "N6"],
            "clicked": [1, 0, 1, 0],
        }
    )

    violations = check_no_future_click_leakage(impressions, interactions)

    assert len(violations) == 0


def test_check_no_future_click_leakage_catches_a_genuine_violation():
    # Impression 20 (09:00) claims N100 was already clicked -- but our logs
    # show N100 was only clicked later, in impression 21 (10:00). That's a
    # real contradiction: history says "before", the log says "after".
    impressions = pd.DataFrame(
        {
            "impression_id": [20, 21],
            "user_id": ["U3", "U3"],
            "time": pd.to_datetime(["2019-11-10 09:00:00", "2019-11-10 10:00:00"]),
            "history": [["N100"], []],
        }
    )
    interactions = pd.DataFrame(
        {
            "impression_id": [20, 21],
            "article_id": ["N50", "N100"],
            "clicked": [0, 1],
        }
    )

    violations = check_no_future_click_leakage(impressions, interactions)

    assert len(violations) == 1
    assert violations.iloc[0]["article_id"] == "N100"
    assert violations.iloc[0]["impression_id"] == 20
