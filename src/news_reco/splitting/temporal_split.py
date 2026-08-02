"""Assign temporal train/val/test splits and check for leakage.

A temporal split assigns each impression to train, val, or test based on
*when* it happened, not randomly. This matters because a recommender is
always deployed into the future: at serving time it has only ever seen the
past, so evaluating it on data that occurred randomly interspersed with its
training data would let it "see the future" in a way that could never
happen in production. Splitting by time -- train on the earliest period,
validate on the next, test on the latest -- is the only split that mirrors
how the system will actually be used.
"""

from __future__ import annotations

import pandas as pd


def assign_split(impressions: pd.DataFrame, val_start: pd.Timestamp, test_start: pd.Timestamp) -> pd.Series:
    """Label each impression "train", "val", or "test" by its timestamp.

    Args:
        impressions: Must have a "time" column (datetime64).
        val_start: First timestamp belonging to validation (train is everything before this).
        test_start: First timestamp belonging to test (val is [val_start, test_start)).

    Returns:
        A Series of "train"/"val"/"test" labels, aligned to impressions' index.
    """
    if not (val_start < test_start):
        raise ValueError(f"val_start ({val_start}) must be before test_start ({test_start})")

    split = pd.Series("train", index=impressions.index, dtype="object")
    split[impressions["time"] >= val_start] = "val"
    split[impressions["time"] >= test_start] = "test"
    return split


def check_split_boundaries(
    impressions: pd.DataFrame, split: pd.Series, val_start: pd.Timestamp, test_start: pd.Timestamp
) -> None:
    """Raise AssertionError if any impression's split label disagrees with its timestamp.

    This exists to catch a bug in assign_split (or in how its output was
    later joined or reordered) before it silently corrupts an evaluation --
    it's a check on the *output*, independent of how that output was produced.
    """
    times = impressions["time"]
    assert (times[split == "train"] < val_start).all(), "some 'train' impression is at or after val_start"
    assert (
        (times[split == "val"] >= val_start) & (times[split == "val"] < test_start)
    ).all(), "some 'val' impression falls outside [val_start, test_start)"
    assert (times[split == "test"] >= test_start).all(), "some 'test' impression is before test_start"


def check_no_future_click_leakage(impressions: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """Verify every article in an impression's history was actually clicked earlier.

    Each impression carries a `history` list -- articles the user is claimed
    to have clicked before this impression. This independently verifies that
    claim against the actual click log instead of trusting it: for every
    (user, history article) pair, the article's first observed click must be
    strictly before the impression's own time.

    History entries whose article was never observed as a click anywhere in
    our logs are NOT violations -- MIND's history can extend before the
    start of the logged window (a user's "real" history is longer than what
    we can see), so "unverifiable" and "leaked" are reported separately.

    A first_click_time exactly equal to the impression time is also not
    counted: MIND timestamps only have 1-second resolution, so a tie is
    consistent with the click having happened at or before that second, not
    demonstrably after it.

    Returns:
        A DataFrame of genuine violations (empty if none found), with columns
        impression_id, user_id, article_id, impression_time, first_click_time.
    """
    clicks = interactions.loc[interactions["clicked"] == 1, ["impression_id", "article_id"]].merge(
        impressions[["impression_id", "user_id", "time"]], on="impression_id"
    )
    first_click_time = (
        clicks.groupby(["user_id", "article_id"])["time"].min().rename("first_click_time").reset_index()
    )

    history = impressions[["impression_id", "user_id", "time", "history"]].explode("history")
    history = history.rename(columns={"time": "impression_time", "history": "article_id"})
    history = history.dropna(subset=["article_id"])  # impressions with empty history produce a NaN row

    checked = history.merge(first_click_time, on=["user_id", "article_id"], how="left")

    unverifiable = checked["first_click_time"].isna().sum()
    violations = checked[checked["first_click_time"] > checked["impression_time"]]

    print(f"History entries checked: {len(checked):,}")
    print(f"  unverifiable (predates our observation window): {unverifiable:,}")
    print(f"  genuine future-click violations: {len(violations):,}")

    return violations.reset_index(drop=True)
