"""Unit tests for head/tail slicing (Milestone 8).

Run with: pytest tests/test_slicing.py -v
"""

from news_reco.evaluation.slicing import classify_head_tail


def test_classify_head_when_popularity_meets_threshold():
    counts = {"A": 100}
    assert classify_head_tail({"A"}, counts, threshold=50) == "head"


def test_classify_tail_when_below_threshold():
    counts = {"A": 10}
    assert classify_head_tail({"A"}, counts, threshold=50) == "tail"


def test_classify_uses_max_popularity_across_multiple_clicks():
    counts = {"A": 5, "B": 100}
    # Impression clicked both A (tail-ish) and B (head) -- one popular hit
    # is enough to classify the impression as head.
    assert classify_head_tail({"A", "B"}, counts, threshold=50) == "head"


def test_classify_article_absent_from_train_counts_as_zero_popularity():
    counts = {"A": 100}
    assert classify_head_tail({"never_in_train"}, counts, threshold=50) == "tail"


def test_classify_exactly_at_threshold_is_head():
    counts = {"A": 50}
    assert classify_head_tail({"A"}, counts, threshold=50) == "head"
