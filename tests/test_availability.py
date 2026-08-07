"""Unit tests for freshness gating (Milestone 10).

Run with: pytest tests/test_availability.py -v
"""

import pandas as pd
import pytest

from news_reco.retrieval.availability import AvailabilityGate


def _articles(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_eligible_at_includes_articles_published_at_or_before_the_cutoff():
    articles = _articles(
        [
            {"article_id": "a", "published_time": pd.Timestamp("2023-05-01")},
            {"article_id": "b", "published_time": pd.Timestamp("2023-05-10")},
            {"article_id": "c", "published_time": pd.Timestamp("2023-05-20")},
        ]
    )
    gate = AvailabilityGate(articles)

    eligible = gate.eligible_at(pd.Timestamp("2023-05-10"))

    assert eligible == {"a", "b"}  # "c" (May 20) is after the cutoff


def test_eligible_at_before_any_publish_time_is_empty():
    articles = _articles([{"article_id": "a", "published_time": pd.Timestamp("2023-05-01")}])
    gate = AvailabilityGate(articles)
    assert gate.eligible_at(pd.Timestamp("2020-01-01")) == set()


def test_eligible_at_after_all_publish_times_includes_everything():
    articles = _articles(
        [
            {"article_id": "a", "published_time": pd.Timestamp("2023-05-01")},
            {"article_id": "b", "published_time": pd.Timestamp("2023-05-10")},
        ]
    )
    gate = AvailabilityGate(articles)
    assert gate.eligible_at(pd.Timestamp("2030-01-01")) == {"a", "b"}


def test_articles_with_missing_published_time_are_never_eligible():
    articles = _articles(
        [
            {"article_id": "a", "published_time": pd.Timestamp("2023-05-01")},
            {"article_id": "unknown", "published_time": pd.NaT},
        ]
    )
    gate = AvailabilityGate(articles)
    eligible = gate.eligible_at(pd.Timestamp("2030-01-01"))
    assert "unknown" not in eligible
    assert eligible == {"a"}


def test_constructor_rejects_a_dataset_with_no_published_time_data():
    articles = _articles(
        [
            {"article_id": "a", "published_time": pd.NaT},
            {"article_id": "b", "published_time": pd.NaT},
        ]
    )
    with pytest.raises(ValueError, match="no published_time data"):
        AvailabilityGate(articles)
