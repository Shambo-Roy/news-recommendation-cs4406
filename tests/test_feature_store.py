"""Verification tests for the FeatureStore (Milestone 4).

These run against the real cleaned + split MIND-small data, so they are
skipped (not failed) if scripts/clean_mind.py and scripts/split_mind.py
haven't been run yet.

Run with: pytest tests/test_feature_store.py -v
"""

from pathlib import Path

import pytest

from news_reco.features.store import FeatureStore
from news_reco.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(PROJECT_ROOT / "configs" / "mind_small.yaml")
SPLITS_PATH = CONFIG.processed_dir / "impression_splits.parquet"

pytestmark = pytest.mark.skipif(
    not SPLITS_PATH.exists(),
    reason="Cleaned/split data not present -- run scripts/clean_mind.py and scripts/split_mind.py first",
)


@pytest.fixture(scope="module")
def store() -> FeatureStore:
    return FeatureStore(CONFIG)


def test_get_article_text_returns_nonempty_string(store: FeatureStore):
    any_article_id = store.articles.iloc[0]["article_id"]
    text = store.get_article_text(any_article_id)
    assert isinstance(text, str)
    assert len(text) > 0


def test_get_article_title_matches_articles_table(store: FeatureStore):
    any_article_id = store.articles.iloc[0]["article_id"]
    assert store.get_article_title(any_article_id) == store.articles.iloc[0]["title"]


def test_split_partition_covers_every_impression_exactly_once(store: FeatureStore):
    train = store.get_impressions("train")
    val = store.get_impressions("val")
    test = store.get_impressions("test")
    all_impressions = store.get_impressions(None)

    assert len(train) + len(val) + len(test) == len(all_impressions)
    assert set(train["impression_id"]).isdisjoint(set(val["impression_id"]))
    assert set(val["impression_id"]).isdisjoint(set(test["impression_id"]))


def test_is_cold_start_matches_history_length(store: FeatureStore):
    impressions = store.get_impressions(None)
    assert (impressions["is_cold_start"] == (impressions["history_length"] == 0)).all()


def test_get_interactions_split_matches_its_impressions_split(store: FeatureStore):
    train_impression_ids = set(store.get_impressions("train")["impression_id"])
    train_interactions = store.get_interactions("train")
    # Every interaction labeled "train" must belong to an impression that is
    # itself labeled "train" -- catches a bug where the two tables' splits
    # were joined inconsistently.
    assert set(train_interactions["impression_id"]).issubset(train_impression_ids)
