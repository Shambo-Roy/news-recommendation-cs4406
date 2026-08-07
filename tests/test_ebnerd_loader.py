"""Verification tests for the EB-NeRD raw-file loader.

These tests run against the real downloaded EB-NeRD demo bundle, so they
are skipped (not failed) if it hasn't been downloaded yet. Run
scripts/download_ebnerd.py first.

Run with: pytest tests/test_ebnerd_loader.py -v
"""

from pathlib import Path

import pytest

from news_reco.data.ebnerd_loader import load_articles, load_behaviors, load_history
from news_reco.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(PROJECT_ROOT / "configs" / "ebnerd_demo.yaml")
DEMO_DIR = CONFIG.raw_dir / "demo"

pytestmark = pytest.mark.skipif(
    not DEMO_DIR.exists(),
    reason="EB-NeRD demo bundle not downloaded yet -- run scripts/download_ebnerd.py first",
)


def test_load_articles_has_expected_columns_and_published_time():
    articles = load_articles(DEMO_DIR / "articles.parquet")
    for expected_column in ("article_id", "title", "subtitle", "published_time", "category_str"):
        assert expected_column in articles.columns
    assert len(articles) > 0
    assert articles["article_id"].is_unique


def test_load_behaviors_has_expected_columns():
    behaviors = load_behaviors(DEMO_DIR / "train" / "behaviors.parquet")
    for expected_column in ("impression_id", "user_id", "impression_time", "article_ids_inview", "article_ids_clicked"):
        assert expected_column in behaviors.columns
    assert len(behaviors) > 0


def test_load_history_has_expected_columns():
    history = load_history(DEMO_DIR / "train" / "history.parquet")
    for expected_column in ("user_id", "article_id_fixed", "impression_time_fixed"):
        assert expected_column in history.columns
    assert len(history) > 0
    assert history["user_id"].is_unique  # one row per user
