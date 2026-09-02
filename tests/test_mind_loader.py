"""Verification tests for the MIND raw-file loader (Milestone 1).

These tests run against the real downloaded MIND-small files, so they are
skipped (not failed) if the dataset hasn't been downloaded yet. Run
scripts/data/download_mind_small.py first.

Run with: pytest tests/test_mind_loader.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

from news_reco.data.mind_loader import load_behaviors, load_news
from news_reco.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(PROJECT_ROOT / "configs" / "mind_small.yaml")
TRAIN_DIR = CONFIG.raw_dir / "train" / "MINDsmall_train"

pytestmark = pytest.mark.skipif(
    not TRAIN_DIR.exists(),
    reason="MIND-small not downloaded yet -- run scripts/data/download_mind_small.py first",
)


def test_load_news_has_expected_columns():
    news = load_news(TRAIN_DIR / "news.tsv")
    assert list(news.columns) == [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    assert len(news) > 0
    assert news["news_id"].is_unique


def test_load_behaviors_has_expected_columns_and_parsed_time():
    behaviors = load_behaviors(TRAIN_DIR / "behaviors.tsv")
    assert list(behaviors.columns) == [
        "impression_id",
        "user_id",
        "time",
        "history",
        "impressions",
    ]
    assert len(behaviors) > 0
    assert pd.api.types.is_datetime64_any_dtype(behaviors["time"])
    assert behaviors["impression_id"].is_unique
