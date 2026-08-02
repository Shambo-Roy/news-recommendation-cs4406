"""Unit tests for the MIND -> common-schema adapter (Milestone 2).

Unlike test_mind_loader.py, these tests use small, hand-crafted DataFrames
with known-correct expected outputs, rather than the real downloaded
dataset. That lets us pin down exact edge-case behaviour (deduplication,
missing abstracts, empty history) precisely, and the tests keep working on
any machine regardless of whether MIND-small has been downloaded.

Run with: pytest tests/test_mind_adapter.py -v
"""

import pandas as pd

from news_reco.data.mind_adapter import build_articles_table, build_impressions_and_interactions


def _news_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame shaped like news_reco.data.mind_loader.load_news()'s output."""
    return pd.DataFrame(rows)


def test_build_articles_table_dedupes_and_fills_missing_abstract():
    train_news = _news_frame(
        [
            {"news_id": "N1", "category": "news", "subcategory": "politics",
             "title": "Title One", "abstract": "Abstract one.", "url": "u1",
             "title_entities": "[]", "abstract_entities": "[]"},
            {"news_id": "N2", "category": "sports", "subcategory": "football",
             "title": "Title Two", "abstract": None, "url": "u2",
             "title_entities": "[]", "abstract_entities": "[]"},
        ]
    )
    dev_news = _news_frame(
        [
            # N1 also appears in dev -- must be deduplicated, keeping train's row.
            {"news_id": "N1", "category": "news", "subcategory": "politics",
             "title": "Title One", "abstract": "Abstract one.", "url": "u1",
             "title_entities": "[]", "abstract_entities": "[]"},
            {"news_id": "N3", "category": "finance", "subcategory": "markets",
             "title": "Title Three", "abstract": "Abstract three.", "url": "u3",
             "title_entities": "[]", "abstract_entities": "[]"},
        ]
    )

    articles = build_articles_table(train_news, dev_news)

    assert len(articles) == 3
    assert set(articles["article_id"]) == {"N1", "N2", "N3"}

    n2 = articles.set_index("article_id").loc["N2"]
    assert n2["abstract"] == ""
    assert n2["text"] == "Title Two"  # no trailing space from the empty abstract

    n1 = articles.set_index("article_id").loc["N1"]
    assert n1["text"] == "Title One Abstract one."


def test_build_impressions_and_interactions_explodes_candidates():
    # Deliberately reuse impression_id=1 in both frames: MIND's train and
    # dev files each number impressions starting from 1 independently, so a
    # correct implementation must not trust these raw IDs to be unique
    # across frames -- it has to mint its own.
    behaviors_a = pd.DataFrame(
        {
            "impression_id": [1],
            "user_id": ["U1"],
            "time": [pd.Timestamp("2019-11-10 10:00:00")],
            "history": ["N1 N2"],
            "impressions": ["N3-0 N4-1"],
        }
    )
    behaviors_b = pd.DataFrame(
        {
            "impression_id": [1],  # collides with behaviors_a's impression_id
            "user_id": ["U2"],
            "time": [pd.Timestamp("2019-11-11 09:00:00")],
            "history": [None],  # cold-start: no prior clicks
            "impressions": ["N5-0"],
        }
    )

    impressions, interactions = build_impressions_and_interactions(behaviors_a, behaviors_b)

    assert len(impressions) == 2
    assert impressions["impression_id"].is_unique, "colliding raw MIND impression_ids must not survive"

    imp1_id, imp2_id = impressions["impression_id"]
    imp1 = impressions.set_index("impression_id").loc[imp1_id]
    assert imp1["history"] == ["N1", "N2"]
    imp2 = impressions.set_index("impression_id").loc[imp2_id]
    assert imp2["history"] == []

    assert len(interactions) == 3
    assert set(interactions["impression_id"]) == {imp1_id, imp2_id}
    imp1_candidates = set(
        zip(
            interactions.loc[interactions["impression_id"] == imp1_id, "article_id"],
            interactions.loc[interactions["impression_id"] == imp1_id, "clicked"],
        )
    )
    assert imp1_candidates == {("N3", 0), ("N4", 1)}

    imp2_candidates = set(
        zip(
            interactions.loc[interactions["impression_id"] == imp2_id, "article_id"],
            interactions.loc[interactions["impression_id"] == imp2_id, "clicked"],
        )
    )
    assert imp2_candidates == {("N5", 0)}
