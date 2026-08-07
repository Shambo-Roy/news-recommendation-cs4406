"""Unit tests for the EB-NeRD -> common-schema adapter.

Small, hand-crafted DataFrames with known-correct expected outputs --
shaped like news_reco.data.ebnerd_loader's actual output (numpy arrays for
inview/clicked/history, matching what pandas gives back from the real
parquet files), not real downloaded data.

Run with: pytest tests/test_ebnerd_adapter.py -v
"""

import numpy as np
import pandas as pd

from news_reco.data.ebnerd_adapter import build_articles_table, build_impressions_and_interactions


def test_build_articles_table_maps_columns_correctly():
    articles_df = pd.DataFrame(
        {
            "article_id": [101, 102],
            "title": ["Foerste overskrift", "Anden overskrift"],
            "subtitle": ["En kort undertekst.", None],
            "published_time": pd.to_datetime(["2023-05-01 10:00:00", "2023-05-02 11:00:00"]),
            "category_str": ["sport", "krimi"],
            "subcategory": [[1, 2], [3]],
        }
    )

    articles = build_articles_table(articles_df)

    assert len(articles) == 2
    assert set(articles["article_id"]) == {"101", "102"}  # cast to str, like MIND's ids

    a101 = articles.set_index("article_id").loc["101"]
    assert a101["abstract"] == "En kort undertekst."
    assert a101["text"] == "Foerste overskrift En kort undertekst."
    assert a101["published_time"] == pd.Timestamp("2023-05-01 10:00:00")

    a102 = articles.set_index("article_id").loc["102"]
    assert a102["abstract"] == ""  # missing subtitle -> empty string, not NaN
    assert a102["text"] == "Anden overskrift"


def test_build_impressions_and_interactions_uses_each_splits_own_history():
    # Same user (U1) in both splits, with DIFFERENT history -- must not
    # cross-contaminate: split-a impressions get history_a's list, split-b
    # impressions get history_b's list. Raw impression_id=1 is reused in
    # both splits on purpose -- it must never collide in the output, since
    # it's never even read (a fresh id is always minted).
    behaviors_a = pd.DataFrame(
        {
            "impression_id": [1],
            "user_id": [1],
            "impression_time": [pd.Timestamp("2023-05-18 10:00:00")],
            "article_ids_inview": [np.array([10, 11, 12])],
            "article_ids_clicked": [np.array([11])],
        }
    )
    history_a = pd.DataFrame({"user_id": [1], "article_id_fixed": [np.array([1, 2])]})

    behaviors_b = pd.DataFrame(
        {
            "impression_id": [1],
            "user_id": [1],
            "impression_time": [pd.Timestamp("2023-05-25 10:00:00")],
            "article_ids_inview": [np.array([20, 21])],
            "article_ids_clicked": [np.array([20, 21])],  # both clicked
        }
    )
    history_b = pd.DataFrame({"user_id": [1], "article_id_fixed": [np.array([1, 2, 11])]})  # extended vs. a

    impressions, interactions = build_impressions_and_interactions(
        (behaviors_a, history_a), (behaviors_b, history_b)
    )

    assert len(impressions) == 2
    assert impressions["impression_id"].is_unique

    imp_a = impressions[impressions["time"] == pd.Timestamp("2023-05-18 10:00:00")].iloc[0]
    imp_b = impressions[impressions["time"] == pd.Timestamp("2023-05-25 10:00:00")].iloc[0]

    assert imp_a["history"] == ["1", "2"]
    assert imp_b["history"] == ["1", "2", "11"]  # split b's own, extended history

    a_candidates = interactions[interactions["impression_id"] == imp_a["impression_id"]]
    assert set(zip(a_candidates["article_id"], a_candidates["clicked"])) == {("10", 0), ("11", 1), ("12", 0)}

    b_candidates = interactions[interactions["impression_id"] == imp_b["impression_id"]]
    assert set(zip(b_candidates["article_id"], b_candidates["clicked"])) == {("20", 1), ("21", 1)}


def test_build_impressions_and_interactions_cold_start_user_gets_empty_history():
    behaviors = pd.DataFrame(
        {
            "impression_id": [1],
            "user_id": [999],  # not present in history_df at all
            "impression_time": [pd.Timestamp("2023-05-18 10:00:00")],
            "article_ids_inview": [np.array([10])],
            "article_ids_clicked": [np.array([])],
        }
    )
    history = pd.DataFrame({"user_id": [1], "article_id_fixed": [np.array([1, 2])]})

    impressions, interactions = build_impressions_and_interactions((behaviors, history))

    assert impressions.iloc[0]["history"] == []
    assert interactions.iloc[0]["clicked"] == 0
