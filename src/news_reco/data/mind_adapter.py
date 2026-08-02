"""Transform raw MIND DataFrames into the project's common schema.

The common schema is dataset-agnostic: three tables (articles, impressions,
interactions) with the same column names regardless of whether the source
was MIND or (later) EB-NeRD. Every module downstream of this one is written
against news_reco.data.schema, not against MIND's raw tsv format.
"""

from __future__ import annotations

import pandas as pd

from news_reco.data.schema import ARTICLES_COLUMNS, IMPRESSIONS_COLUMNS, INTERACTIONS_COLUMNS


def build_articles_table(*news_frames: pd.DataFrame) -> pd.DataFrame:
    """Union one or more raw MIND news.tsv frames into one article catalog.

    MIND's news.tsv is scoped per split -- train/dev each list only the
    articles that appear in that split's impressions -- so the full catalog
    is the union across splits. Articles appearing in both are deduplicated
    by news_id, keeping the first occurrence (an article's content doesn't
    change between splits).
    """
    combined = pd.concat(news_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="news_id", keep="first")

    articles = pd.DataFrame(
        {
            "article_id": combined["news_id"],
            "category": combined["category"],
            "subcategory": combined["subcategory"],
            "title": combined["title"],
            # Missing abstracts (~5% of articles) become an empty string
            # rather than dropping the article: losing real catalog items
            # over one optional field would silently shrink recall for
            # every retrieval method built on top of this table.
            "abstract": combined["abstract"].fillna(""),
        }
    )
    # `text` is the single field BM25 and embeddings will actually index --
    # computed once here so every retrieval method sees an identical corpus.
    articles["text"] = (articles["title"] + " " + articles["abstract"]).str.strip()

    return articles[ARTICLES_COLUMNS].reset_index(drop=True)


def _parse_history(history: object) -> list:
    """Turn a raw space-separated history string (or NaN) into a list of IDs."""
    if pd.isna(history):
        return []
    return history.split()


def build_impressions_and_interactions(
    *behaviors_frames: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explode raw MIND behaviors frames into normalized impressions + interactions.

    `impressions` holds one row per impression (its user, time, and click
    history); `interactions` holds one row per (impression, candidate
    article, clicked) pair. Splitting these into two tables, rather than
    repeating `history` on every candidate row, avoids duplicating a
    dozens-of-items list across every candidate of the same impression.
    """
    behaviors = pd.concat(behaviors_frames, ignore_index=True)

    # MIND's own impression_id is only unique *within* one split's file --
    # train and dev each number their impressions starting from 1, so
    # concatenating them naively creates duplicate IDs across splits (every
    # dev impression_id collides with a train one). Every join in this
    # project treats impression_id as a global primary key, so we replace it
    # here with a fresh, guaranteed-unique id rather than trusting MIND's
    # file-local numbering to mean what it looks like it means.
    global_impression_id = pd.RangeIndex(len(behaviors))

    impressions = pd.DataFrame(
        {
            "impression_id": global_impression_id,
            "user_id": behaviors["user_id"],
            "time": behaviors["time"],
            "history": behaviors["history"].apply(_parse_history),
        }
    )[IMPRESSIONS_COLUMNS].reset_index(drop=True)

    # Explode "N20678-0 N39317-0 ..." into one row per candidate. Vectorized
    # string ops here, rather than a Python loop, because this runs over
    # tens of thousands of impressions with dozens of candidates each.
    candidate_pairs = behaviors["impressions"].str.split()
    exploded = (
        pd.DataFrame({"impression_id": global_impression_id, "pair": candidate_pairs})
        .explode("pair")
    )
    # rsplit from the right, once: article IDs never contain "-", but
    # splitting from the label side is the robust way to write this either way.
    split_pair = exploded["pair"].str.rsplit("-", n=1, expand=True)

    interactions = pd.DataFrame(
        {
            "impression_id": exploded["impression_id"],
            "article_id": split_pair[0],
            "clicked": split_pair[1].astype("int8"),
        }
    )[INTERACTIONS_COLUMNS].reset_index(drop=True)

    return impressions, interactions
