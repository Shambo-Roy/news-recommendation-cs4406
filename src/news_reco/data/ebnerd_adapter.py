"""Transform raw EB-NeRD DataFrames into the project's common schema.

Structurally different from MIND in three ways, all verified directly
against the downloaded demo bundle before writing this adapter (see
reports/engineering_log.md):

1. articles.parquet is one shared file across every split (MIND's news.tsv
   was split-scoped, needing union + dedup); no dedup needed here.
2. History is a *separate* per-user file, one per split, genuinely updated
   between splits (a user's train-split history predates and differs from
   their validation-split history) -- MIND's history was embedded directly
   per-impression and static across the whole observed window.
3. Candidates come as two parallel arrays (article_ids_inview,
   article_ids_clicked), not MIND's single "id-label" string.
"""

from __future__ import annotations

import pandas as pd

from news_reco.data.schema import ARTICLES_COLUMNS, IMPRESSIONS_COLUMNS, INTERACTIONS_COLUMNS


def build_articles_table(articles_df: pd.DataFrame) -> pd.DataFrame:
    """Map EB-NeRD's articles.parquet into the common schema.

    Unlike MIND, this is a single already-deduplicated file shared across
    every split, so there's no union/dedup step here.
    """
    articles = pd.DataFrame(
        {
            "article_id": articles_df["article_id"].astype(str),
            "category": articles_df["category_str"],
            "subcategory": articles_df["subcategory"],
            "title": articles_df["title"].fillna(""),
            # EB-NeRD's subtitle is the closest analogue to MIND's abstract
            # -- a short summary line, not the full article body. Kept
            # consistent with the assignment's "titles and abstracts"
            # wording rather than indexing the much larger body text.
            "abstract": articles_df["subtitle"].fillna(""),
            # EB-NeRD genuinely has publish timestamps (unlike MIND) --
            # passed through as-is, enabling real freshness gating later.
            "published_time": articles_df["published_time"],
        }
    )
    articles["text"] = (articles["title"] + " " + articles["abstract"]).str.strip()
    return articles[ARTICLES_COLUMNS].reset_index(drop=True)


def build_impressions_and_interactions(
    *split_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explode EB-NeRD (behaviors, history) pairs into normalized impressions + interactions.

    Args:
        split_frames: One (behaviors_df, history_df) pair per raw split
            (e.g. one for official train, one for official validation).
            Each behaviors frame is joined against *its own* history frame
            by user_id -- using the wrong split's history would silently
            leak future-relative-to-that-split reading behaviour in, or
            use stale history where fresher was available.

    Like mind_adapter, mints a fresh globally-unique impression_id after
    concatenation rather than trusting the raw files' own numbering to be
    unique across splits -- defensive by design after Milestone 4's bug,
    even though a direct check found no actual collision in EB-NeRD's IDs.
    """
    all_impressions = []
    all_interactions = []
    next_impression_id = 0

    for behaviors_df, history_df in split_frames:
        history_lookup = dict(zip(history_df["user_id"], history_df["article_id_fixed"]))

        n = len(behaviors_df)
        global_ids = pd.RangeIndex(next_impression_id, next_impression_id + n)
        next_impression_id += n

        history_lists = [
            [str(article_id) for article_id in history_lookup.get(user_id, [])]
            for user_id in behaviors_df["user_id"]
        ]

        impressions = pd.DataFrame(
            {
                "impression_id": global_ids,
                "user_id": behaviors_df["user_id"].astype(str),
                "time": behaviors_df["impression_time"],
                "history": history_lists,
            }
        )
        all_impressions.append(impressions)

        inview_lists = [[str(a) for a in arr] for arr in behaviors_df["article_ids_inview"]]
        clicked_sets = [{str(a) for a in arr} for arr in behaviors_df["article_ids_clicked"]]

        exploded = pd.DataFrame({"impression_id": global_ids, "inview": inview_lists}).explode("inview")
        clicked_lookup = dict(zip(global_ids, clicked_sets))
        clicked_labels = [
            1 if article_id in clicked_lookup[impression_id] else 0
            for impression_id, article_id in zip(exploded["impression_id"], exploded["inview"])
        ]

        interactions = pd.DataFrame(
            {
                "impression_id": exploded["impression_id"],
                "article_id": exploded["inview"],
                "clicked": pd.array(clicked_labels, dtype="int8"),
            }
        )
        all_interactions.append(interactions)

    impressions_table = pd.concat(all_impressions, ignore_index=True)[IMPRESSIONS_COLUMNS]
    interactions_table = pd.concat(all_interactions, ignore_index=True)[INTERACTIONS_COLUMNS]

    return impressions_table.reset_index(drop=True), interactions_table.reset_index(drop=True)
