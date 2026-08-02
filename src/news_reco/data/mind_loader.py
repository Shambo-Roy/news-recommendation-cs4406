"""Load raw MIND TSV files into pandas DataFrames.

MIND ships two flat files per split (train/dev): news.tsv (article
metadata) and behaviors.tsv (impression logs). Both are tab-separated with
no header row, so column names are supplied here based on the official
MIND data format -- verified directly against the downloaded files, not
assumed from documentation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIORS_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]


def load_news(news_path: str | Path) -> pd.DataFrame:
    """Load news.tsv into a DataFrame with one row per article.

    `title_entities` and `abstract_entities` are left as raw JSON strings --
    parsing them is deferred to whichever later module actually needs entity
    information, so this loader stays a thin, fast I/O layer.
    """
    return pd.read_csv(
        news_path,
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
        dtype={"news_id": str, "category": str, "subcategory": str},
    )


def load_behaviors(behaviors_path: str | Path) -> pd.DataFrame:
    """Load behaviors.tsv into a DataFrame with one row per impression.

    `history` and `impressions` are left as raw space-separated strings --
    exploding `impressions` into one row per (article, clicked) pair happens
    during cleaning (Milestone 2), not here, since that's a bigger decision
    about the impression-candidate table's shape.
    """
    df = pd.read_csv(
        behaviors_path,
        sep="\t",
        header=None,
        names=BEHAVIORS_COLUMNS,
        dtype={"impression_id": int, "user_id": str},
    )
    # MIND timestamps look like "11/11/2019 9:05:58 AM". Parsing here (rather
    # than leaving it a string) is what lets Milestone 3's temporal split
    # compare impressions by actual time, not lexicographic string order.
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    return df
