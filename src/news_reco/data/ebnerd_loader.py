"""Load raw EB-NeRD parquet files into pandas DataFrames.

EB-NeRD ships its data as three parquet files per bundle: a single
top-level articles.parquet (the whole catalog -- unlike MIND's news.tsv,
this is *not* split-scoped, every split shares one file), and per-split
behaviors.parquet + history.parquet (under train/ and validation/
subdirectories). Parquet already carries typed columns, so unlike MIND's
raw TSV loader there's no dtype/column-name work to do here -- these are
thin, named wrappers that give the adapter a stable interface, verified
directly against the downloaded files (see reports/engineering_log.md),
not assumed from documentation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_articles(path: str | Path) -> pd.DataFrame:
    """Load articles.parquet -- the full EB-NeRD article catalog (shared across all splits)."""
    return pd.read_parquet(path)


def load_behaviors(path: str | Path) -> pd.DataFrame:
    """Load one split's behaviors.parquet (impression logs)."""
    return pd.read_parquet(path)


def load_history(path: str | Path) -> pd.DataFrame:
    """Load one split's history.parquet (per-user click history, as of that split's window start)."""
    return pd.read_parquet(path)
