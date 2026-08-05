"""A minimal feature store: fast, consistent lookups over the cleaned MIND tables.

A feature store's job is to compute a derived signal exactly once and serve
it the same way to every consumer -- training code, evaluation code, a
retrieval script -- so nobody re-derives a join or a column slightly
differently and silently gets different numbers. At this project's current
scale that doesn't need to be a database or an infrastructure system: this
class loads the cleaned parquet tables once and gives every later module the
same lookups and the same feature definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from news_reco.utils.config import PipelineConfig


class FeatureStore:
    """Loads the cleaned + split-labeled MIND tables once and serves lookups.

    Every later module (retrieval, evaluation) should read articles and
    impressions through this class instead of re-reading and re-joining the
    parquet files itself, so there is exactly one definition of "this
    impression's split" or "this article's text" that everything agrees on.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._articles = pd.read_parquet(config.interim_dir / "articles.parquet").set_index(
            "article_id", drop=False
        )

        impressions = pd.read_parquet(config.interim_dir / "impressions.parquet")
        interactions = pd.read_parquet(config.interim_dir / "interactions.parquet")
        splits_path = config.processed_dir / "impression_splits.parquet"
        if not splits_path.exists():
            raise FileNotFoundError(f"{splits_path} not found -- run scripts/split_mind.py first.")
        splits = pd.read_parquet(splits_path)

        impressions = impressions.merge(splits, on="impression_id", how="left")
        if impressions["split"].isna().any():
            raise ValueError("Some impressions have no split label -- splits file is stale or incomplete.")

        # These are "features" in the ML sense (derived signals later
        # modules consume), which is this class's job -- not Milestone 2
        # cleaning's job. They're safe to compute directly from `history`:
        # Milestone 3 already verified that field is causally correct
        # (doesn't reference the future), so nothing here can leak.
        impressions["history_length"] = impressions["history"].map(len)
        impressions["is_cold_start"] = impressions["history_length"] == 0

        self._impressions = impressions
        self._interactions = interactions.merge(splits, on="impression_id", how="left")

        # Embeddings are optional at construction time: most modules built
        # before Milestone 7 (and most scripts today) never touch them, so
        # loading is skipped gracefully -- not required -- if the file
        # doesn't exist yet, rather than adding a new hard dependency to
        # every existing script.
        self._embeddings: dict[str, np.ndarray] | None = None
        embeddings_path = config.interim_dir / "article_embeddings.npz"
        if embeddings_path.exists():
            loaded = np.load(embeddings_path)
            self._embeddings = dict(zip(loaded["article_ids"], loaded["embeddings"]))

    @property
    def articles(self) -> pd.DataFrame:
        """The full article catalog, indexed by article_id."""
        return self._articles

    def get_article_text(self, article_id: str) -> str:
        """Return the indexable text (title + abstract) for one article."""
        return self._articles.loc[article_id, "text"]

    def get_article_title(self, article_id: str) -> str:
        """Return just the title for one article (used to build BM25 queries from history)."""
        return self._articles.loc[article_id, "title"]

    def get_article_embedding(self, article_id: str) -> np.ndarray:
        """Return the precomputed embedding vector for one article.

        Raises RuntimeError if scripts/compute_article_embeddings.py hasn't
        been run yet -- embeddings are loaded only if present (see
        __init__), so this is where the missing-file case actually surfaces.
        """
        if self._embeddings is None:
            raise RuntimeError(
                "Article embeddings not found -- run scripts/compute_article_embeddings.py first."
            )
        return self._embeddings[article_id]

    def get_impressions(self, split: str | None = None) -> pd.DataFrame:
        """Return impressions, with history_length/is_cold_start attached.

        Args:
            split: "train", "val", or "test" to filter to one split; None
                for all impressions across every split.
        """
        if split is None:
            return self._impressions
        return self._impressions[self._impressions["split"] == split]

    def get_interactions(self, split: str | None = None) -> pd.DataFrame:
        """Return (impression, candidate article, clicked) rows, optionally filtered to one split."""
        if split is None:
            return self._interactions
        return self._interactions[self._interactions["split"] == split]
