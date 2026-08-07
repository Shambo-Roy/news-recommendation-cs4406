"""Clean raw EB-NeRD train/validation files into the project's common schema.

Reads data/raw/<dataset>/<bundle>/{articles.parquet, train/, validation/},
unions and normalizes them, and writes three parquet tables to
data/interim/<dataset>/: articles.parquet, impressions.parquet,
interactions.parquet -- the same three tables Milestone 2 produces for
MIND, so every downstream module (splitting, features, retrieval,
evaluation) works unmodified.

Usage:
    python scripts/clean_ebnerd.py --config configs/ebnerd_demo.yaml --bundle demo
"""

from __future__ import annotations

import argparse

from news_reco.data.ebnerd_adapter import build_articles_table, build_impressions_and_interactions
from news_reco.data.ebnerd_loader import load_articles, load_behaviors, load_history
from news_reco.utils.config import PipelineConfig, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument("--bundle", choices=["demo", "small"], default="demo")
    args = parser.parse_args()
    config: PipelineConfig = load_config(args.config)

    bundle_dir = config.raw_dir / args.bundle

    print("Loading raw files ...")
    articles_raw = load_articles(bundle_dir / "articles.parquet")
    train_behaviors = load_behaviors(bundle_dir / "train" / "behaviors.parquet")
    train_history = load_history(bundle_dir / "train" / "history.parquet")
    val_behaviors = load_behaviors(bundle_dir / "validation" / "behaviors.parquet")
    val_history = load_history(bundle_dir / "validation" / "history.parquet")

    print("Building common-schema tables ...")
    articles = build_articles_table(articles_raw)
    impressions, interactions = build_impressions_and_interactions(
        (train_behaviors, train_history), (val_behaviors, val_history)
    )

    config.interim_dir.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(config.interim_dir / "articles.parquet", index=False)
    impressions.to_parquet(config.interim_dir / "impressions.parquet", index=False)
    interactions.to_parquet(config.interim_dir / "interactions.parquet", index=False)

    print(f"\narticles:     {len(articles):,} rows")
    print(f"impressions:  {len(impressions):,} rows")
    print(f"interactions: {len(interactions):,} rows")
    print(f"click rate:   {interactions['clicked'].mean():.4f}")
    print(f"empty text (title+abstract both blank): {(articles['text'] == '').sum():,}")
    print(f"articles missing published_time: {articles['published_time'].isna().sum():,} / {len(articles):,}")
    print(f"\nWrote parquet files to {config.interim_dir}")


if __name__ == "__main__":
    main()
