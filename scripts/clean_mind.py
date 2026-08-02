"""Clean raw MIND-small train/dev files into the project's common schema.

Reads data/raw/mind_small/{train,dev}/.../{news,behaviors}.tsv, unions and
normalizes them, and writes three parquet tables to data/interim/mind_small/:
articles.parquet, impressions.parquet, interactions.parquet.

Usage:
    python scripts/clean_mind.py --config configs/mind_small.yaml
"""

from __future__ import annotations

import argparse

from news_reco.data.mind_adapter import build_articles_table, build_impressions_and_interactions
from news_reco.data.mind_loader import load_behaviors, load_news
from news_reco.utils.config import PipelineConfig, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()
    config: PipelineConfig = load_config(args.config)

    train_dir = config.raw_dir / "train" / "MINDsmall_train"
    dev_dir = config.raw_dir / "dev" / "MINDsmall_dev"

    print("Loading raw files ...")
    train_news = load_news(train_dir / "news.tsv")
    dev_news = load_news(dev_dir / "news.tsv")
    train_behaviors = load_behaviors(train_dir / "behaviors.tsv")
    dev_behaviors = load_behaviors(dev_dir / "behaviors.tsv")

    print("Building common-schema tables ...")
    articles = build_articles_table(train_news, dev_news)
    impressions, interactions = build_impressions_and_interactions(train_behaviors, dev_behaviors)

    config.interim_dir.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(config.interim_dir / "articles.parquet", index=False)
    impressions.to_parquet(config.interim_dir / "impressions.parquet", index=False)
    interactions.to_parquet(config.interim_dir / "interactions.parquet", index=False)

    dedup_count = (len(train_news) + len(dev_news)) - len(articles)
    print(f"\narticles:     {len(articles):,} rows ({dedup_count:,} duplicates removed across train/dev)")
    print(f"impressions:  {len(impressions):,} rows")
    print(f"interactions: {len(interactions):,} rows")
    print(f"click rate:   {interactions['clicked'].mean():.4f}")
    print(f"empty text (title+abstract both blank): {(articles['text'] == '').sum():,}")
    print(f"\nWrote parquet files to {config.interim_dir}")


if __name__ == "__main__":
    main()
