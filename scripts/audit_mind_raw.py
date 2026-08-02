"""Audit the raw MIND-small files: shapes, date ranges, and basic sanity checks.

This is Milestone 1's verification step -- it doesn't clean or transform
anything, it just reports what's actually in the raw files so later
milestones (cleaning, temporal splitting) are built against facts, not
assumptions about the dataset.

Usage:
    python scripts/audit_mind_raw.py --config configs/mind_small.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from news_reco.data.mind_loader import load_behaviors, load_news
from news_reco.utils.config import PipelineConfig, load_config


def audit_split(split_name: str, news_path: Path, behaviors_path: Path) -> None:
    """Print shape, date-range, and label-sanity facts for one split."""
    news = load_news(news_path)
    behaviors = load_behaviors(behaviors_path)

    print(f"\n=== {split_name} ===")
    print(f"news.tsv:      {len(news):,} articles, {news['category'].nunique()} categories")
    print(f"  missing abstract: {news['abstract'].isna().sum():,} / {len(news):,}")
    print(f"behaviors.tsv: {len(behaviors):,} impressions, {behaviors['user_id'].nunique():,} unique users")
    print(f"  time range: {behaviors['time'].min()}  to  {behaviors['time'].max()}")

    # Sanity-check the "NewsID-label" format in a sample before any later
    # module relies on it: every label must be exactly "0" or "1".
    sample_impressions = behaviors["impressions"].head(1000).str.split()
    all_labels = {pair.rsplit("-", 1)[1] for row in sample_impressions for pair in row}
    print(f"  impression labels seen in first 1,000 rows: {sorted(all_labels)}")

    empty_history = behaviors["history"].isna().sum()
    print(f"  impressions with empty click history (cold-start candidates): {empty_history:,} / {len(behaviors):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()
    config: PipelineConfig = load_config(args.config)

    audit_split(
        "train",
        config.raw_dir / "train" / "MINDsmall_train" / "news.tsv",
        config.raw_dir / "train" / "MINDsmall_train" / "behaviors.tsv",
    )
    audit_split(
        "dev",
        config.raw_dir / "dev" / "MINDsmall_dev" / "news.tsv",
        config.raw_dir / "dev" / "MINDsmall_dev" / "behaviors.tsv",
    )


if __name__ == "__main__":
    main()
