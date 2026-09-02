"""Materialize the small feature store: impressions with derived features.

Uses news_reco.features.store.FeatureStore to attach split labels and the
cold-start/warm flag to every impression, and writes the result to
data/processed/mind_small/impressions_features.parquet. Also prints the
cold-start/warm counts per split -- the required rubric slice.

Usage:
    python scripts/build_feature_store.py --config configs/mind_small.yaml
"""

from __future__ import annotations

import argparse

from news_reco.features.store import FeatureStore
from news_reco.utils.config import PipelineConfig, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()
    config: PipelineConfig = load_config(args.config)

    store = FeatureStore(config)
    impressions = store.get_impressions(None)

    print("Cold-start vs. warm impressions, by split:")
    print(impressions.groupby("split")["is_cold_start"].value_counts().to_string())

    out_columns = ["impression_id", "user_id", "time", "split", "history_length", "is_cold_start"]
    out_path = config.processed_dir / "impressions_features.parquet"
    impressions[out_columns].to_parquet(out_path, index=False)
    print(f"\nWrote {len(impressions):,} rows to {out_path}")


if __name__ == "__main__":
    main()
