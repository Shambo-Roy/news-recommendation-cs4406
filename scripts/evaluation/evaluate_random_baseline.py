"""Sanity-check the evaluation harness against real data using random scores.

No retrieval method exists yet (that's Milestones 6-7) -- this script wires
the harness up to real val-split interactions with a *random* score per
candidate, purely to validate the metric implementations end-to-end on
real, large-scale data before anything downstream depends on them. A random
ranking has a well-known expected AUC of 0.5; if this script reports
something far from that, the metrics (not the "model") are broken.

Usage:
    python scripts/evaluate_random_baseline.py --config configs/mind_small.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from news_reco.evaluation.harness import aggregate, evaluate_ranking
from news_reco.features.store import FeatureStore
from news_reco.utils.config import PipelineConfig, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()
    config: PipelineConfig = load_config(args.config)

    store = FeatureStore(config)
    interactions = store.get_interactions("val")
    print(
        f"Evaluating {interactions['impression_id'].nunique():,} val impressions "
        f"({len(interactions):,} candidate rows) with random scores ..."
    )

    rng = np.random.default_rng(config.random_seed)
    per_impression_results = []
    for _, group in interactions.groupby("impression_id", sort=False):
        labels = group["clicked"].to_numpy()
        scores = rng.random(len(labels))
        per_impression_results.append(evaluate_ranking(labels, scores))

    summary = aggregate(per_impression_results)
    print()
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    auc_mean = summary.loc["auc", "mean"]
    print(f"\nSanity check: random-score AUC = {auc_mean:.4f} (expected ~0.5)")


if __name__ == "__main__":
    main()
