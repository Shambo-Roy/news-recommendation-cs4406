"""Milestone 6: evaluate the trained GBDT re-ranker against the existing
semantic baseline, on the SAME impressions/candidates/labels, using
Component 1's own metric implementations unchanged.

Fair-comparison guarantee: the "semantic baseline" score used here is not
recomputed independently -- it's literally the `semantic_score` column
already sitting in the cached feature table (produced by the identical
EmbeddingIndex.score_documents() call the real Codabench submission uses,
via news_reco.reranking.gbdt.build_impression_rows). So this comparison is
guaranteed apples-to-apples: same split, same impressions, same candidate
sets, same labels, same AUC/MRR/nDCG/bootstrap code
(news_reco.evaluation.harness/bootstrap, reused unmodified) -- the ONLY
thing that differs between the two systems' numbers below is the score used
to rank each impression's candidates.

Beyond-accuracy scope note: Component 1's diversity/novelty/coverage
(scripts/evaluate_full.py) are computed over OPEN-CORPUS top-500 retrieval
lists (RRF-fused across the whole catalog). The GBDT re-ranker only ever
re-ranks an impression's OWN shown candidates (per the assignment's brief --
"rank the ACTUAL candidates in each impression"), so there is no open-corpus
list to measure here. This script instead reports diversity/novelty/coverage
over each impression's own top-10 *within its candidate set* -- a
CONFINED-CANDIDATE definition, computed identically for both systems so the
GBDT-vs-semantic comparison stays fair, but NOT numerically comparable to
Component 1's open-corpus numbers in reports/engineering_log.md. This
difference is called out again in the printed report, not just here.

Usage:
    python scripts/evaluate_gbdt_reranker.py --config configs/gbdt_mind_small.yaml --split val
    python scripts/evaluate_gbdt_reranker.py --config configs/gbdt_mind_small.yaml --split test
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sentence_transformers  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from news_reco.evaluation.beyond_accuracy import CoverageTracker, NoveltyModel, intra_list_diversity  # noqa: E402
from news_reco.evaluation.bootstrap import bootstrap_ci, paired_bootstrap_ci  # noqa: E402
from news_reco.evaluation.harness import aggregate, evaluate_ranking  # noqa: E402
from news_reco.evaluation.slicing import classify_head_tail  # noqa: E402
from news_reco.features.store import FeatureStore  # noqa: E402
from news_reco.reranking.gbdt import load_gbdt_config  # noqa: E402
from news_reco.utils.config import load_config  # noqa: E402

BEYOND_ACCURACY_K = 10
SYSTEMS = ("gbdt", "semantic_baseline")
COVERAGE_BOOTSTRAP_RESAMPLES = 200  # lower than the usual 1000 -- see bootstrap_coverage_ci


def bootstrap_coverage_ci(
    top_k_lists_a: list[list[str]],
    top_k_lists_b: list[list[str]],
    n_articles: int,
    n_resamples: int = COVERAGE_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs for confined-candidate coverage: each system's own CI,
    plus a PAIRED CI for the delta (a - b).

    Coverage is a set-union statistic over the WHOLE evaluation run, not a
    per-impression average -- news_reco.evaluation.bootstrap's
    bootstrap_ci/paired_bootstrap_ci (Component 1, reused everywhere else
    in this script) assume one scalar value per resampling unit, which
    coverage isn't. This resamples at the IMPRESSION level instead: each
    resample draws a set of impression indices (with replacement, same
    count as the real run) and recomputes each system's coverage as the
    union of ONLY those resampled impressions' top-k lists over the
    catalog. The SAME resampled indices are used for both systems on each
    resample, preserving the pairing -- same convention as
    paired_bootstrap_ci, applied to a statistic that function can't
    handle directly.

    n_resamples defaults lower than the project's usual 1000 (see
    bootstrap_ci/paired_bootstrap_ci) because this is O(n_resamples x
    n_impressions x k) in pure Python (rebuilding a set per resample, not
    a vectorized array op) -- at EB-NeRD test scale (~245K impressions)
    1000 resamples would take several minutes for one number; 200 is
    still enough to estimate a stable 95% interval and keeps this a
    one-time, end-of-run cost.
    """
    n = len(top_k_lists_a)
    rng = np.random.default_rng(seed)
    coverage_a = np.empty(n_resamples)
    coverage_b = np.empty(n_resamples)
    for r in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        seen_a: set[str] = set()
        seen_b: set[str] = set()
        for i in indices:
            seen_a.update(top_k_lists_a[i])
            seen_b.update(top_k_lists_b[i])
        coverage_a[r] = len(seen_a) / n_articles
        coverage_b[r] = len(seen_b) / n_articles
    diff = coverage_a - coverage_b
    return {
        "a": (float(np.percentile(coverage_a, 2.5)), float(np.percentile(coverage_a, 97.5))),
        "b": (float(np.percentile(coverage_b, 2.5)), float(np.percentile(coverage_b, 97.5))),
        "diff": (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a configs/gbdt_*.yaml file.")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to evaluate.")
    parser.add_argument(
        "--ablation",
        default="full",
        help="Milestone 7: evaluate the model trained with --ablation <name> "
        "(scripts/train_gbdt_reranker.py). 'full' (default) is the Milestone 5/6 model.",
    )
    args = parser.parse_args()

    gbdt_config = load_gbdt_config(args.config)
    dataset_config = load_config(gbdt_config.base_config_path)

    feature_table_path = gbdt_config.feature_table_dir / f"{args.split}_features.parquet"
    if not feature_table_path.exists():
        raise FileNotFoundError(f"{feature_table_path} not found -- run scripts/train_gbdt_reranker.py first.")
    table = pd.read_parquet(feature_table_path)
    if "clicked" not in table.columns:
        raise ValueError(f"{feature_table_path} has no 'clicked' column -- this script needs a LABELED split.")

    model_dir = gbdt_config.model_dir if args.ablation == "full" else gbdt_config.model_dir / args.ablation
    model_path = model_dir / "lgbm_ranker.joblib"
    meta_path = model_dir / "model_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found -- run: python scripts/train_gbdt_reranker.py --config {args.config} "
            f"--ablation {args.ablation}"
        )
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]  # exact training column order -- never re-derive this from the table

    print(f"Dataset: {dataset_config.dataset_name}    Split: {args.split}    Ablation: {args.ablation}")
    print(f"Model: {model_path} (trained on: {meta['dataset_name']}, best_iteration={meta['best_iteration']})")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Rows: {len(table):,}    Impressions: {table['impression_id'].nunique():,}\n")

    table = table.copy()
    table["gbdt_score"] = model.predict(table[feature_names])
    score_column = {"gbdt": "gbdt_score", "semantic_baseline": "semantic_score"}

    # Same popularity/threshold definitions as scripts/evaluate_full.py,
    # reused verbatim -- head/tail and novelty must use the identical
    # convention Component 1 already established, not a re-derived one.
    store = FeatureStore(dataset_config)
    n_articles = len(store.articles)
    train_interactions = store.get_interactions("train")
    train_click_counts = (
        train_interactions.loc[train_interactions["clicked"] == 1, "article_id"].value_counts().to_dict()
    )
    novelty_model = NoveltyModel(train_click_counts, n_articles)
    popularities = sorted(train_click_counts.values())
    head_tail_threshold = popularities[len(popularities) // 2] if popularities else 0

    per_impression_metrics: dict[str, dict[str, list]] = {s: defaultdict(list) for s in SYSTEMS}
    diversity_values: dict[str, dict[str, list]] = {s: defaultdict(list) for s in SYSTEMS}
    novelty_values: dict[str, dict[str, list]] = {s: defaultdict(list) for s in SYSTEMS}
    coverage_trackers: dict[str, CoverageTracker] = {s: CoverageTracker(n_articles) for s in SYSTEMS}
    paired_values: dict[str, list[tuple[float, float]]] = defaultdict(list)  # metric -> [(gbdt, baseline), ...]
    # Per-impression top-k lists for the "all" slice only, kept so coverage
    # (a set-union statistic, not a per-impression average -- see
    # bootstrap_coverage_ci) can be bootstrapped after the main loop.
    top_k_lists: dict[str, list[list[str]]] = {s: [] for s in SYSTEMS}

    print(f"Evaluating {table['impression_id'].nunique():,} impressions ...")
    for impression_id, group in table.groupby("impression_id", sort=False):
        labels = group["clicked"].to_numpy()
        candidate_ids = group["article_id"].tolist()
        clicked_ids = set(group.loc[group["clicked"] == 1, "article_id"])
        slice_name = classify_head_tail(clicked_ids, train_click_counts, head_tail_threshold)

        results_by_system = {}
        for system in SYSTEMS:
            scores = group[score_column[system]].to_numpy()
            ranking_result = evaluate_ranking(labels, scores)
            results_by_system[system] = ranking_result

            top_k_order = np.argsort(-scores, kind="stable")[:BEYOND_ACCURACY_K]
            top_k_ids = [candidate_ids[i] for i in top_k_order]
            diversity = intra_list_diversity(top_k_ids, store.get_article_embedding)
            novelty = novelty_model.mean_score(top_k_ids)
            coverage_trackers[system].add(top_k_ids)
            top_k_lists[system].append(top_k_ids)

            for slice_key in ("all", slice_name):
                per_impression_metrics[system][slice_key].append(ranking_result)
                if diversity is not None:
                    diversity_values[system][slice_key].append(diversity)
                if novelty is not None:
                    novelty_values[system][slice_key].append(novelty)

        for metric_name in ("auc", "mrr", "ndcg@5", "ndcg@10"):
            gbdt_value = results_by_system["gbdt"][metric_name]
            baseline_value = results_by_system["semantic_baseline"][metric_name]
            if gbdt_value is not None and baseline_value is not None:
                paired_values[metric_name].append((gbdt_value, baseline_value))

    # --- Report ---
    print(f"\n{'=' * 78}\nSplit: {args.split}\n{'=' * 78}")
    for system in SYSTEMS:
        print(f"\n### {system.upper()} ###")
        print(
            f"Confined-candidate top-{BEYOND_ACCURACY_K} catalog coverage "
            f"(NOT comparable to evaluate_full.py's open-corpus coverage): "
            f"{coverage_trackers[system].coverage:.4f}"
        )
        for slice_name in ("all", "head", "tail"):
            results = per_impression_metrics[system][slice_name]
            if not results:
                print(f"  [{slice_name}] n=0")
                continue
            summary = aggregate(results)
            print(f"  --- slice: {slice_name} (n={len(results):,}) ---")
            print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
            div_values = np.array(diversity_values[system][slice_name])
            nov_values = np.array(novelty_values[system][slice_name])
            if len(div_values) > 0:
                d_lo, d_hi = bootstrap_ci(div_values, seed=dataset_config.random_seed)
                print(f"    mean diversity (confined top-{BEYOND_ACCURACY_K}): {div_values.mean():.4f}  95% CI=[{d_lo:.4f}, {d_hi:.4f}]")
            if len(nov_values) > 0:
                n_lo, n_hi = bootstrap_ci(nov_values, seed=dataset_config.random_seed)
                print(f"    mean novelty (confined top-{BEYOND_ACCURACY_K}): {nov_values.mean():.4f}  95% CI=[{n_lo:.4f}, {n_hi:.4f}]")

    print(f"\n### PAIRED BOOTSTRAP: GBDT vs. semantic baseline (delta = GBDT - semantic, slice: all) ###")
    for metric_name in ("auc", "mrr", "ndcg@5", "ndcg@10"):
        pairs = paired_values[metric_name]
        gbdt_arr = np.array([g for g, _ in pairs])
        baseline_arr = np.array([b for _, b in pairs])
        lower, upper = paired_bootstrap_ci(gbdt_arr, baseline_arr, seed=dataset_config.random_seed)
        diff = gbdt_arr.mean() - baseline_arr.mean()
        print(f"  {metric_name:8s} n={len(pairs):,}  diff={diff:+.4f}  95% CI=[{lower:+.4f}, {upper:+.4f}]", end="  ")
        if lower > 0:
            print("-> GBDT significantly better (CI excludes zero, positive).")
        elif upper < 0:
            print("-> GBDT significantly worse (CI excludes zero, negative).")
        else:
            print("-> No significant difference at 95% confidence (CI includes zero).")

    # Diversity/novelty ARE per-impression values (unlike coverage below), so
    # they can reuse paired_bootstrap_ci directly. diversity_values["gbdt"]["all"]
    # and [...]["semantic_baseline"]["all"] are index-aligned by construction:
    # a diversity/novelty value is appended for BOTH systems on the same
    # condition (len(top_k_ids) >= 2 / > 0), and both systems always share the
    # same top_k_ids length for a given impression (same candidate_ids, same
    # BEYOND_ACCURACY_K) -- so neither list can skip an impression the other keeps.
    print(f"\n### PAIRED BOOTSTRAP: beyond-accuracy, confined-candidate top-{BEYOND_ACCURACY_K} (slice: all) ###")
    for metric_name, values_dict in (("diversity", diversity_values), ("novelty", novelty_values)):
        gbdt_arr = np.array(values_dict["gbdt"]["all"])
        baseline_arr = np.array(values_dict["semantic_baseline"]["all"])
        lower, upper = paired_bootstrap_ci(gbdt_arr, baseline_arr, seed=dataset_config.random_seed)
        diff = gbdt_arr.mean() - baseline_arr.mean()
        print(f"  {metric_name:9s} n={len(gbdt_arr):,}  diff={diff:+.4f}  95% CI=[{lower:+.4f}, {upper:+.4f}]", end="  ")
        if lower > 0:
            print("-> GBDT significantly higher.")
        elif upper < 0:
            print("-> GBDT significantly lower.")
        else:
            print("-> No significant difference at 95% confidence.")

    # Coverage is NOT a per-impression value (it's a set-union over the whole
    # run), so it needs its own resampling scheme -- see bootstrap_coverage_ci.
    coverage_ci = bootstrap_coverage_ci(top_k_lists["gbdt"], top_k_lists["semantic_baseline"], n_articles)

    def _fmt_ci(ci: tuple[float, float]) -> str:
        return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"

    gbdt_cov, baseline_cov = coverage_trackers["gbdt"].coverage, coverage_trackers["semantic_baseline"].coverage
    print(
        f"  coverage  n={len(top_k_lists['gbdt']):,}  gbdt={gbdt_cov:.4f} 95% CI={_fmt_ci(coverage_ci['a'])}  "
        f"semantic={baseline_cov:.4f} 95% CI={_fmt_ci(coverage_ci['b'])}  "
        f"diff={gbdt_cov - baseline_cov:+.4f} 95% CI={_fmt_ci(coverage_ci['diff'])}"
        f" ({COVERAGE_BOOTSTRAP_RESAMPLES} resamples, fewer than the usual 1000 -- see bootstrap_coverage_ci)"
    )

    # --- Cold-start slice: reported explicitly, never fabricated ---
    # This split's feature table (loaded above) already excludes cold-start
    # impressions entirely -- train_gbdt_reranker.py's warm_impressions()
    # filters them out before any feature row is ever built, for the same
    # reason evaluate_full.py excludes them from Component 1's own
    # evaluation: BM25/semantic retrieval cannot form a query with no
    # history, so both systems being compared here collapse to the same
    # undefined behavior for these impressions. There is therefore no
    # ranking to report for this slice -- reporting its SIZE explicitly
    # (rather than letting it silently disappear) is what satisfies
    # Assignment 2's "cold-start vs. warm" slice requirement honestly.
    all_impressions = store.get_impressions(args.split)
    n_total_impressions = len(all_impressions)
    n_cold_start = int(all_impressions["is_cold_start"].sum())
    cold_start_pct = n_cold_start / n_total_impressions if n_total_impressions else 0.0
    print(f"\n### COLD-START SLICE (split: {args.split}) ###")
    print(
        f"n={n_cold_start:,} / {n_total_impressions:,} impressions ({cold_start_pct:.2%}) have empty history."
    )
    print(
        "Structurally excluded from both systems above -- BM25/semantic retrieval cannot form a "
        "query with no history (the same limitation Assignment 1's own evaluate_full.py already has), "
        "so the GBDT re-ranker built on top of that retrieval inherits it. No ranking is fabricated "
        "for this slice; see reports/engineering_log.md Milestone 5 for the full reasoning."
    )


if __name__ == "__main__":
    main()
