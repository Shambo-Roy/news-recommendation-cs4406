"""Full offline evaluation: BM25 vs. semantic vs. RRF hybrid.

For every non-cold-start impression in the given split:
  - AUC / MRR / nDCG@5 / nDCG@10 over the impression's *actual shown*
    candidates (re-scored by each method), via the Milestone 5 harness.
  - recall@{50,100,200} over *open-corpus* retrieval (Milestone 6/7's
    protocol), against the articles the user actually clicked.
  - Beyond-accuracy metrics (intra-list diversity, novelty, catalog
    coverage) over each method's open-corpus top-10 recommendation list.
  - Sliced by head/tail (was the clicked article popular in train, or not)
    and reported separately for the cold-start impressions this pipeline
    structurally cannot serve at all.

The hybrid method combines BM25 and semantic via Reciprocal Rank Fusion
(RRF): open-corpus hybrid retrieval fuses each method's top-500 ranked
list; candidate-set hybrid scoring fuses each method's *local* ranking of
just the impression's actual candidates (cheap, no full-catalog rank
lookup needed -- see news_reco.retrieval.hybrid.fuse_score_dicts).

Ends with a paired bootstrap comparison of BM25 vs. semantic on AUC --
the statistically rigorous version of the recall@K CIs from Milestones 6-7,
which only supported an *independent*-CI, non-overlap argument.

Usage:
    python scripts/evaluate_full.py --config configs/mind_small.yaml --split val
    python scripts/evaluate_full.py --config configs/mind_small.yaml --split val --limit 200  # quick smoke test
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import order fix -- see news_reco.retrieval.semantic.embeddings for why
# this must come before anything that imports pandas/pyarrow (FeatureStore,
# below). This script doesn't call compute_embeddings() (embeddings are
# already cached to disk), but the import itself is enough to trigger the
# conflict if pandas loads first, so we still need this guard.
import sentence_transformers  # noqa: E402,F401

import argparse  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

from news_reco.evaluation.beyond_accuracy import (  # noqa: E402
    CoverageTracker,
    NoveltyModel,
    intra_list_diversity,
)
from news_reco.evaluation.bootstrap import paired_bootstrap_ci  # noqa: E402
from news_reco.evaluation.harness import aggregate, evaluate_ranking  # noqa: E402
from news_reco.evaluation.metrics import recall_at_k  # noqa: E402
from news_reco.evaluation.slicing import classify_head_tail  # noqa: E402
from news_reco.features.store import FeatureStore  # noqa: E402
from news_reco.retrieval.hybrid import fuse_score_dicts, reciprocal_rank_fusion  # noqa: E402
from news_reco.retrieval.lexical.bm25 import BM25Index, build_query_from_history  # noqa: E402
from news_reco.retrieval.semantic.embeddings import EmbeddingIndex, build_query_embedding  # noqa: E402
from news_reco.utils.config import PipelineConfig, load_config  # noqa: E402

K_VALUES = (50, 100, 200)
OPEN_CORPUS_POOL = 500  # top-N each method contributes to hybrid open-corpus RRF fusion
BEYOND_ACCURACY_K = 10  # recommendation-list size for diversity/novelty/coverage
METHODS = ("bm25", "semantic", "hybrid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N impressions (smoke test).")
    args = parser.parse_args()
    config: PipelineConfig = load_config(args.config)

    store = FeatureStore(config)
    articles = store.articles
    article_ids = articles["article_id"].tolist()
    n_articles = len(article_ids)

    print("Building BM25 index ...")
    bm25 = BM25Index()
    bm25.fit(article_ids, articles["text"].tolist())

    print("Building embedding index ...")
    embeddings_matrix = np.stack([store.get_article_embedding(aid) for aid in article_ids])
    semantic = EmbeddingIndex()
    semantic.fit(article_ids, embeddings_matrix)

    # Popularity model from TRAIN clicks only (never the split being
    # evaluated) -- used for both novelty scoring and the head/tail
    # threshold, so neither metric is circular with respect to the data
    # it's measuring.
    train_interactions = store.get_interactions("train")
    train_click_counts = (
        train_interactions.loc[train_interactions["clicked"] == 1, "article_id"].value_counts().to_dict()
    )
    novelty_model = NoveltyModel(train_click_counts, n_articles)
    popularities = sorted(train_click_counts.values())
    head_tail_threshold = popularities[len(popularities) // 2] if popularities else 0
    print(f"Head/tail threshold (median train click count among ever-clicked articles): {head_tail_threshold}")

    impressions = store.get_impressions(args.split)
    if args.limit is not None:
        impressions = impressions.head(args.limit)
    interactions = store.get_interactions(args.split)
    candidates_by_impression = interactions.groupby("impression_id")
    clicked_lookup = interactions.loc[interactions["clicked"] == 1].groupby("impression_id")["article_id"].apply(set)

    # method -> slice -> list of per-impression evaluate_ranking() dicts
    per_impression_metrics: dict[str, dict[str, list]] = {m: defaultdict(list) for m in METHODS}
    # method -> k -> slice -> list of recall values
    recall_values: dict[str, dict[int, dict[str, list]]] = {
        m: {k: defaultdict(list) for k in K_VALUES} for m in METHODS
    }
    diversity_values: dict[str, dict[str, list]] = {m: defaultdict(list) for m in METHODS}
    novelty_values: dict[str, dict[str, list]] = {m: defaultdict(list) for m in METHODS}
    coverage_trackers: dict[str, CoverageTracker] = {m: CoverageTracker(n_articles) for m in METHODS}

    bm25_query_cache: dict[str, list] = {}
    semantic_query_cache: dict[tuple, list] = {}

    n_cold_start = 0
    n_evaluated = 0

    print(f"Evaluating {len(impressions):,} {args.split} impressions ...")
    for impression in impressions.itertuples(index=False):
        history = list(impression.history)
        impression_id = impression.impression_id

        if len(history) == 0:
            n_cold_start += 1
            continue

        candidates_df = candidates_by_impression.get_group(impression_id)
        candidate_ids = candidates_df["article_id"].tolist()
        labels = candidates_df["clicked"].to_numpy()
        clicked_ids = clicked_lookup.get(impression_id, set())

        # Every impression evaluated here is, by construction, non-cold-start
        # ("warm") -- see n_cold_start above for that slice's own report.
        slices_for_this_impression = ["all", classify_head_tail(clicked_ids, train_click_counts, head_tail_threshold)]

        bm25_query = build_query_from_history(history, store.get_article_title)
        if bm25_query not in bm25_query_cache:
            bm25_query_cache[bm25_query] = bm25.search(bm25_query, top_k=OPEN_CORPUS_POOL)
        bm25_open_ids = [doc_id for doc_id, _ in bm25_query_cache[bm25_query]]
        bm25_candidate_scores = bm25.score_documents(bm25_query, candidate_ids)

        semantic_query_vector = build_query_embedding(history, store.get_article_embedding)
        semantic_query_key = tuple(sorted(history))  # mean-pooling is order-invariant -> valid cache key
        if semantic_query_key not in semantic_query_cache:
            semantic_query_cache[semantic_query_key] = semantic.search(semantic_query_vector, top_k=OPEN_CORPUS_POOL)
        semantic_open_ids = [doc_id for doc_id, _ in semantic_query_cache[semantic_query_key]]
        semantic_candidate_scores = semantic.score_documents(semantic_query_vector, candidate_ids)

        hybrid_open_ids = [doc_id for doc_id, _ in reciprocal_rank_fusion([bm25_open_ids, semantic_open_ids])]
        hybrid_candidate_scores = fuse_score_dicts([bm25_candidate_scores, semantic_candidate_scores])

        method_data = {
            "bm25": (np.array([bm25_candidate_scores[c] for c in candidate_ids]), bm25_open_ids),
            "semantic": (np.array([semantic_candidate_scores[c] for c in candidate_ids]), semantic_open_ids),
            "hybrid": (np.array([hybrid_candidate_scores[c] for c in candidate_ids]), hybrid_open_ids),
        }

        for method_name, (scores_array, open_ids) in method_data.items():
            ranking_result = evaluate_ranking(labels, scores_array)
            top_k_list = open_ids[:BEYOND_ACCURACY_K]
            diversity = intra_list_diversity(top_k_list, store.get_article_embedding)
            novelty = novelty_model.mean_score(top_k_list)
            coverage_trackers[method_name].add(top_k_list)

            for slice_name in slices_for_this_impression:
                per_impression_metrics[method_name][slice_name].append(ranking_result)
                for k in K_VALUES:
                    recall = recall_at_k(open_ids, clicked_ids, k)
                    if recall is not None:
                        recall_values[method_name][k][slice_name].append(recall)
                if diversity is not None:
                    diversity_values[method_name][slice_name].append(diversity)
                if novelty is not None:
                    novelty_values[method_name][slice_name].append(novelty)

        n_evaluated += 1

    # --- Report ---
    print(f"\n{'=' * 70}\nSplit: {args.split}    Evaluated: {n_evaluated:,}\n{'=' * 70}")

    for method_name in METHODS:
        print(f"\n### {method_name.upper()} ###")
        print(f"Catalog coverage (top-{BEYOND_ACCURACY_K} lists, union across all evaluated impressions): "
              f"{coverage_trackers[method_name].coverage:.4f}")
        for slice_name in ("all", "head", "tail"):
            results = per_impression_metrics[method_name][slice_name]
            if len(results) == 0:
                print(f"  [{slice_name}] n=0")
                continue
            summary = aggregate(results)
            print(f"  --- slice: {slice_name} (n={len(results):,}) ---")
            print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
            for k in K_VALUES:
                values = np.array(recall_values[method_name][k][slice_name])
                if len(values) > 0:
                    print(f"    recall@{k}: {values.mean():.4f} (n={len(values):,})")
            div_values = np.array(diversity_values[method_name][slice_name])
            nov_values = np.array(novelty_values[method_name][slice_name])
            if len(div_values) > 0:
                print(f"    mean diversity (top-{BEYOND_ACCURACY_K}): {div_values.mean():.4f}")
            if len(nov_values) > 0:
                print(f"    mean novelty (top-{BEYOND_ACCURACY_K}): {nov_values.mean():.4f}")

    print(f"\n### COLD-START SLICE ###")
    print(f"n={n_cold_start:,} impressions with empty history -- structurally excluded from every method")
    print("here (BM25/semantic/hybrid all require non-empty history to build a query at all).")

    print(f"\n### PAIRED BOOTSTRAP: semantic vs. BM25 (metric: AUC, slice: all) ###")
    bm25_all = per_impression_metrics["bm25"]["all"]
    semantic_all = per_impression_metrics["semantic"]["all"]
    paired_auc = [
        (r_s["auc"], r_b["auc"])
        for r_s, r_b in zip(semantic_all, bm25_all)
        if r_s["auc"] is not None and r_b["auc"] is not None
    ]
    semantic_auc = np.array([s for s, _ in paired_auc])
    bm25_auc = np.array([b for _, b in paired_auc])
    lower, upper = paired_bootstrap_ci(semantic_auc, bm25_auc, seed=config.random_seed)
    diff = semantic_auc.mean() - bm25_auc.mean()
    print(f"n={len(paired_auc):,}  mean AUC diff (semantic - bm25) = {diff:.4f}  95% CI=[{lower:.4f}, {upper:.4f}]")
    if lower > 0:
        print("Semantic is significantly better than BM25 (CI excludes zero, entirely positive).")
    elif upper < 0:
        print("BM25 is significantly better than semantic (CI excludes zero, entirely negative).")
    else:
        print("No significant difference at 95% confidence (CI includes zero).")


if __name__ == "__main__":
    main()
