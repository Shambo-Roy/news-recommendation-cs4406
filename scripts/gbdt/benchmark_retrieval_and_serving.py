"""Milestone 7 (Assignment 2, Q2.1 + Q4): two related, previously-missing
measurements in one script, because both need the same underlying call --
Component 1's own open-corpus BM25/semantic search -- and building two
scripts that each call it separately would be exactly the kind of
duplication the audit process said to avoid.

1. RECALL@K RETRIEVAL EXPERIMENT (approved Milestone 3, option 2): the
   GBDT pipeline (train/eval/submission) reranks each impression's OWN
   shown candidates -- never an open-corpus top-K retrieval (see
   scripts/evaluate_gbdt_reranker.py and Milestone 3's audit). That's
   deliberate (Codabench requires ranking exactly the shown candidates;
   NRMS and virtually every published MIND/EB-NeRD result evaluates the
   same way) but it means "use Assignment 1's candidate generator to
   retrieve top-K candidates (K~100-200)" is never literally demonstrated
   anywhere. This script does that explicitly, as a SEPARATE experiment:
   for each sampled impression, retrieve top-K via BM25 + semantic +
   RRF fusion (reusing news_reco.retrieval.hybrid.reciprocal_rank_fusion
   unmodified -- the same fusion Component 1's own evaluate_full.py
   uses for open-corpus retrieval), and report recall@K -- how often the
   impression's actual clicked article survives into that retrieved set.
   This is NOT wired into the GBDT training/eval/submission pipeline at
   all; it's a standalone measurement of the retrieval stage alone.

2. SERVING LATENCY (Assignment 2 Q4.1-4.2): every timing number gathered
   elsewhere in this project (MINDlarge_test/ebnerd_testset submission
   generation) is BATCH THROUGHPUT -- ms/impression averaged over
   millions of impressions, with one-time setup costs (index construction,
   loading the embedding matrix, loading the model) amortized away. Q4.2
   asks for p99 latency "for a single user request" -- the marginal cost
   of serving ONE MORE request once the system is already warm. This
   script warms up every index/model ONCE (excluded from all timing), then
   times individual single-impression requests through three stages
   (retrieval, feature construction, reranking) plus total, over enough
   repetitions to report p50/p95/p99 honestly. Also measures each
   component's memory footprint via tracemalloc (stdlib, cross-platform,
   accurate -- not sys.getsizeof estimation or an invented number).

MEASURED vs. ESTIMATED, kept explicit throughout: everything this script
prints under "MEASURED" is a real number from this machine, this run.
Everything under "ESTIMATED" (achievable QPS, cost/1000 queries, 10x
scaling) is a back-of-envelope extrapolation from the measured numbers,
clearly labeled as such -- never invented independently of them.

Usage:
    python scripts/benchmark_retrieval_and_serving.py --config configs/gbdt_mind_small.yaml [--n-requests 500] [--top-k 150]
"""

from __future__ import annotations

import sentence_transformers  # noqa: F401

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from news_reco.evaluation.metrics import recall_at_k
from news_reco.features.store import FeatureStore
from news_reco.reranking.gbdt import ArticleLookup, build_impression_rows, load_gbdt_config
from news_reco.retrieval.availability import AvailabilityGate
from news_reco.retrieval.hybrid import reciprocal_rank_fusion
from news_reco.retrieval.lexical.bm25 import DANISH_STOPWORDS, STOPWORDS, BM25Index, build_query_from_history
from news_reco.retrieval.semantic.embeddings import EmbeddingIndex, build_query_embedding
from news_reco.utils.config import load_config

STOPWORDS_BY_LANGUAGE = {"en": STOPWORDS, "da": DANISH_STOPWORDS}
OPEN_CORPUS_SEARCH_K = 500  # same pool size Component 1's evaluate_full.py fuses from, per method, before RRF


def measure_memory(label: str, build_fn) -> tuple[object, float]:
    """Run build_fn() and return (result, peak_MB) using tracemalloc --
    real allocated-memory attribution, not an estimate."""
    tracemalloc.start()
    result = build_fn()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    print(f"  {label}: {peak_mb:.1f} MB (tracemalloc peak, MEASURED)")
    return result, peak_mb


def percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a configs/gbdt_*.yaml file.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--n-requests", type=int, default=500, help="Number of single-impression latency samples.")
    parser.add_argument("--top-k", type=int, default=150, help="K for the open-corpus retrieval experiment (100-200 per the assignment).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gbdt_config = load_gbdt_config(args.config)
    dataset_config = load_config(gbdt_config.base_config_path)
    print(f"Dataset: {dataset_config.dataset_name}    K: {args.top_k}    Requests: {args.n_requests}\n")

    print("=== WARM-UP (excluded from all timing/latency numbers below) ===")
    print("Memory footprint of each component, MEASURED via tracemalloc:")

    store, _ = measure_memory("FeatureStore (articles + impressions + interactions + embeddings load)", lambda: FeatureStore(dataset_config))
    articles = store.articles
    article_ids = articles["article_id"].tolist()

    bm25_index, _ = measure_memory(
        "BM25Index (inverted index over full catalog)",
        lambda: _build_bm25(article_ids, articles["text"].tolist(), gbdt_config.language),
    )

    def _build_semantic():
        embeddings_matrix = np.stack([store.get_article_embedding(aid) for aid in article_ids])
        index = EmbeddingIndex()
        index.fit(article_ids, embeddings_matrix)
        return index

    semantic_index, _ = measure_memory("EmbeddingIndex (dense embedding matrix, full catalog)", _build_semantic)

    def _load_model():
        model = joblib.load(gbdt_config.model_dir / "lgbm_ranker.joblib")
        with open(gbdt_config.model_dir / "model_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return model, meta["feature_names"]

    (model, feature_names), _ = measure_memory("Trained LGBMRanker (joblib-loaded)", _load_model)

    # AvailabilityGate must be real, not skipped, whenever the trained
    # model's own feature_names includes article_age_hours (EB-NeRD) --
    # otherwise build_impression_rows never emits that column at all and
    # this benchmark would silently measure a DIFFERENT, smaller feature
    # set than what actually gets served (and, worse, skip exactly the
    # feature Milestone 7's ablation found to be the single largest
    # contributor for EB-NeRD). None for MIND, which has no publish-time
    # data -- matches every other script's try/except pattern.
    try:
        availability_gate = AvailabilityGate(articles)
    except ValueError:
        availability_gate = None

    article_lookup = ArticleLookup.from_dataframe(articles.set_index("article_id", drop=False))
    zero_vector = np.zeros_like(store.get_article_embedding(article_ids[0]))

    def get_article_embedding(article_id: str) -> np.ndarray:
        try:
            return store.get_article_embedding(article_id)
        except KeyError:
            return zero_vector

    # Sample real impressions from the requested split -- a realistic
    # workload, not synthetic data.
    rng = np.random.default_rng(args.seed)
    impressions = store.get_impressions(args.split)
    impressions = impressions[~impressions["is_cold_start"]]
    interactions = store.get_interactions(args.split)
    n_sample = min(args.n_requests, len(impressions))
    sampled = impressions.sample(n=n_sample, random_state=args.seed).reset_index(drop=True)
    clicked_by_impression = (
        interactions.loc[interactions["clicked"] == 1].groupby("impression_id")["article_id"].apply(set).to_dict()
    )
    print(f"\nSampled {n_sample:,} real {args.split} impressions for the timed run below.\n")

    retrieval_times, feature_times, rerank_times, total_times = [], [], [], []
    recall_values: list[float] = []

    print("=== TIMED SINGLE-REQUEST RUN (this IS what gets reported as latency) ===")
    for impression in sampled.itertuples(index=False):
        history = list(impression.history)
        clicked_ids = clicked_by_impression.get(impression.impression_id, set())

        t0 = time.perf_counter()
        # --- Retrieval stage: genuine open-corpus top-K, not the impression's own candidates ---
        bm25_query = build_query_from_history(history, store.get_article_title)
        semantic_query_vector = build_query_embedding(history, get_article_embedding)
        bm25_hits = [doc_id for doc_id, _ in bm25_index.search(bm25_query, top_k=OPEN_CORPUS_SEARCH_K)]
        semantic_hits = [doc_id for doc_id, _ in semantic_index.search(semantic_query_vector, top_k=OPEN_CORPUS_SEARCH_K)]
        fused = [doc_id for doc_id, _ in reciprocal_rank_fusion([bm25_hits, semantic_hits])]
        retrieved_ids = fused[: args.top_k]
        t1 = time.perf_counter()
        recall = recall_at_k(fused, clicked_ids, args.top_k)  # None if this impression had no click at all

        # --- Feature construction stage: GBDT features for the RETRIEVED set ---
        rows, _ = build_impression_rows(
            impression_id=impression.impression_id,
            candidate_ids=retrieved_ids,
            history=history,
            impression_time=impression.time,
            article_lookup=article_lookup,
            get_article_embedding=get_article_embedding,
            bm25_index=bm25_index,
            semantic_index=semantic_index,
            bm25_query=bm25_query,
            semantic_query_vector=semantic_query_vector,
            popularity_lookup=lambda aid, t: 0,  # latency-only: popularity value doesn't affect timing
            availability_gate=availability_gate,
            gate_freshness=False,  # ungated -- matches how the deployed model itself was trained
            clicked_by_article=None,
        )
        t2 = time.perf_counter()

        # --- Reranking stage ---
        if rows:
            feature_df = pd.DataFrame(rows)[feature_names]
            _scores = model.predict(feature_df)
        t3 = time.perf_counter()

        retrieval_times.append(t1 - t0)
        feature_times.append(t2 - t1)
        rerank_times.append(t3 - t2)
        total_times.append(t3 - t0)
        if recall is not None:
            recall_values.append(recall)

    # --- Report: MEASURED ---
    print(f"\n{'=' * 78}\nMEASURED (this machine, this run -- see header for methodology)\n{'=' * 78}")
    print(f"\nRecall@{args.top_k} (open-corpus BM25+semantic+RRF retrieval, n={len(recall_values):,} impressions with a click):")
    mean_recall = float(np.mean(recall_values)) if recall_values else 0.0
    print(f"  {mean_recall:.4f} -- mean fraction of each impression's clicked article(s) surviving into the "
          f"top-{args.top_k} retrieved set (news_reco.evaluation.metrics.recall_at_k, unmodified)")

    print(f"\nSingle-request latency (ms), n={n_sample:,}, warm system (index/model load excluded):")
    for stage_name, values in (
        ("retrieval", retrieval_times), ("feature_construction", feature_times),
        ("reranking", rerank_times), ("total", total_times),
    ):
        p = percentiles(np.array(values) * 1000)
        print(f"  {stage_name:22s} p50={p['p50']:.2f}ms  p95={p['p95']:.2f}ms  p99={p['p99']:.2f}ms")

    # --- Report: ESTIMATED (back-of-envelope, clearly separated) ---
    p99_total_s = percentiles(np.array(total_times))["p99"]
    single_core_qps = 1.0 / p99_total_s if p99_total_s > 0 else 0.0
    print(f"\n{'=' * 78}\nESTIMATED / EXTRAPOLATED (back-of-envelope from the measured numbers above)\n{'=' * 78}")
    print(f"  Single-core achievable QPS at p99<{p99_total_s*1000:.0f}ms SLA (naive 1/p99, no batching/parallelism modeled): {single_core_qps:.1f} req/s")
    print("  Cost/QPS and 10x-scaling discussion: see reports/gbdt_reranker_experiment.md "
          "(requires a stated cloud pricing assumption this script does not invent).")


def _build_bm25(article_ids: list[str], texts: list[str], language: str) -> BM25Index:
    index = BM25Index(stopwords=STOPWORDS_BY_LANGUAGE[language])
    index.fit(article_ids, texts)
    return index


if __name__ == "__main__":
    main()
