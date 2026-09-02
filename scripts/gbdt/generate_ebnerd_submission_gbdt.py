"""Milestone 10: generate a NEW Codabench submission for the RecSys 2024 /
EB-NeRD Challenge using the trained GBDT re-ranker, scored against the
official ebnerd_testset -- separate from, and never overwriting, the
existing semantic-baseline submission (scripts/generate_ebnerd_submission.py).

Reuses, unmodified:
  - news_reco.data.ebnerd_adapter.build_articles_table
  - news_reco.retrieval.lexical.bm25.BM25Index (Danish stopwords) /
    news_reco.retrieval.semantic.embeddings.EmbeddingIndex
  - news_reco.retrieval.availability.AvailabilityGate
  - news_reco.reranking.gbdt.build_feature_table (the same feature logic
    trained and evaluated in Milestones 4-8)
  - news_reco.submission.{scores_to_ranks, write_prediction_file, zip_prediction_file}
  - the cached article embeddings the baseline submission already computed
    (data/interim/ebnerd_testset/article_embeddings.npz)

SCALE, measured directly before committing to this design (see
reports/gbdt_reranker_experiment.md, Milestone 10): 13,536,710 impressions,
807,677 users, average history length ~145 articles/user, average ~11.7
candidates/impression. A clean 8,000-impression benchmark measured
6.05 ms/impression -- ~22.7 hours for the full run, dominated by Component
1's own BM25Index.score_documents (its cost scales with candidates x
unique-query-terms, and EB-NeRD's ~145-article average history produces a
much larger query than MIND's ~42). Explicitly approved to run at this cost
rather than compromise on feature parity with the trained model (checked
directly with the user rather than silently truncating history or dropping
BM25 features to save time).

Memory design -- the two things that caused the whole-machine crash in
Milestone 13's baseline script must both be avoided here, and this script's
job is materially bigger (11 features per candidate, not 1 cosine score):

  1. Per-user history, not per-impression: history.parquet is read ONCE
     into a dict {user_id: [article_id, ...]}, with every article_id string
     interned via sys.intern() so repeated article IDs across users' and
     impressions' histories share ONE string object instead of a fresh copy
     each time. Without interning, ~117M total history-list entries (807K
     users x ~145 articles) would each be a distinct ~50-byte string
     object -- the same failure mode as Milestone 13's crash, just shaped
     differently (per-user-once rather than per-impression, but still large
     if not deduplicated). With interning, memory is bounded by the ~125K
     DISTINCT article IDs actually in the catalog, not the ~117M list
     entries referencing them.
  2. Streamed behaviors, batched feature tables: behaviors.parquet (13.5M
     rows) is read via pyarrow row-group batches (as Milestone 13's fix
     already established for this exact file), and each batch is turned
     into its OWN bounded feature table (build_feature_table), scored, and
     written -- never one feature table for all 13.5M impressions (which
     at ~11.7 candidates/impression would be ~158M rows).

Usage:
    python scripts/generate_ebnerd_submission_gbdt.py
    python scripts/generate_ebnerd_submission_gbdt.py --limit 2000   # smoke test
"""

from __future__ import annotations

# Must be the first import that touches torch/pandas -- see
# src/news_reco/retrieval/semantic/embeddings.py's docstring for why.
import sentence_transformers  # noqa: F401

import argparse
import json
import sys
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from news_reco.data.ebnerd_adapter import build_articles_table
from news_reco.data.ebnerd_loader import load_articles
from news_reco.reranking.gbdt import build_feature_table
from news_reco.retrieval.availability import AvailabilityGate
from news_reco.retrieval.lexical.bm25 import DANISH_STOPWORDS, BM25Index
from news_reco.retrieval.semantic.embeddings import EmbeddingIndex
from news_reco.submission import scores_to_ranks, write_prediction_file, zip_prediction_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ebnerd_testset" / "ebnerd_testset"
EMBEDDINGS_CACHE = PROJECT_ROOT / "data" / "interim" / "ebnerd_testset" / "article_embeddings.npz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ebnerd_testset"
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "ebnerd_small" / "gbdt"
BEHAVIORS_BATCH_SIZE = 20_000  # impressions per streamed parquet batch AND per feature-table build


def load_model() -> tuple[object, list[str]]:
    model = joblib.load(MODEL_DIR / "lgbm_ranker.joblib")
    with open(MODEL_DIR / "model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta["feature_names"]


def build_user_history(history_path: Path) -> dict[int, list[str]]:
    """One-time, per-USER (not per-impression) history lookup.

    Every article_id is passed through sys.intern() so that the same
    article referenced from many different users' histories (popular
    articles are read by many people) shares ONE string object -- see this
    file's module docstring for why that matters at this scale.
    """
    history = pd.read_parquet(history_path, columns=["user_id", "article_id_fixed"])
    return {
        user_id: [sys.intern(str(article_id)) for article_id in article_ids]
        for user_id, article_ids in zip(history["user_id"], history["article_id_fixed"])
    }


def stream_predictions(
    behaviors_path: Path,
    user_history: dict[int, list[str]],
    articles: pd.DataFrame,
    get_article_embedding,
    get_article_title,
    bm25_index: BM25Index,
    semantic_index: EmbeddingIndex,
    availability_gate: AvailabilityGate,
    popularity_lookup,
    model,
    feature_names: list[str],
    limit: int | None,
    skip_rows: int = 0,
) -> Iterator[tuple[object, list[int]]]:
    """Yield (impression_id, ranks) one PER RAW ROW of behaviors.parquet, in
    that file's own row order, streaming it in row-group batches so the
    full 13.5M-row table (and the ~158M-row feature table it would imply)
    is never resident in memory at once.

    CRITICAL: ebnerd_testset's own "beyond accuracy" evaluation slice (its
    last 200,000 rows, flagged is_beyond_accuracy=True) all share the
    placeholder impression_id=0 -- confirmed directly against the already-
    submitted baseline's own predictions.txt, which likewise emits 200,000
    separate "0 [...]" lines, each with that ROW's own candidate list. Any
    grouping/deduplication keyed on impression_id VALUE would silently
    merge those 200,000 distinct rows into a handful -- a real bug caught
    only by comparing output line counts against the baseline's (see
    reports/gbdt_reranker_experiment.md, Milestone 10). To avoid it, this
    function groups internally by RAW ROW POSITION within each batch
    (always unique, regardless of what the impression_id column contains),
    and only reattaches the original (possibly-repeated) impression_id
    value when yielding the final (id, ranks) pair for output.

    Args:
        skip_rows: number of leading raw rows to skip entirely -- used to
            resume after an interruption by row COUNT rather than by
            impression_id (id-based dedup is unsound here for exactly the
            reason above: id=0 does not identify a unique row). Since
            output is written strictly in input row order, "N lines already
            written" and "skip the first N rows" are the same thing.
    """
    parquet_file = pq.ParquetFile(behaviors_path)
    produced = 0
    total_violations = 0
    rows_seen = 0

    for batch in parquet_file.iter_batches(
        batch_size=BEHAVIORS_BATCH_SIZE,
        columns=["impression_id", "user_id", "impression_time", "article_ids_inview"],
    ):
        batch_len = batch.num_rows
        if rows_seen + batch_len <= skip_rows:
            rows_seen += batch_len
            continue  # this whole batch was already processed in a prior run

        chunk = batch.to_pandas()
        if rows_seen < skip_rows:
            chunk = chunk.iloc[skip_rows - rows_seen :].reset_index(drop=True)
        rows_seen += batch_len

        # _row_id: 0..len(chunk)-1, unique WITHIN this batch regardless of
        # impression_id repeats -- this, not impression_id, is what
        # build_feature_table groups by internally.
        row_ids = list(range(len(chunk)))
        original_impression_id_by_row = dict(zip(row_ids, chunk["impression_id"]))

        impressions = pd.DataFrame(
            {
                "impression_id": row_ids,
                "time": chunk["impression_time"].to_numpy(),
                "history": [user_history.get(uid, []) for uid in chunk["user_id"]],
            }
        )
        candidates_by_impression = {
            row_id: [str(a) for a in inview] for row_id, inview in zip(row_ids, chunk["article_ids_inview"])
        }

        table, violations = build_feature_table(
            impressions,
            candidates_by_impression,
            articles=articles,
            get_article_embedding=get_article_embedding,
            get_article_title=get_article_title,
            bm25_index=bm25_index,
            semantic_index=semantic_index,
            popularity_lookup=popularity_lookup,
            availability_gate=availability_gate,
            gate_freshness=False,  # ungated, matching how the model itself was trained
            clicked_lookup=None,  # unlabeled official test set
        )
        total_violations += violations

        if len(table) > 0:
            scores = model.predict(table[feature_names])
            table = table.assign(_gbdt_score=scores)
            # sort=False + row_ids built in ascending 0..N-1 order together
            # guarantee this reproduces the batch's original row order
            # exactly -- required for the beyond-accuracy block, where line
            # POSITION (not the shared id=0 value) is what distinguishes
            # one row's prediction from another's.
            for row_id, group in table.groupby("impression_id", sort=False):
                candidate_ids = group["article_id"].tolist()
                score_lookup = dict(zip(candidate_ids, group["_gbdt_score"]))
                original_impression_id = original_impression_id_by_row[row_id]
                yield original_impression_id, scores_to_ranks(candidate_ids, score_lookup)
                produced += 1
                if limit is not None and produced >= limit:
                    print(f"(age violations so far: {total_violations:,})")
                    return

    print(f"Total temporal-availability violations (NaN'd, ungated): {total_violations:,}")


def count_completed_rows(txt_path: Path) -> int:
    """Count complete, well-formed lines already written to an existing,
    partially-written prediction file -- used to resume by ROW POSITION
    (see stream_predictions' `skip_rows` argument and its docstring for why
    impression_id VALUES cannot be used to identify already-done rows in
    ebnerd_testset's own data: its beyond-accuracy slice repeats id=0
    across 200,000 distinct rows).

    A line is only counted if it parses as "<int> [<int>,...]" -- an
    interrupted run can leave a truncated final line (killed mid-write),
    which must NOT be counted as completed (it will be correctly
    regenerated once row-position skipping is resumed past it).
    """
    completed = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            impression_id_str, sep, rest = line.partition(" ")
            if not sep or not line.endswith("\n") or not rest.startswith("[") or not rest.rstrip("\n").endswith("]"):
                break
            try:
                int(impression_id_str)
                [int(r) for r in rest.strip()[1:-1].split(",")]
            except ValueError:
                break
            completed += 1
    return completed


def append_prediction_file(rows: Iterator[tuple[object, list[int]]], txt_path: Path) -> None:
    """Same line format as news_reco.submission.write_prediction_file, but
    appending -- that function always opens in "w" (truncate) mode, which
    would discard already-completed work when resuming. A small, deliberate
    duplication rather than modifying Component-1 code to add a mode it
    doesn't need for its own (non-resumable) callers."""
    with open(txt_path, "a", encoding="utf-8", newline="\n") as f:
        for impression_id, ranks in rows:
            rank_str = ",".join(str(r) for r in ranks)
            f.write(f"{impression_id} [{rank_str}]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N impressions (smoke test).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip the leading rows already present (by ROW COUNT, not impression_id -- see "
        "count_completed_rows) in the existing output file and append the rest, instead of "
        "overwriting from scratch. Use after an interrupted run.",
    )
    args = parser.parse_args()

    print("Loading trained GBDT model (EB-NeRD-small, full feature set) ...")
    model, feature_names = load_model()
    print(f"Features ({len(feature_names)}): {feature_names}")

    print("Loading article catalog ...")
    articles_raw = load_articles(RAW_DIR / "articles.parquet")
    articles = build_articles_table(articles_raw)
    articles_indexed = articles.set_index("article_id", drop=False)
    print(f"Catalog: {len(articles):,} articles")

    print("Building BM25 index (Danish stopwords) ...")
    bm25_index = BM25Index(stopwords=DANISH_STOPWORDS)
    bm25_index.fit(articles["article_id"].tolist(), articles["text"].tolist())

    print(f"Loading cached embeddings from {EMBEDDINGS_CACHE} ...")
    cached = np.load(EMBEDDINGS_CACHE, allow_pickle=False)
    if list(cached["article_ids"]) != list(articles["article_id"]):
        raise ValueError(
            f"{EMBEDDINGS_CACHE} doesn't match the current catalog -- run "
            "scripts/generate_ebnerd_submission.py first to (re)build it."
        )
    embeddings = cached["embeddings"]
    embedding_by_id = dict(zip(articles["article_id"], embeddings))
    zero_vector = np.zeros(embeddings.shape[1])

    def get_article_embedding(article_id: str) -> np.ndarray:
        return embedding_by_id.get(article_id, zero_vector)

    title_by_id = dict(zip(articles["article_id"], articles["title"]))

    def get_article_title(article_id: str) -> str:
        return title_by_id.get(article_id, "")

    semantic_index = EmbeddingIndex()
    semantic_index.fit(articles["article_id"].tolist(), embeddings)

    availability_gate = AvailabilityGate(articles)  # EB-NeRD has real publish times, unlike MIND

    print("Loading article_training_click_count popularity (from EB-NeRD-small's train split) ...")
    from news_reco.features.store import FeatureStore
    from news_reco.reranking.gbdt import build_frozen_train_click_counts
    from news_reco.utils.config import load_config

    small_store = FeatureStore(load_config(PROJECT_ROOT / "configs" / "ebnerd_small.yaml"))
    train_click_counts = build_frozen_train_click_counts(small_store.get_interactions("train"))

    def popularity_lookup(article_id: str, time: pd.Timestamp) -> int:
        return train_click_counts.get(article_id, 0)

    print("Building per-user history lookup (interned, one-time) ...")
    start = time.time()
    user_history = build_user_history(RAW_DIR / "test" / "history.parquet")
    print(f"  {len(user_history):,} users in {time.time() - start:.1f}s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = OUTPUT_DIR / "predictions_gbdt.txt"
    zip_path = OUTPUT_DIR / "ebnerd_submission_gbdt.zip"

    skip_rows = 0
    if args.resume:
        if not txt_path.exists():
            raise FileNotFoundError(f"--resume given but {txt_path} doesn't exist -- nothing to resume from.")
        print(f"Counting already-completed rows in {txt_path} ...")
        skip_rows = count_completed_rows(txt_path)
        print(f"  {skip_rows:,} rows already done -- will skip that many raw rows and append the rest.")

    print("Ranking impressions (streamed, batched feature tables) ...")
    predictions = stream_predictions(
        RAW_DIR / "test" / "behaviors.parquet",
        user_history,
        articles_indexed,
        get_article_embedding,
        get_article_title,
        bm25_index,
        semantic_index,
        availability_gate,
        popularity_lookup,
        model,
        feature_names,
        args.limit,
        skip_rows=skip_rows,
    )
    count = 0

    def counted(rows: Iterator[tuple[object, list[int]]]) -> Iterator[tuple[object, list[int]]]:
        nonlocal count
        for row in rows:
            count += 1
            if count % 200_000 == 0:
                print(f"  ... {count:,} impressions ranked")
            yield row

    start = time.time()
    if args.resume:
        append_prediction_file(counted(predictions), txt_path)
    else:
        write_prediction_file(counted(predictions), txt_path)
    elapsed = time.time() - start
    print(f"Ranked {count:,} impressions in {elapsed:.1f}s ({elapsed / max(count, 1) * 1000:.2f} ms/impression)")

    zip_prediction_file(txt_path, zip_path)
    print(f"\nWrote {txt_path}")
    print(f"Wrote {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        print(f"Zip contents: {zf.namelist()}")


if __name__ == "__main__":
    main()
