"""Generate a Codabench submission (predictions.txt, zipped) for the RecSys
2024 / EB-NeRD Challenge, scored against the official ebnerd_testset.

Ranks every impression's candidates by semantic similarity to a mean-
pooled history embedding -- the method chosen because it's the one with
significant, reproducible AUC gains over BM25 in every logged evaluation
on both datasets (see reports/engineering_log.md, Milestones 8 and 11-12).
There's no local accuracy to check here: the official test set is
genuinely unlabeled, so this script's only job is to produce a correctly
formatted ranking for Codabench's own scorer.

Memory design (rewritten after the first version crashed the whole
machine, not just the process -- see reports/engineering_log.md): the
official test set has 13.5M impressions and an 807K-row history file.
Building a Python history-ID list *per impression* (even though most
impressions share a user, and users' history doesn't vary within this
snapshot) meant materializing on the order of a billion small Python
string objects. Fixed by precomputing one mean-pooled query embedding
*per user* up front (a flat float32 matrix, not per-impression lists),
and by streaming behaviors.parquet in row-group batches via pyarrow and
writing predictions incrementally, rather than ever holding the full
13.5M-row table or the full output in memory at once.

Article embeddings are computed once over ebnerd_testset's own catalog
(a different set of articles than the demo/small bundles) and cached to
data/interim/ebnerd_testset/article_embeddings.npz -- re-run with
--recompute-embeddings to force a fresh encode.

Usage:
    python scripts/generate_ebnerd_submission.py
    python scripts/generate_ebnerd_submission.py --limit 1000   # smoke test
"""

from __future__ import annotations

# Must be the first import that touches torch/pandas -- see
# src/news_reco/retrieval/semantic/embeddings.py's docstring for why
# (Windows DLL conflict between torch and pyarrow, order-dependent).
import sentence_transformers  # noqa: F401

import argparse
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from news_reco.data.ebnerd_adapter import build_articles_table
from news_reco.data.ebnerd_loader import load_articles
from news_reco.retrieval.semantic.embeddings import EmbeddingIndex, compute_embeddings
from news_reco.submission import scores_to_ranks, write_prediction_file, zip_prediction_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ebnerd_testset" / "ebnerd_testset"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "ebnerd_testset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ebnerd_testset"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  # same model used for EB-NeRD demo/small (Milestone 10)
BEHAVIORS_BATCH_SIZE = 200_000  # impressions per streamed chunk -- keeps peak memory flat regardless of file size


def get_or_compute_embeddings(articles: pd.DataFrame, recompute: bool) -> np.ndarray:
    cache_path = INTERIM_DIR / "article_embeddings.npz"
    if cache_path.exists() and not recompute:
        print(f"Loading cached embeddings from {cache_path} ...")
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["article_ids"]
        if list(cached_ids) == list(articles["article_id"]):
            return cached["embeddings"]
        print("Cached article_ids don't match the current catalog -- recomputing.")

    print(f"Encoding {len(articles):,} articles with {MODEL_NAME} ...")
    embeddings = compute_embeddings(articles["text"].tolist(), model_name=MODEL_NAME)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        # .to_numpy().astype(str), not .astype(str).to_numpy() -- numpy-level
        # cast gives a fixed-width dtype np.load can read back under
        # allow_pickle=False; pandas-level cast stays an object array and
        # gets refused. (Real bug hit once already -- see generate_mind_submission.py.)
        article_ids=articles["article_id"].to_numpy().astype(str),
        embeddings=embeddings,
    )
    print(f"Wrote embeddings to {cache_path}")
    return embeddings


def build_user_query_matrix(
    history_path: Path, doc_id_to_index: dict[str, int], article_embeddings: np.ndarray
) -> tuple[dict[int, int], np.ndarray]:
    """Precompute one mean-pooled query embedding per user, up front.

    Returns (user_id -> row index, float32 matrix of shape (n_users, dim)).
    A user with no history article present in the catalog gets an
    all-zero row -- EmbeddingIndex.score_documents already treats a
    zero-norm query as "score everything 0.0" (its cold-start convention),
    so no special-casing is needed at scoring time.

    Only the two columns actually needed are read from history.parquet --
    it also carries impression_time_fixed, scroll_percentage_fixed, and
    read_time_fixed arrays that this script never uses, and reading the
    full file was a meaningful chunk of the original memory blowup.
    """
    history = pd.read_parquet(history_path, columns=["user_id", "article_id_fixed"])
    dim = article_embeddings.shape[1]
    n_users = len(history)
    matrix = np.zeros((n_users, dim), dtype=np.float32)

    for row_index, article_ids in enumerate(history["article_id_fixed"]):
        indices = [doc_id_to_index[str(a)] for a in article_ids if str(a) in doc_id_to_index]
        if indices:
            matrix[row_index] = article_embeddings[indices].mean(axis=0)

    user_id_to_row = dict(zip(history["user_id"].to_numpy(), range(n_users)))
    return user_id_to_row, matrix


def stream_predictions(
    behaviors_path: Path,
    user_id_to_row: dict[int, int],
    user_query_matrix: np.ndarray,
    index: EmbeddingIndex,
    limit: int | None,
) -> Iterator[tuple[object, list[int]]]:
    """Yield (impression_id, ranks) one impression at a time, reading
    behaviors.parquet in row-group batches so the full 13.5M-row table is
    never resident in memory at once."""
    zero_query = np.zeros(user_query_matrix.shape[1], dtype=np.float32)
    parquet_file = pq.ParquetFile(behaviors_path)
    produced = 0

    for batch in parquet_file.iter_batches(
        batch_size=BEHAVIORS_BATCH_SIZE, columns=["impression_id", "user_id", "article_ids_inview"]
    ):
        chunk = batch.to_pandas()
        for impression_id, user_id, inview in zip(chunk["impression_id"], chunk["user_id"], chunk["article_ids_inview"]):
            row = user_id_to_row.get(user_id)
            query = user_query_matrix[row] if row is not None else zero_query
            candidates = [str(a) for a in inview]
            scores = index.score_documents(query, candidates)
            yield impression_id, scores_to_ranks(candidates, scores)

            produced += 1
            if limit is not None and produced >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N impressions (smoke test).")
    parser.add_argument("--recompute-embeddings", action="store_true")
    args = parser.parse_args()

    print("Loading article catalog ...")
    articles_raw = load_articles(RAW_DIR / "articles.parquet")
    articles = build_articles_table(articles_raw)
    print(f"Catalog: {len(articles):,} articles")

    embeddings = get_or_compute_embeddings(articles, args.recompute_embeddings)

    index = EmbeddingIndex()
    index.fit(articles["article_id"].tolist(), embeddings)
    doc_id_to_index = {doc_id: i for i, doc_id in enumerate(articles["article_id"])}

    print("Precomputing per-user query embeddings ...")
    start = time.time()
    user_id_to_row, user_query_matrix = build_user_query_matrix(
        RAW_DIR / "test" / "history.parquet", doc_id_to_index, embeddings
    )
    print(f"  {len(user_id_to_row):,} users in {time.time() - start:.1f}s")

    print("Ranking impressions (streamed) ...")
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = OUTPUT_DIR / "predictions.txt"
    zip_path = OUTPUT_DIR / "ebnerd_submission.zip"

    predictions = stream_predictions(
        RAW_DIR / "test" / "behaviors.parquet", user_id_to_row, user_query_matrix, index, args.limit
    )
    count = 0

    def counted(rows: Iterator[tuple[object, list[int]]]) -> Iterator[tuple[object, list[int]]]:
        nonlocal count
        for row in rows:
            count += 1
            yield row

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
