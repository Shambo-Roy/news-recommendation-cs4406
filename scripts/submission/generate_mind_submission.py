"""Generate a Codabench submission (prediction.txt, zipped) for the MIND
News Recommendation Competition, scored against MINDlarge_test.

Ranks every impression's candidates by semantic similarity to a mean-
pooled history embedding -- the method chosen because it's the one with
significant, reproducible AUC gains over BM25 in every logged evaluation
on both datasets (see reports/engineering_log.md, Milestones 8 and 11-12).
There's no local accuracy to check here: the official test set is
genuinely unlabeled, so this script's only job is to produce a correctly
formatted ranking for Codabench's own scorer.

Article embeddings are computed once over MINDlarge_test's own catalog
(a different, larger set of articles than MIND-small) and cached to
data/interim/mind_large_test/article_embeddings.npz -- re-run with
--recompute-embeddings to force a fresh encode.

Usage:
    python scripts/generate_mind_submission.py
    python scripts/generate_mind_submission.py --limit 1000   # smoke test
"""

from __future__ import annotations

# Must be the first import that touches torch/pandas -- see
# src/news_reco/retrieval/semantic/embeddings.py's docstring for why
# (Windows DLL conflict between torch and pyarrow, order-dependent).
import sentence_transformers  # noqa: F401

import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from news_reco.data.mind_adapter import build_articles_table, parse_unlabeled_behaviors
from news_reco.data.mind_loader import load_behaviors, load_news
from news_reco.retrieval.semantic.embeddings import EmbeddingIndex, build_query_embedding, compute_embeddings
from news_reco.submission import scores_to_ranks, write_prediction_file, zip_prediction_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mind_large_test" / "MINDlarge_test"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "mind_large_test"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mind_large_test"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # same model used for MIND-small (Milestone 7)


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
        # .to_numpy().astype(str) (numpy-level cast, fixed-width unicode
        # dtype) -- not .astype(str).to_numpy() (pandas-level cast, stays an
        # object array of Python strings, which np.load then refuses under
        # allow_pickle=False). Same bug already hit and fixed once in
        # compute_article_embeddings.py (Milestone 7); reproduced here by
        # writing the two casts in the wrong order.
        article_ids=articles["article_id"].to_numpy().astype(str),
        embeddings=embeddings,
    )
    print(f"Wrote embeddings to {cache_path}")
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N impressions (smoke test).")
    parser.add_argument("--recompute-embeddings", action="store_true")
    args = parser.parse_args()

    print("Loading raw MINDlarge_test files ...")
    news = load_news(RAW_DIR / "news.tsv")
    behaviors = load_behaviors(RAW_DIR / "behaviors.tsv")
    if args.limit:
        behaviors = behaviors.head(args.limit)

    articles = build_articles_table(news)
    print(f"Catalog: {len(articles):,} articles")

    embeddings = get_or_compute_embeddings(articles, args.recompute_embeddings)

    index = EmbeddingIndex()
    index.fit(articles["article_id"].tolist(), embeddings)

    embedding_by_id = dict(zip(articles["article_id"], embeddings))
    zero_vector = np.zeros(embeddings.shape[1])

    def get_embedding(article_id: str) -> np.ndarray:
        return embedding_by_id.get(article_id, zero_vector)

    parsed = parse_unlabeled_behaviors(behaviors)
    print(f"Ranking {len(parsed):,} impressions ...")

    start = time.time()
    rows: list[tuple[object, list[int]]] = []
    for impression_id, history, candidates in zip(parsed["impression_id"], parsed["history"], parsed["candidates"]):
        query = build_query_embedding(history, get_embedding)
        scores = index.score_documents(query, candidates)
        ranks = scores_to_ranks(candidates, scores)
        rows.append((impression_id, ranks))
    elapsed = time.time() - start
    print(f"Ranked {len(rows):,} impressions in {elapsed:.1f}s ({elapsed / max(len(rows), 1) * 1000:.2f} ms/impression)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = OUTPUT_DIR / "prediction.txt"
    zip_path = OUTPUT_DIR / "mind_submission.zip"
    write_prediction_file(rows, txt_path)
    zip_prediction_file(txt_path, zip_path)

    print(f"\nWrote {txt_path}")
    print(f"Wrote {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        print(f"Zip contents: {zf.namelist()}")


if __name__ == "__main__":
    main()
