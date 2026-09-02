"""Milestone 9: generate a NEW Codabench submission for the MIND News
Recommendation Competition using the trained GBDT re-ranker, scored against
the official MINDlarge_test set -- separate from, and never overwriting,
the existing semantic-baseline submission (scripts/generate_mind_submission.py).

Reuses, unmodified:
  - news_reco.data.mind_adapter.{build_articles_table, parse_unlabeled_behaviors}
  - news_reco.retrieval.lexical.bm25.BM25Index / news_reco.retrieval.semantic.embeddings.EmbeddingIndex
  - news_reco.reranking.gbdt.build_feature_table (the same feature logic
    trained and evaluated in Milestones 4-8)
  - news_reco.submission.{scores_to_ranks, write_prediction_file, zip_prediction_file}
  - the cached article embeddings the baseline submission already computed
    (data/interim/mind_large_test/article_embeddings.npz) -- no need to
    re-run the ~76-minute encode job a second time.

New relative to the baseline script: a fresh BM25Index built over
MINDlarge_test's own catalog (the baseline only ever needed semantic
scores; the GBDT needs bm25_score/bm25_rank too), and the full 11-feature
build_feature_table() pipeline instead of a single cosine-similarity score.

IMPORTANT LIMITATION -- article_training_click_count on this submission:
the trained model's article_training_click_count feature was learned from
MIND-SMALL's train-split clicks. MINDlarge_test is a genuinely different
(much larger) catalog, and only ~4.89% of its candidates were ever clicked
in MIND-small's train log (measured directly -- see
reports/gbdt_reranker_experiment.md, Milestone 9). The other ~95.11% of
candidates get article_training_click_count=0 by construction (the "unknown
article" convention already used everywhere in this codebase), NOT because
those articles are unpopular -- there's no way to know, since MINDlarge_test
has no labels at all and MIND-small's train population barely overlaps it.
This is an honest generalization limitation of training on MIND-small and
deploying to MINDlarge_test, not a bug, and not something a leaky shortcut
could fix (there is no other click population to draw from for the official
test set). semantic_score/bm25_score and the content/semantic-history
features are NOT affected -- they're computed fresh over MINDlarge_test's
own catalog either way, with no cross-catalog dependency.

Memory design: like scripts/generate_ebnerd_submission.py (rewritten after
a whole-machine crash in Milestone 13), this NEVER builds one feature table
for all 2.37M impressions at once (94M+ rows -- a genuine memory risk, not
a hypothetical one). Impressions are processed in fixed-size batches;
predictions are written incrementally.

Usage:
    python scripts/generate_mind_submission_gbdt.py
    python scripts/generate_mind_submission_gbdt.py --limit 2000   # smoke test
"""

from __future__ import annotations

# Must be the first import that touches torch/pandas -- see
# src/news_reco/retrieval/semantic/embeddings.py's docstring for why.
import sentence_transformers  # noqa: F401

import argparse
import json
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from news_reco.data.mind_adapter import build_articles_table, parse_unlabeled_behaviors
from news_reco.data.mind_loader import load_behaviors, load_news
from news_reco.features.store import FeatureStore
from news_reco.reranking.gbdt import build_frozen_train_click_counts, build_feature_table
from news_reco.retrieval.lexical.bm25 import STOPWORDS, BM25Index
from news_reco.retrieval.semantic.embeddings import EmbeddingIndex
from news_reco.submission import scores_to_ranks, write_prediction_file, zip_prediction_file
from news_reco.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mind_large_test" / "MINDlarge_test"
EMBEDDINGS_CACHE = PROJECT_ROOT / "data" / "interim" / "mind_large_test" / "article_embeddings.npz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mind_large_test"
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "mind_small" / "gbdt"
MIND_SMALL_CONFIG = PROJECT_ROOT / "configs" / "mind_small.yaml"
BATCH_SIZE = 10_000  # impressions per feature-table batch -- keeps peak memory bounded regardless of test-set size


def load_model() -> tuple[object, list[str]]:
    model = joblib.load(MODEL_DIR / "lgbm_ranker.joblib")
    with open(MODEL_DIR / "model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta["feature_names"]


def stream_predictions(
    impressions: pd.DataFrame,
    candidates_by_impression: dict,
    articles: pd.DataFrame,
    get_article_embedding,
    get_article_title,
    bm25_index: BM25Index,
    semantic_index: EmbeddingIndex,
    popularity_lookup,
    model,
    feature_names: list[str],
) -> Iterator[tuple[object, list[int]]]:
    """Yield (impression_id, ranks) one impression at a time, processing
    impressions in fixed-size batches so the full feature table for all
    2.37M official impressions is never built at once (see this file's
    module docstring)."""
    total_violations = 0
    n_impressions = len(impressions)

    for start in range(0, n_impressions, BATCH_SIZE):
        batch = impressions.iloc[start : start + BATCH_SIZE]
        table, violations = build_feature_table(
            batch,
            candidates_by_impression,
            articles=articles,
            get_article_embedding=get_article_embedding,
            get_article_title=get_article_title,
            bm25_index=bm25_index,
            semantic_index=semantic_index,
            popularity_lookup=popularity_lookup,
            availability_gate=None,  # MIND has no publish_time data -- no age feature, no gating possible
            gate_freshness=False,
            clicked_lookup=None,  # unlabeled official test set
        )
        total_violations += violations
        if len(table) == 0:
            continue

        scores = model.predict(table[feature_names])
        table = table.assign(_gbdt_score=scores)

        for impression_id, group in table.groupby("impression_id", sort=False):
            candidate_ids = group["article_id"].tolist()
            score_lookup = dict(zip(candidate_ids, group["_gbdt_score"]))
            yield impression_id, scores_to_ranks(candidate_ids, score_lookup)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N impressions (smoke test).")
    args = parser.parse_args()

    print("Loading trained GBDT model (MIND-small, full feature set) ...")
    model, feature_names = load_model()
    print(f"Features ({len(feature_names)}): {feature_names}")

    print("Loading MIND-small train click counts (for article_training_click_count) ...")
    small_store = FeatureStore(load_config(MIND_SMALL_CONFIG))
    train_click_counts = build_frozen_train_click_counts(small_store.get_interactions("train"))

    def popularity_lookup(article_id: str, time: pd.Timestamp) -> int:
        return train_click_counts.get(article_id, 0)

    print("Loading raw MINDlarge_test files ...")
    news = load_news(RAW_DIR / "news.tsv")
    behaviors = load_behaviors(RAW_DIR / "behaviors.tsv")
    if args.limit:
        behaviors = behaviors.head(args.limit)

    articles = build_articles_table(news)
    articles_indexed = articles.set_index("article_id", drop=False)
    print(f"Catalog: {len(articles):,} articles")

    print("Building BM25 index over MINDlarge_test's own catalog ...")
    bm25_index = BM25Index(stopwords=STOPWORDS)
    bm25_index.fit(articles["article_id"].tolist(), articles["text"].tolist())

    print(f"Loading cached embeddings from {EMBEDDINGS_CACHE} ...")
    cached = np.load(EMBEDDINGS_CACHE, allow_pickle=False)
    if list(cached["article_ids"]) != list(articles["article_id"]):
        raise ValueError(
            f"{EMBEDDINGS_CACHE} doesn't match the current catalog -- run "
            "scripts/generate_mind_submission.py first to (re)build it."
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

    print("Parsing unlabeled behaviors ...")
    parsed = parse_unlabeled_behaviors(behaviors)
    impressions = pd.DataFrame(
        {"impression_id": parsed["impression_id"], "time": behaviors["time"].to_numpy(), "history": parsed["history"]}
    )
    candidates_by_impression = dict(zip(parsed["impression_id"], parsed["candidates"]))
    print(f"Ranking {len(impressions):,} impressions in batches of {BATCH_SIZE:,} ...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = OUTPUT_DIR / "prediction_gbdt.txt"
    zip_path = OUTPUT_DIR / "mind_submission_gbdt.zip"

    predictions = stream_predictions(
        impressions,
        candidates_by_impression,
        articles_indexed,
        get_article_embedding,
        get_article_title,
        bm25_index,
        semantic_index,
        popularity_lookup,
        model,
        feature_names,
    )
    count = 0

    def counted(rows: Iterator[tuple[object, list[int]]]) -> Iterator[tuple[object, list[int]]]:
        nonlocal count
        for row in rows:
            count += 1
            yield row

    start = time.time()
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
