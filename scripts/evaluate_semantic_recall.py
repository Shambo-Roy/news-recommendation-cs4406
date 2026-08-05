"""Evaluate semantic (embedding-based) candidate generation via recall@K.

Mirrors scripts/evaluate_bm25_recall.py's structure and evaluation
protocol, so the two methods' recall@K numbers are directly comparable:
same val impressions, same non-empty-history filter, same K values.

Loads precomputed article embeddings (scripts/compute_article_embeddings.py
must have been run first) -- this script never touches sentence_transformers
or torch itself, only numpy, since EmbeddingIndex operates on already-
computed vectors. That split (compute is expensive and needs torch; search
is cheap and needs only numpy) is why this script has none of the
torch/pyarrow import-order concerns that scripts/compute_article_embeddings.py
has to work around.

For each val impression with non-empty history, builds a user profile
vector by mean-pooling the embeddings of their history articles (Q3.3),
retrieves the top-200 nearest articles by cosine similarity, and reports
recall@{50,100,200} against the articles the user actually clicked.

Usage:
    python scripts/evaluate_semantic_recall.py --config configs/mind_small.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from news_reco.evaluation.bootstrap import bootstrap_ci
from news_reco.evaluation.metrics import recall_at_k
from news_reco.features.store import FeatureStore
from news_reco.retrieval.semantic.embeddings import EmbeddingIndex, build_query_embedding
from news_reco.utils.config import PipelineConfig, load_config

K_VALUES = (50, 100, 200)
MAX_K = max(K_VALUES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()
    config: PipelineConfig = load_config(args.config)

    store = FeatureStore(config)

    print("Building embedding index over the full article catalog ...")
    articles = store.articles
    article_ids = articles["article_id"].tolist()
    embeddings = np.stack([store.get_article_embedding(aid) for aid in article_ids])
    index = EmbeddingIndex()
    index.fit(article_ids, embeddings)
    print(f"Indexed {len(article_ids):,} articles.")

    impressions = store.get_impressions("val")
    interactions = store.get_interactions("val")
    clicked_lookup = (
        interactions[interactions["clicked"] == 1].groupby("impression_id")["article_id"].apply(set)
    )

    per_impression_recall: dict[int, list] = {k: [] for k in K_VALUES}
    n_no_history = 0

    print(f"Evaluating {len(impressions):,} val impressions ...")
    for impression in impressions.itertuples(index=False):
        history = list(impression.history)
        if len(history) == 0:
            n_no_history += 1
            continue

        query_vector = build_query_embedding(history, store.get_article_embedding)
        retrieved = [doc_id for doc_id, _ in index.search(query_vector, top_k=MAX_K)]
        relevant = clicked_lookup.get(impression.impression_id, set())

        for k in K_VALUES:
            recall = recall_at_k(retrieved, relevant, k)
            if recall is not None:
                per_impression_recall[k].append(recall)

    print(f"\nImpressions evaluated: {len(impressions) - n_no_history:,}")
    print(f"Excluded (empty history, cold-start -- no query possible): {n_no_history:,}")
    print()

    for k in K_VALUES:
        values = np.array(per_impression_recall[k])
        lower, upper = bootstrap_ci(values, seed=config.random_seed)
        print(f"recall@{k}: mean={values.mean():.4f}  95% CI=[{lower:.4f}, {upper:.4f}]  n={len(values):,}")


if __name__ == "__main__":
    main()
