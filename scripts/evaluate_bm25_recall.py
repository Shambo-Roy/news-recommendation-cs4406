"""Evaluate BM25 candidate generation via recall@K on real val-split data.

Builds a BM25 index over the full article catalog (title + abstract, per
Q2.1), then for each val impression with non-empty click history, builds a
query by concatenating history titles (Q2.2), retrieves the top-200
candidates, and reports recall@{50,100,200} (Q2.4): out of the articles the
user actually clicked, how many did BM25 manage to surface in its top-K?

Impressions with empty history (cold-start users) have no query to build
and are excluded, not scored as 0 -- reported separately, since it's a real
finding (this query-construction method structurally cannot serve
cold-start users at all) rather than noise to average away.

Note: retrieval searches the *full* article catalog with no time-gating --
MIND's news.tsv has no publish-date field (confirmed Milestone 2), so we
cannot enforce "only articles that existed by this impression's time." This
is a known, documented limitation, not an oversight.

Usage:
    python scripts/evaluate_bm25_recall.py --config configs/mind_small.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from news_reco.evaluation.bootstrap import bootstrap_ci
from news_reco.evaluation.metrics import recall_at_k
from news_reco.features.store import FeatureStore
from news_reco.retrieval.lexical.bm25 import BM25Index, build_query_from_history
from news_reco.utils.config import PipelineConfig, load_config

K_VALUES = (50, 100, 200)
MAX_K = max(K_VALUES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()
    config: PipelineConfig = load_config(args.config)

    store = FeatureStore(config)

    print("Building BM25 index over the full article catalog ...")
    articles = store.articles
    index = BM25Index()
    index.fit(articles["article_id"].tolist(), articles["text"].tolist())
    print(f"Indexed {len(articles):,} articles.")

    impressions = store.get_impressions("val")
    interactions = store.get_interactions("val")
    clicked_lookup = (
        interactions[interactions["clicked"] == 1].groupby("impression_id")["article_id"].apply(set)
    )

    per_impression_recall: dict[int, list] = {k: [] for k in K_VALUES}
    n_no_history = 0

    # MIND's history field is a static per-user snapshot (verified: it
    # doesn't grow across a user's later impressions within our observed
    # window), so many impressions share an identical history and therefore
    # an identical query. Caching search results per unique query text
    # avoids re-running BM25 on a query we've already scored.
    query_cache: dict[str, list[str]] = {}

    print(f"Evaluating {len(impressions):,} val impressions ...")
    for impression in impressions.itertuples(index=False):
        history = list(impression.history)
        if len(history) == 0:
            n_no_history += 1
            continue

        query = build_query_from_history(history, store.get_article_title)
        if query not in query_cache:
            query_cache[query] = [doc_id for doc_id, _ in index.search(query, top_k=MAX_K)]
        retrieved = query_cache[query]
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
