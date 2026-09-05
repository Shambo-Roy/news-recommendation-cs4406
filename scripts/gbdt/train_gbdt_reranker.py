"""Build the GBDT re-ranker's train/val/test feature tables from real data,
verify impression grouping (Milestone 4), and train a small LGBMRanker with
early stopping on validation nDCG (Milestone 5).

Feature tables are cached to gbdt_config.feature_table_dir -- pass
--rebuild-features to force recomputation (e.g. after changing a feature
definition in news_reco.reranking.gbdt). Building them from scratch is the
expensive step (~40 minutes for MIND-small's ~4.5M-row train table, a plain
per-impression Python loop -- see this file's own runtime log in
reports/gbdt_reranker_experiment.md); training itself is seconds to low
minutes once the tables exist.

Mirrors scripts/evaluate_full.py's setup exactly (FeatureStore, BM25Index,
EmbeddingIndex, language-selected stopwords) so the re-ranker trains against
the identical retrieval scoring Component 1 already uses -- no retrieval
logic is redefined here.

Design note on cold-start impressions: like evaluate_full.py, this EXCLUDES
impressions with empty history from every split. BM25/semantic can't form a
meaningful query for them (empty string / None vector -> uniform 0.0 scores
for every candidate, already covered by news_reco.reranking.gbdt's own
cold-start tests), and Component 1's own evaluation numbers (the ones this
work is compared against) already exclude them the same way -- keeping the
identical impression population is required by the "fair comparison"
constraint, not an independent choice made here.

Design note on popularity: train rows use ExpandingTrainPopularity (leakage-
safe, time-aware); val/test rows use the frozen whole-train click count
(safe because every train timestamp precedes every val/test timestamp) --
see news_reco.reranking.gbdt and the approved design decision in
docs/gbdt_reranker.md.

Usage:
    python scripts/train_gbdt_reranker.py --config configs/gbdt_mind_small.yaml
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import order fix -- see news_reco.retrieval.semantic.embeddings for why
# this must come before anything that imports pandas/pyarrow (FeatureStore).
import sentence_transformers  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMRanker  # noqa: E402

from news_reco.features.store import FeatureStore  # noqa: E402
from news_reco.reranking.gbdt import (  # noqa: E402
    ABLATIONS,
    COMMON_FEATURE_NAMES,
    DWELL_FEATURE_NAME,
    RECENCY_FEATURE_NAME,
    SESSION_CLICKS_SO_FAR_FEATURE_NAME,
    SESSION_IMPRESSION_INDEX_FEATURE_NAME,
    TEMPORAL_FEATURE_NAME,
    ExpandingTrainPopularity,
    GBDTConfig,
    build_feature_table,
    build_frozen_train_click_counts,
    load_ebnerd_history_detail,
    load_ebnerd_session_detail,
    load_gbdt_config,
)
from news_reco.retrieval.availability import AvailabilityGate  # noqa: E402
from news_reco.retrieval.lexical.bm25 import DANISH_STOPWORDS, STOPWORDS, BM25Index  # noqa: E402
from news_reco.retrieval.semantic.embeddings import EmbeddingIndex  # noqa: E402
from news_reco.utils.config import PipelineConfig, load_config  # noqa: E402

STOPWORDS_BY_LANGUAGE = {"en": STOPWORDS, "da": DANISH_STOPWORDS}


def load_ebnerd_history_detail_by_split(dataset_config: PipelineConfig) -> dict[str, dict] | None:
    """{"train": ..., "val": ..., "test": ...} history-detail dicts (see
    news_reco.reranking.gbdt.load_ebnerd_history_detail), or None for MIND.

    Mirrors scripts/data/clean_ebnerd.py's own raw-file mapping exactly
    (verified by reading that script, not assumed): this project's train
    split is entirely built from EB-NeRD's raw "train" behaviors+history
    pair, and BOTH this project's val and test splits are carved (by
    val_start/test_start date cutoffs) from EB-NeRD's raw "validation"
    pair -- there's no separate raw "test" history file for the small/demo
    bundles, so val and test correctly share the SAME underlying history
    detail here, exactly as they already implicitly share the same
    article-ID-only history via the common schema. Using the wrong raw
    file (e.g. train's history for a val/test impression) would attach
    timestamps belonging to a DIFFERENT snapshot of that user's history
    than the article-ID list already in play -- silently wrong, not just
    incomplete -- so this mapping is load-bearing, not a convenience.

    The bundle name ("small"/"demo") isn't stored in PipelineConfig --
    inferred from dataset_name the same way scripts/pipeline/build_pipeline.py
    already does (`dataset_name.removeprefix("ebnerd_")`), not a new
    convention invented here.
    """
    if not dataset_config.dataset_name.startswith("ebnerd_"):
        return None
    bundle = dataset_config.dataset_name.removeprefix("ebnerd_")
    bundle_dir = dataset_config.raw_dir / bundle
    train_detail = load_ebnerd_history_detail(bundle_dir / "train" / "history.parquet")
    val_detail = load_ebnerd_history_detail(bundle_dir / "validation" / "history.parquet")
    return {"train": train_detail, "val": val_detail, "test": val_detail}


def load_ebnerd_session_detail_by_split(dataset_config: PipelineConfig) -> dict[str, dict] | None:
    """{"train": ..., "val": ..., "test": ...} session-detail dicts (see
    news_reco.reranking.gbdt.load_ebnerd_session_detail), or None for MIND.

    Same raw-file-to-split mapping as load_ebnerd_history_detail_by_split,
    and for the same reason -- but reads behaviors.parquet, not
    history.parquet, since session_id only exists in the former.
    """
    if not dataset_config.dataset_name.startswith("ebnerd_"):
        return None
    bundle = dataset_config.dataset_name.removeprefix("ebnerd_")
    bundle_dir = dataset_config.raw_dir / bundle
    train_detail = load_ebnerd_session_detail(bundle_dir / "train" / "behaviors.parquet")
    val_detail = load_ebnerd_session_detail(bundle_dir / "validation" / "behaviors.parquet")
    return {"train": train_detail, "val": val_detail, "test": val_detail}


def warm_impressions(store: FeatureStore, split: str) -> pd.DataFrame:
    """Impressions for `split` with non-empty history only -- see this
    file's module docstring for why cold-start impressions are excluded."""
    impressions = store.get_impressions(split)
    return impressions[~impressions["is_cold_start"]]


def build_split_dataset(
    gbdt_config: GBDTConfig,
    store: FeatureStore,
    bm25_index: BM25Index,
    semantic_index: EmbeddingIndex,
    availability_gate: AvailabilityGate | None,
    popularity_lookup,
    split: str,
    history_detail_by_user: dict | None = None,
    session_detail_by_user: dict | None = None,
) -> tuple[pd.DataFrame, int]:
    """Assemble one split's feature table, end to end, from FeatureStore."""
    impressions = warm_impressions(store, split)
    interactions = store.get_interactions(split)

    candidates_by_impression = interactions.groupby("impression_id")["article_id"].apply(list).to_dict()
    clicked_lookup = {
        impression_id: dict(zip(group["article_id"], group["clicked"]))
        for impression_id, group in interactions.groupby("impression_id")
    }

    table, n_violations = build_feature_table(
        impressions,
        candidates_by_impression,
        articles=store.articles,
        get_article_embedding=store.get_article_embedding,
        get_article_title=store.get_article_title,
        bm25_index=bm25_index,
        semantic_index=semantic_index,
        popularity_lookup=popularity_lookup,
        availability_gate=availability_gate,
        gate_freshness=gbdt_config.gate_freshness,
        clicked_lookup=clicked_lookup,
        history_detail_by_user=history_detail_by_user,
        session_detail_by_user=session_detail_by_user,
    )
    return table, n_violations


def get_or_build_split_dataset(
    gbdt_config: GBDTConfig,
    store: FeatureStore,
    bm25_index: BM25Index,
    semantic_index: EmbeddingIndex,
    availability_gate: AvailabilityGate | None,
    popularity_lookup,
    split: str,
    rebuild: bool,
    history_detail_by_user: dict | None = None,
    session_detail_by_user: dict | None = None,
) -> tuple[pd.DataFrame, int]:
    """Load a cached split_features.parquet if present, else build + save it."""
    out_path = gbdt_config.feature_table_dir / f"{split}_features.parquet"
    if out_path.exists() and not rebuild:
        print(f"  [{split}] loading cached {out_path}")
        return pd.read_parquet(out_path), -1  # -1: violation count unknown for a cached table

    start = time.time()
    table, n_violations = build_split_dataset(
        gbdt_config, store, bm25_index, semantic_index, availability_gate, popularity_lookup, split,
        history_detail_by_user=history_detail_by_user,
        session_detail_by_user=session_detail_by_user,
    )
    elapsed = time.time() - start
    verify_impression_groups(table, split)
    print(f"           built in {elapsed:.1f}s")

    gbdt_config.feature_table_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    print(f"           wrote {out_path}\n")
    return table, n_violations


def verify_impression_groups(table: pd.DataFrame, split_name: str) -> pd.Series:
    """Assert LGBMRanker's group-size precondition holds, then report it.

    LGBMRanker requires the training matrix's rows to be grouped
    contiguously by query (impression), with a `group` array giving each
    group's size in that same row order. build_feature_table already sorts
    by impression_id (stable sort), so `sort=False` here just reads off the
    sizes in the table's own row order -- if that ever produced a
    non-contiguous impression_id (a bug), the groupby/size split would
    silently produce more groups than there are distinct impression_ids,
    which the assertion below catches.
    """
    group_sizes = table.groupby("impression_id", sort=False).size()
    assert group_sizes.sum() == len(table), "group sizes don't sum to the table's row count"
    assert group_sizes.index.is_unique, "impression_id is not contiguous -- groupby split one impression in two"

    print(f"  [{split_name}] impressions={len(group_sizes):,}  rows={len(table):,}  "
          f"candidates/impression: min={group_sizes.min()} mean={group_sizes.mean():.1f} max={group_sizes.max()}")
    return group_sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a configs/gbdt_*.yaml file.")
    parser.add_argument(
        "--rebuild-features", action="store_true", help="Recompute feature tables even if cached parquet files exist."
    )
    parser.add_argument(
        "--ablation",
        default="full",
        choices=["full"] + list(ABLATIONS),
        help="Milestone 7: train on a restricted feature subset instead of every feature. "
        "'full' (default) is the Milestone 5/6 model -- unchanged behavior, written to model_dir directly. "
        "Any other choice is written to model_dir/<ablation>/ so it never overwrites the full model.",
    )
    args = parser.parse_args()

    gbdt_config = load_gbdt_config(args.config)
    dataset_config = load_config(gbdt_config.base_config_path)

    print(f"Dataset: {dataset_config.dataset_name}  (base config: {gbdt_config.base_config_path.name})")
    store = FeatureStore(dataset_config)
    articles = store.articles
    article_ids = articles["article_id"].tolist()

    print("Building BM25 index ...")
    bm25_index = BM25Index(stopwords=STOPWORDS_BY_LANGUAGE[gbdt_config.language])
    bm25_index.fit(article_ids, articles["text"].tolist())

    print("Building embedding index ...")
    embeddings_matrix = np.stack([store.get_article_embedding(aid) for aid in article_ids])
    semantic_index = EmbeddingIndex()
    semantic_index.fit(article_ids, embeddings_matrix)

    try:
        availability_gate = AvailabilityGate(articles)
        print("Freshness data available -- article_age_hours will be computed.")
    except ValueError:
        availability_gate = None
        print("No publish_time data for this dataset -- article_age_hours is omitted (not fabricated).")

    train_impressions = store.get_impressions("train")  # includes cold-start rows on purpose here:
    train_interactions = store.get_interactions("train")  # popularity counts clicks, not impressions
    frozen_train_counts = build_frozen_train_click_counts(train_interactions)
    expanding_popularity = ExpandingTrainPopularity(train_impressions, train_interactions)

    def train_popularity_lookup(article_id: str, time: pd.Timestamp) -> int:
        return expanding_popularity.count_before(article_id, time)

    def frozen_popularity_lookup(article_id: str, time: pd.Timestamp) -> int:
        return frozen_train_counts.get(article_id, 0)

    popularity_by_split = {
        "train": train_popularity_lookup,
        "val": frozen_popularity_lookup,
        "test": frozen_popularity_lookup,
    }

    print("Loading EB-NeRD history detail (recency-decay + dwell-time features) ..." if
          dataset_config.dataset_name.startswith("ebnerd_") else
          "Not EB-NeRD -- recency-decay + dwell-time features are omitted, not fabricated.")
    history_detail_by_split = load_ebnerd_history_detail_by_split(dataset_config)

    print("Loading EB-NeRD session detail (within-session click-pattern features) ..." if
          dataset_config.dataset_name.startswith("ebnerd_") else
          "Not EB-NeRD -- within-session click-pattern features are omitted, not fabricated.")
    session_detail_by_split = load_ebnerd_session_detail_by_split(dataset_config)

    # full_feature_names is what build_feature_table actually produces (used
    # to validate the cached table below); model_feature_names is what gets
    # fed to LGBMRanker -- identical for --ablation full, a fixed subset
    # otherwise. Restricting columns, not rebuilding features, is what makes
    # an ablation run cheap: the cached parquet already has every feature.
    full_feature_names = list(COMMON_FEATURE_NAMES) + ([TEMPORAL_FEATURE_NAME] if availability_gate is not None else [])
    if history_detail_by_split is not None:
        full_feature_names += [RECENCY_FEATURE_NAME, DWELL_FEATURE_NAME]
    if session_detail_by_split is not None:
        full_feature_names += [SESSION_IMPRESSION_INDEX_FEATURE_NAME, SESSION_CLICKS_SO_FAR_FEATURE_NAME]
    model_feature_names = full_feature_names if args.ablation == "full" else ABLATIONS[args.ablation]
    print(f"\nAblation: {args.ablation}")
    print(f"Full feature set ({len(full_feature_names)}): {full_feature_names}")
    print(f"Features fed to this model ({len(model_feature_names)}): {model_feature_names}\n")

    gbdt_config.feature_table_dir.mkdir(parents=True, exist_ok=True)

    print("Building (or loading cached) feature tables and verifying impression groups:")
    tables: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        table, n_violations = get_or_build_split_dataset(
            gbdt_config,
            store,
            bm25_index,
            semantic_index,
            availability_gate,
            popularity_by_split[split],
            split,
            rebuild=args.rebuild_features,
            history_detail_by_user=history_detail_by_split[split] if history_detail_by_split is not None else None,
            session_detail_by_user=session_detail_by_split[split] if session_detail_by_split is not None else None,
        )
        if availability_gate is not None and n_violations >= 0:
            n_rows_with_age = table[TEMPORAL_FEATURE_NAME].notna().sum()
            print(
                f"           article_age_hours: {n_rows_with_age:,}/{len(table):,} defined; "
                f"{n_violations:,} temporal-availability violations (NaN'd, ungated) "
                f"[{n_violations / max(len(table), 1):.4%} of rows]"
            )
        if availability_gate is not None:
            assert (table[TEMPORAL_FEATURE_NAME].dropna() >= 0).all(), "found a negative article_age_hours"
        tables[split] = table

    print("Milestone 4 verified: feature tables built and impression-group invariants hold for all splits.\n")

    # --- Milestone 5: train a small, conservative LGBMRanker ---
    train_table, val_table = tables["train"], tables["val"]

    def to_xy_group(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        X = table[model_feature_names]
        y = table["clicked"]
        # sort=False: group sizes MUST be read off in the table's own row
        # order (already impression-contiguous, see build_feature_table),
        # not sorted by impression_id value -- LGBMRanker requires the
        # group array to describe the rows in the order they appear in X.
        group_sizes = table.groupby("impression_id", sort=False).size().to_numpy()
        return X, y, group_sizes

    X_train, y_train, group_train = to_xy_group(train_table)
    X_val, y_val, group_val = to_xy_group(val_table)
    print(f"Training rows: {len(X_train):,} across {len(group_train):,} impressions (groups)")
    print(f"Validation rows: {len(X_val):,} across {len(group_val):,} impressions (groups)")

    model = LGBMRanker(**gbdt_config.lightgbm_params)
    print(f"\nTraining LGBMRanker with params: {gbdt_config.lightgbm_params}")
    model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        eval_at=[5, 10],
        eval_metric="ndcg",
        callbacks=[lgb.early_stopping(stopping_rounds=gbdt_config.early_stopping_rounds, verbose=True)],
    )

    print(f"\nBest iteration: {model.best_iteration_}")
    print(f"Best validation scores: {model.best_score_}")

    importances = pd.Series(model.feature_importances_, index=model_feature_names).sort_values(ascending=False)
    print("\nFeature importances (gain-based split count, descending):")
    print(importances.to_string())

    # --ablation full (default) writes directly to model_dir, exactly as
    # Milestone 5/6 already did -- any other ablation goes in its own
    # subdirectory so it can never overwrite the full model's artifacts.
    output_dir = gbdt_config.model_dir if args.ablation == "full" else gbdt_config.model_dir / args.ablation
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lgbm_ranker.joblib"
    joblib.dump(model, model_path)
    meta_path = output_dir / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ablation": args.ablation,
                "feature_names": model_feature_names,
                "best_iteration": model.best_iteration_,
                "best_score": {
                    eval_set_name: {metric: float(value) for metric, value in metrics.items()}
                    for eval_set_name, metrics in model.best_score_.items()
                },
                "lightgbm_params": gbdt_config.lightgbm_params,
                "dataset_name": dataset_config.dataset_name,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {model_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
