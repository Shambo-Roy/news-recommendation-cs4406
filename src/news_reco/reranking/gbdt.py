"""Feature engineering for the GBDT learning-to-rank re-ranker (Component 2).

This module is entirely NEW, additive code -- it imports Component 1's
retrieval/data modules read-only (BM25Index, EmbeddingIndex,
AvailabilityGate, submission.scores_to_ranks) and never modifies their
behavior. See docs/gbdt_reranker.md for the full design writeup and
reports/gbdt_reranker_experiment.md for experiment results.

Scope of this module: turning (impression, candidate) pairs into feature
rows for LGBMRanker. Training/inference/evaluation orchestration lives in
scripts/gbdt/train_gbdt_reranker.py and scripts/gbdt/evaluate_gbdt_reranker.py, kept
separate so the feature functions here stay independently unit-testable
against small hand-crafted DataFrames (see tests/test_gbdt_reranker.py),
the same style already used for news_reco.data's adapters.

Feature families (see docs/gbdt_reranker.md for full definitions):
  retrieval   -- bm25_score, semantic_score, and each one's rank within
                 the impression's own candidate set (reuses
                 BM25Index.score_documents / EmbeddingIndex.score_documents
                 / submission.scores_to_ranks unchanged).
  content     -- title token overlap and category/subcategory match against
                 the user's click history.
  behavioral  -- history_length, article_training_click_count.
  semantic    -- max/mean cosine similarity to history article embeddings.
  temporal    -- article_age_hours (EB-NeRD only; MIND has no publish times).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from news_reco.retrieval.availability import AvailabilityGate
from news_reco.retrieval.lexical.bm25 import BM25Index, tokenize
from news_reco.retrieval.semantic.embeddings import EmbeddingIndex
from news_reco.submission import scores_to_ranks

# Feature names every dataset produces. article_age_hours is appended
# separately, only when an AvailabilityGate is available (EB-NeRD) -- MIND's
# feature table simply has one fewer column, never a fabricated value.
COMMON_FEATURE_NAMES = [
    "semantic_score",
    "bm25_score",
    "semantic_rank_within_impression",
    "bm25_rank_within_impression",
    "title_token_overlap_with_history",
    "category_match_with_history",
    "subcategory_match_with_history",
    "history_length",
    "article_training_click_count",
    "max_cosine_similarity_to_any_history_article",
    "mean_cosine_similarity_to_history_articles",
]
TEMPORAL_FEATURE_NAME = "article_age_hours"

# Feature families, named exactly as in docs/gbdt_reranker.md, used to build
# the Milestone-7 ablation study. Each ablation below is RETRIEVAL features
# plus exactly one other family -- an isolated (not cumulative) addition, so
# comparing ablation X against "retrieval_only" measures family X's own
# marginal contribution without confounding it with any other family's
# effect. "full" (all families together) is the Milestone 5/6 model, not
# retrained here -- it already IS the union of every family below.
RETRIEVAL_FEATURES = [
    "semantic_score",
    "bm25_score",
    "semantic_rank_within_impression",
    "bm25_rank_within_impression",
]
CONTENT_FEATURES = [
    "title_token_overlap_with_history",
    "category_match_with_history",
    "subcategory_match_with_history",
]
BEHAVIORAL_FEATURES = ["history_length", "article_training_click_count"]
SEMANTIC_HISTORY_FEATURES = [
    "max_cosine_similarity_to_any_history_article",
    "mean_cosine_similarity_to_history_articles",
]

# retrieval_semantic_history isn't one of the four ablations named in the
# original brief (A-E) -- added because Milestone 5's feature-importance
# result ranked the two semantic-history features #2 and #3 overall, ahead
# of every content feature, making "how much does semantic-history alone
# add over bare retrieval" a genuinely open question the brief's own B-D
# ablations don't answer on their own.
ABLATIONS: dict[str, list[str]] = {
    "retrieval_only": list(RETRIEVAL_FEATURES),
    "retrieval_content": RETRIEVAL_FEATURES + CONTENT_FEATURES,
    "retrieval_behavioral": RETRIEVAL_FEATURES + BEHAVIORAL_FEATURES,
    "retrieval_semantic_history": RETRIEVAL_FEATURES + SEMANTIC_HISTORY_FEATURES,
    # EB-NeRD only -- article_age_hours doesn't exist in MIND's feature
    # table at all (no publish timestamps), so this ablation will raise a
    # KeyError (loudly, not silently) if run against a MIND config. Added
    # after the EB-NeRD ablation run showed retrieval_only/content/
    # behavioral/semantic_history each gain ~+0.008-0.009 AUC, nowhere
    # close to the full model's +0.1283 -- and article_age_hours was the
    # single most-used feature in that full model (Milestone 8), making
    # "how much of the full model's gain is just freshness" a real,
    # previously-untested question the other four ablations can't answer.
    "retrieval_temporal": RETRIEVAL_FEATURES + [TEMPORAL_FEATURE_NAME],
}


# --------------------------------------------------------------------------
# Config -- a NEW, GBDT-only config layer. Deliberately not added to
# news_reco.utils.config.PipelineConfig (Component 1's loader): that
# dataclass's REQUIRED_KEYS/fields are Component 1's contract, and this
# config carries fields (hyperparameters, model paths) Component 1 has no
# use for. Each GBDT config instead *points at* an existing Component-1
# dataset config (mind_small.yaml / ebnerd_small.yaml) rather than
# duplicating its fields.
# --------------------------------------------------------------------------


@dataclasses.dataclass
class GBDTConfig:
    """Typed view of a configs/gbdt_*.yaml file. All paths are resolved to
    absolute paths at load time, the same convention as PipelineConfig."""

    base_config_path: Path
    language: str  # "en" or "da" -- selects BM25's stopword list, same convention as evaluate_full.py
    gate_freshness: bool
    random_seed: int
    model_dir: Path
    feature_table_dir: Path
    lightgbm_params: dict
    early_stopping_rounds: int


def load_gbdt_config(config_path: str | Path) -> GBDTConfig:
    """Read a configs/gbdt_*.yaml file into a validated GBDTConfig."""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    required_keys = (
        "base_config",
        "language",
        "gate_freshness",
        "random_seed",
        "model_dir",
        "feature_table_dir",
        "lightgbm_params",
        "early_stopping_rounds",
    )
    missing = [key for key in required_keys if key not in raw]
    if missing:
        raise ValueError(f"Config file {config_path} is missing required keys: {missing}")

    # Same anchoring convention as news_reco.utils.config.load_config: a
    # config file lives at <project_root>/configs/<name>.yaml, so the
    # project root is two levels up -- makes every path absolute and
    # independent of the caller's working directory.
    project_root = config_path.parent.parent

    return GBDTConfig(
        base_config_path=(project_root / raw["base_config"]).resolve(),
        language=raw["language"],
        gate_freshness=bool(raw["gate_freshness"]),
        random_seed=int(raw["random_seed"]),
        model_dir=(project_root / raw["model_dir"]).resolve(),
        feature_table_dir=(project_root / raw["feature_table_dir"]).resolve(),
        lightgbm_params=dict(raw["lightgbm_params"]),
        early_stopping_rounds=int(raw["early_stopping_rounds"]),
    )


# --------------------------------------------------------------------------
# Content features
# --------------------------------------------------------------------------


def title_token_overlap_with_history(candidate_title: str, history_titles: list[str]) -> float:
    """Max Jaccard token overlap between the candidate's title and each
    history article's title (tokenized via bm25.tokenize -- the same
    tokenizer BM25 itself indexes with, so "overlap" means the same thing
    here as it does to the retrieval stage).

    Jaccard = |intersection| / |union|, taken as the MAXIMUM over history
    articles (a candidate that closely echoes even ONE past click is a
    meaningful signal, even if the rest of history is unrelated) -- "max-
    over-history" is the formulation the design brief asked for first.

    Returns 0.0 (never NaN) for an empty history, a candidate title with no
    tokens, or a history article title with no tokens -- there being nothing
    to compare is a well-defined "no overlap", not a missing value.
    """
    candidate_tokens = frozenset(tokenize(candidate_title))
    history_token_sets = [frozenset(tokenize(t)) for t in history_titles]
    return max_jaccard_overlap(candidate_tokens, history_token_sets)


def max_jaccard_overlap(candidate_tokens: frozenset[str], history_token_sets: list[frozenset[str]]) -> float:
    """Same definition as title_token_overlap_with_history, but taking
    ALREADY-TOKENIZED sets -- the performance-critical path.

    Exists because EB-NeRD's history lists average ~290 articles per user
    (vs. MIND's ~30), and title_token_overlap_with_history's naive per-
    candidate re-tokenization of every history title would mean re-running
    the regex tokenizer O(candidates x 290) times PER IMPRESSION. Tokenizing
    each history title once per impression (in build_impression_rows) and
    reusing the resulting sets across every candidate turns that into O(290)
    tokenize calls per impression, with only cheap set intersection/union
    left in the O(candidates x 290) inner loop.
    """
    if not history_token_sets or not candidate_tokens:
        return 0.0

    best = 0.0
    for hist_tokens in history_token_sets:
        union = candidate_tokens | hist_tokens
        if not union:
            continue
        overlap = len(candidate_tokens & hist_tokens) / len(union)
        if overlap > best:
            best = overlap
    return best


def _as_label_set(value: object) -> set:
    """Normalize a category/subcategory cell into a hashable set of labels.

    MIND's subcategory is a single string; EB-NeRD's can be a list of
    ints (multi-label, see tests/test_ebnerd_adapter.py's fixtures). Treating
    both as a set makes *_match_with_history dataset-agnostic: "match" means
    "at least one shared label", regardless of whether a dataset represents
    that as one value or several.
    """
    if value is None:
        return set()
    if isinstance(value, (list, tuple, np.ndarray)):
        return {v for v in value if pd.notna(v)}
    if pd.isna(value):
        return set()
    return {value}


def label_match_with_history(candidate_value: object, history_values: Iterable[object]) -> float:
    """1.0 if the candidate shares at least one label with the UNION of
    labels across every history article; 0.0 otherwise (including an empty
    history, or a candidate with no label at all -- both are well-defined
    "no match" cases, not missing data).

    Used for both category_match_with_history and
    subcategory_match_with_history -- same definition, different column.
    """
    history_labels: set = set()
    for value in history_values:
        history_labels |= _as_label_set(value)
    return label_match_with_history_union(candidate_value, history_labels)


def label_match_with_history_union(candidate_value: object, history_label_union: set) -> float:
    """Same definition as label_match_with_history, but taking an ALREADY-
    COMPUTED union of history labels -- the performance-critical path.

    Same reasoning as max_jaccard_overlap: with EB-NeRD's ~290-long average
    history, recomputing the history-label union from scratch for every one
    of an impression's candidates is wasted, repeated work -- compute it
    once per impression instead (see build_impression_rows).
    """
    candidate_labels = _as_label_set(candidate_value)
    if not candidate_labels:
        return 0.0
    return 1.0 if candidate_labels & history_label_union else 0.0


# --------------------------------------------------------------------------
# Semantic-history features
# --------------------------------------------------------------------------


def semantic_history_similarity(
    candidate_embedding: np.ndarray, history_embeddings: list[np.ndarray]
) -> tuple[float, float]:
    """Max and mean cosine similarity between a candidate's embedding and
    each history article's embedding.

    Embeddings are already L2-normalized (see
    news_reco.retrieval.semantic.embeddings.compute_embeddings), so cosine
    similarity is a plain dot product -- no re-normalization needed here.

    Returns (0.0, 0.0) for an empty history, matching this project's
    existing cold-start convention: EmbeddingIndex.score_documents already
    scores a None/empty query 0.0 for everything, so a semantic-history
    feature of 0.0 for a cold-start candidate is consistent with how
    semantic_score itself behaves for the same impression.
    """
    if not history_embeddings:
        return 0.0, 0.0
    similarities = [float(candidate_embedding @ history_vector) for history_vector in history_embeddings]
    return max(similarities), sum(similarities) / len(similarities)


def semantic_history_similarity_matrix(
    candidate_embedding: np.ndarray, history_matrix: np.ndarray | None
) -> tuple[float, float]:
    """Same definition as semantic_history_similarity, but taking a
    PRE-STACKED (n_history, dim) matrix instead of a list -- the
    performance-critical path.

    Exists for the same reason as max_jaccard_overlap: EB-NeRD's ~290-long
    average history turns semantic_history_similarity's per-candidate Python
    loop of individual dot() calls into a real bottleneck (Python-level loop
    overhead dominates for many small dot products). One matrix-vector
    product (BLAS, vectorized) per candidate replaces that whole loop.
    """
    if history_matrix is None or len(history_matrix) == 0:
        return 0.0, 0.0
    similarities = history_matrix @ candidate_embedding
    return float(similarities.max()), float(similarities.mean())


def batch_semantic_history_similarity(
    candidate_matrix: np.ndarray, history_matrix: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """semantic_history_similarity_matrix for EVERY candidate in an
    impression at once -- one (n_candidates, n_history) matrix multiply
    instead of n_candidates separate matrix-vector products.

    Needed at MINDlarge_test/ebnerd_testset scale (millions of impressions):
    even one BLAS call per candidate has enough fixed Python-level overhead
    that summing it across every candidate in every impression becomes
    measurable (see reports/gbdt_reranker_experiment.md's Milestone 9/10
    timing notes). Batching to one call per IMPRESSION instead of one per
    CANDIDATE removes that per-candidate overhead entirely.

    Returns (max_similarities, mean_similarities), each shape (n_candidates,).
    A candidate row is all-zero (matching semantic_history_similarity_matrix's
    empty-history convention) when history_matrix is None/empty.
    """
    n_candidates = len(candidate_matrix)
    if history_matrix is None or len(history_matrix) == 0:
        return np.zeros(n_candidates), np.zeros(n_candidates)
    similarity_matrix = candidate_matrix @ history_matrix.T  # (n_candidates, n_history)
    return similarity_matrix.max(axis=1), similarity_matrix.mean(axis=1)


def batch_max_jaccard_overlap(
    candidate_token_sets: list[frozenset[str]], history_token_sets: list[frozenset[str]]
) -> np.ndarray:
    """max_jaccard_overlap for EVERY candidate in an impression at once, via
    one small local one-hot vocabulary + matrix multiply instead of
    n_candidates x n_history individual Python set intersections.

    Even after tokenizing history titles only once per impression (see
    max_jaccard_overlap), the remaining O(candidates x history) loop of
    Python-level set operations is still the dominant cost at
    MINDlarge_test/ebnerd_testset scale: measured at ~6-7ms/impression,
    projecting to ~4.5 hours for MINDlarge_test's 2.37M impressions and,
    proportionally, far worse for ebnerd_testset's 13.5M (see
    reports/gbdt_reranker_experiment.md). Encoding each impression's titles
    as binary presence vectors over a LOCAL vocabulary (built fresh per
    impression -- typically a few hundred distinct tokens, not the whole
    corpus) turns "count shared tokens" into one matrix multiply (BLAS),
    and "count each title's own tokens" into one row-sum -- both replace
    what was a Python-level nested loop.

    Returns an array of shape (n_candidates,): the same values
    max_jaccard_overlap would return if called once per candidate.
    """
    n_candidates = len(candidate_token_sets)
    if n_candidates == 0:
        return np.zeros(0)
    if not history_token_sets:
        return np.zeros(n_candidates)

    vocab: dict[str, int] = {}
    for token_set in candidate_token_sets:
        for token in token_set:
            vocab.setdefault(token, len(vocab))
    for token_set in history_token_sets:
        for token in token_set:
            vocab.setdefault(token, len(vocab))
    if not vocab:  # every candidate AND every history title tokenized to nothing
        return np.zeros(n_candidates)

    def _one_hot(token_sets: list[frozenset[str]]) -> np.ndarray:
        matrix = np.zeros((len(token_sets), len(vocab)), dtype=np.float32)
        for row, token_set in enumerate(token_sets):
            for token in token_set:
                matrix[row, vocab[token]] = 1.0
        return matrix

    candidate_matrix = _one_hot(candidate_token_sets)
    history_matrix = _one_hot(history_token_sets)

    intersection = candidate_matrix @ history_matrix.T  # (n_candidates, n_history): shared-token counts
    candidate_sizes = candidate_matrix.sum(axis=1, keepdims=True)  # (n_candidates, 1)
    history_sizes = history_matrix.sum(axis=1, keepdims=True).T  # (1, n_history)
    union_sizes = candidate_sizes + history_sizes - intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union_sizes > 0, intersection / union_sizes, 0.0)
    # A candidate with zero tokens has union_sizes==0 against every history
    # title (0 + h - 0 could still be >0 if h>0 -- guard explicitly instead
    # of relying on that): matches max_jaccard_overlap's "empty candidate ->
    # 0.0" rule regardless of how non-empty history is.
    empty_candidate_mask = candidate_sizes.flatten() == 0
    result = jaccard.max(axis=1)
    result[empty_candidate_mask] = 0.0
    return result


# --------------------------------------------------------------------------
# Behavioral / popularity features
# --------------------------------------------------------------------------


def build_frozen_train_click_counts(train_interactions: pd.DataFrame) -> dict[str, int]:
    """Total click count per article across the ENTIRE train split.

    Safe as article_training_click_count for VAL/TEST rows only: every train
    impression's time is strictly before val_start/test_start by
    construction (news_reco.splitting.temporal_split), so nothing here can
    depend on a val/test impression's own future. Must NOT be used for TRAIN
    rows themselves -- see ExpandingTrainPopularity, which exists precisely
    to avoid that leakage.
    """
    clicked = train_interactions.loc[train_interactions["clicked"] == 1]
    return clicked["article_id"].value_counts().to_dict()


class ExpandingTrainPopularity:
    """Per-article click counts strictly before a given time, from TRAIN
    clicks only -- the leakage-safe article_training_click_count for TRAIN
    rows (approved design decision: an early-train impression must never be
    influenced by a click that happens later in the same split).

    Implementation mirrors news_reco.retrieval.availability.AvailabilityGate
    on purpose: one sorted array of click timestamps per article, looked up
    by binary search (np.searchsorted), rather than materializing a
    per-impression snapshot dict. That choice isn't cosmetic -- this project
    already had one whole-machine crash (EB-NeRD submission generation,
    Milestone 13) from a per-impression Python structure at scale; a
    per-impression copy of a running-counts dict here would be
    O(n_impressions x n_articles_clicked_so_far), which is the same failure
    mode. This is O(total train clicks) in memory and O(log n) per lookup.
    """

    def __init__(self, train_impressions: pd.DataFrame, train_interactions: pd.DataFrame) -> None:
        clicked = train_interactions.loc[train_interactions["clicked"] == 1, ["impression_id", "article_id"]].merge(
            train_impressions[["impression_id", "time"]], on="impression_id"
        )
        self._click_times_by_article: dict[str, np.ndarray] = {
            article_id: np.sort(group["time"].to_numpy()) for article_id, group in clicked.groupby("article_id")
        }

    def count_before(self, article_id: str, time: pd.Timestamp) -> int:
        """Number of train clicks on article_id strictly before `time`.

        side="left" on a sorted array returns how many entries are < time;
        an entry exactly equal to `time` is deliberately NOT counted -- two
        events at the identical timestamp are temporally ambiguous at this
        data's resolution, not demonstrably ordered, so the conservative
        (leakage-safe) choice is to treat "at the same instant" as "not yet
        happened" rather than guess an ordering.
        """
        times = self._click_times_by_article.get(article_id)
        if times is None:
            return 0
        return int(np.searchsorted(times, np.datetime64(time), side="left"))


# --------------------------------------------------------------------------
# Retrieval features (thin wrappers -- all real work is Component 1's)
# --------------------------------------------------------------------------


def retrieval_rank_features(
    candidate_ids: list[str], bm25_scores: dict[str, float], semantic_scores: dict[str, float]
) -> tuple[list[int], list[int]]:
    """Rank (1 = best) of each candidate within its OWN impression's
    candidate set, by BM25 score and by semantic score respectively.

    Reuses news_reco.submission.scores_to_ranks unchanged -- the exact same
    function (and tie-breaking rule: stable sort on candidate_ids' original
    order) the Codabench submission itself uses to turn scores into a
    ranking, so this rank feature means the same thing operationally as what
    the semantic-only baseline submission would produce.
    """
    return scores_to_ranks(candidate_ids, bm25_scores), scores_to_ranks(candidate_ids, semantic_scores)


# --------------------------------------------------------------------------
# Per-impression row assembly
# --------------------------------------------------------------------------


@dataclasses.dataclass
class ArticleLookup:
    """Precomputed article_id -> field dicts, built ONCE per feature-table
    build (see build_feature_table.from_dataframe / ArticleLookup.from_dataframe)
    instead of repeatedly indexing a DataFrame via .loc.

    Exists purely for performance, not for a cleaner API: DataFrame.loc[id,
    column] dispatches through pandas' full indexing machinery on every
    call. Profiled as ~30% of total runtime building MINDlarge_test's
    feature table (millions of impressions, each looking up several fields
    for every candidate and every history article -- see
    reports/gbdt_reranker_experiment.md, Milestone 9). A plain dict.get()
    is a flat hash lookup with none of that overhead.
    """

    title_by_id: dict[str, str]
    category_by_id: dict[str, object]
    subcategory_by_id: dict[str, object]
    published_time_by_id: dict[str, pd.Timestamp]

    @classmethod
    def from_dataframe(cls, articles: pd.DataFrame) -> "ArticleLookup":
        return cls(
            title_by_id=articles["title"].to_dict(),
            category_by_id=articles["category"].to_dict(),
            subcategory_by_id=articles["subcategory"].to_dict(),
            published_time_by_id=articles["published_time"].to_dict(),
        )

    def __contains__(self, article_id: str) -> bool:
        return article_id in self.title_by_id


def build_impression_rows(
    *,
    impression_id: object,
    candidate_ids: list[str],
    history: list[str],
    impression_time: pd.Timestamp,
    article_lookup: ArticleLookup,
    get_article_embedding: Callable[[str], np.ndarray],
    bm25_index: BM25Index,
    semantic_index: EmbeddingIndex,
    bm25_query: str,
    semantic_query_vector: np.ndarray | None,
    popularity_lookup: Callable[[str, pd.Timestamp], int],
    availability_gate: AvailabilityGate | None,
    gate_freshness: bool,
    clicked_by_article: dict[str, int] | None,
) -> tuple[list[dict], int]:
    """Build one feature row per candidate for a single impression.

    Candidate order in the output matches `candidate_ids` order exactly,
    MINUS any candidate dropped by an explicitly enabled freshness gate
    (gate_freshness=True with an availability_gate) -- gating removes a
    not-yet-published candidate from the impression entirely, the same way
    news_reco.retrieval.availability.AvailabilityGate already does for
    open-corpus retrieval in scripts/evaluation/evaluate_full.py. When gate_freshness
    is False (the default, matching how both existing submission scripts
    behave), every candidate is kept and an unpublished-at-impression-time
    candidate gets article_age_hours = NaN instead of being dropped.

    Args:
        clicked_by_article: {article_id: 0 or 1} for this impression, or
            None when building rows for an unlabeled (official test)
            impression -- in which case no "clicked" column is emitted at
            all, so a labeled and an unlabeled table are never confused.
        popularity_lookup: article_id, time -> int. Callers pass a frozen
            dict lookup for val/test rows and ExpandingTrainPopularity for
            train rows (see docs/gbdt_reranker.md, Design Decision A).

    Returns:
        (rows, n_age_violations) -- rows is a list of plain dicts (one per
        surviving candidate); n_age_violations counts candidates that were
        NOT yet published at impression_time and gate_freshness was False
        (so they were kept with article_age_hours=NaN rather than dropped).
    """
    bm25_scores = bm25_index.score_documents(bm25_query, candidate_ids)
    semantic_scores = semantic_index.score_documents(semantic_query_vector, candidate_ids)
    bm25_ranks, semantic_ranks = retrieval_rank_features(candidate_ids, bm25_scores, semantic_scores)

    # Tokenized/unioned/stacked ONCE per impression, not once per candidate --
    # essential once history lists get long (EB-NeRD averages ~290 articles
    # per user; MIND's own history_titles-per-candidate re-tokenization was
    # fine at MIND's ~30-article average but would not scale here). See
    # max_jaccard_overlap / label_match_with_history_union /
    # semantic_history_similarity_matrix for the fast per-candidate paths
    # this precomputation feeds.
    history_titles = [article_lookup.title_by_id.get(aid, "") for aid in history]
    history_categories = [article_lookup.category_by_id.get(aid) for aid in history]
    history_subcategories = [article_lookup.subcategory_by_id.get(aid) for aid in history]
    history_embeddings = [get_article_embedding(aid) for aid in history if aid in article_lookup]

    history_token_sets = [frozenset(tokenize(t)) for t in history_titles]
    history_category_union: set = set()
    for value in history_categories:
        history_category_union |= _as_label_set(value)
    history_subcategory_union: set = set()
    for value in history_subcategories:
        history_subcategory_union |= _as_label_set(value)
    history_matrix = np.stack(history_embeddings) if history_embeddings else None

    history_length = len(history)

    eligible_ids: set[str] | None = None
    if availability_gate is not None:
        eligible_ids = availability_gate.eligible_at(impression_time)

    # Precomputed for EVERY candidate at once, before the per-candidate loop
    # (and before any freshness-gate filtering) -- one batched call per
    # impression instead of one call per candidate. At MINDlarge_test /
    # ebnerd_testset scale (millions of impressions), the per-candidate
    # Python-level overhead of even an already-vectorized single-candidate
    # call adds up to a multi-hour bottleneck (measured -- see
    # reports/gbdt_reranker_experiment.md, Milestone 9/10). Index alignment
    # with candidate_ids is preserved regardless of gating, since gating
    # only decides whether a row is EMITTED below, not how these arrays are
    # built.
    candidate_titles = [article_lookup.title_by_id.get(aid, "") for aid in candidate_ids]
    candidate_token_sets = [frozenset(tokenize(t)) for t in candidate_titles]
    candidate_embeddings = np.stack([get_article_embedding(aid) for aid in candidate_ids])

    title_overlaps = batch_max_jaccard_overlap(candidate_token_sets, history_token_sets)
    max_sims, mean_sims = batch_semantic_history_similarity(candidate_embeddings, history_matrix)

    rows: list[dict] = []
    n_age_violations = 0

    for rank_index, article_id in enumerate(candidate_ids):
        is_eligible = eligible_ids is None or article_id in eligible_ids
        if availability_gate is not None and gate_freshness and not is_eligible:
            continue  # explicitly gated out -- never reaches the reranker at all

        category = article_lookup.category_by_id.get(article_id)
        subcategory = article_lookup.subcategory_by_id.get(article_id)

        row = {
            "impression_id": impression_id,
            "article_id": article_id,
            "semantic_score": semantic_scores[article_id],
            "bm25_score": bm25_scores[article_id],
            "semantic_rank_within_impression": semantic_ranks[rank_index],
            "bm25_rank_within_impression": bm25_ranks[rank_index],
            "title_token_overlap_with_history": float(title_overlaps[rank_index]),
            "category_match_with_history": label_match_with_history_union(category, history_category_union),
            "subcategory_match_with_history": label_match_with_history_union(subcategory, history_subcategory_union),
            "history_length": history_length,
            "article_training_click_count": popularity_lookup(article_id, impression_time),
            "max_cosine_similarity_to_any_history_article": float(max_sims[rank_index]),
            "mean_cosine_similarity_to_history_articles": float(mean_sims[rank_index]),
        }

        if availability_gate is not None:
            if is_eligible:
                published_time = article_lookup.published_time_by_id.get(article_id, pd.NaT)
                age_hours = (impression_time - published_time).total_seconds() / 3600.0
                row[TEMPORAL_FEATURE_NAME] = age_hours
            else:
                row[TEMPORAL_FEATURE_NAME] = np.nan
                n_age_violations += 1

        if clicked_by_article is not None:
            row["clicked"] = clicked_by_article.get(article_id, 0)

        rows.append(row)

    return rows, n_age_violations


# --------------------------------------------------------------------------
# Full-table assembly
# --------------------------------------------------------------------------


def build_feature_table(
    impressions: pd.DataFrame,
    candidates_by_impression: dict[object, list[str]],
    *,
    articles: pd.DataFrame,
    get_article_embedding: Callable[[str], np.ndarray],
    get_article_title: Callable[[str], str],
    bm25_index: BM25Index,
    semantic_index: EmbeddingIndex,
    popularity_lookup: Callable[[str, pd.Timestamp], int],
    availability_gate: AvailabilityGate | None = None,
    gate_freshness: bool = False,
    clicked_lookup: dict[object, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Build the full feature table for a set of impressions, one row per
    surviving (impression, candidate) pair, sorted by impression_id so that
    `groupby("impression_id").size()` gives LightGBM's group-size array in
    the same row order as the table itself.

    Args:
        impressions: Must have impression_id, time, history (list[str]).
        candidates_by_impression: {impression_id: [candidate_article_ids]}.
        clicked_lookup: {impression_id: {article_id: 0/1}}, or None to build
            an unlabeled table (official Codabench test impressions).

    Returns:
        (feature_table, total_age_violations) -- see build_impression_rows
        for what an "age violation" means.
    """
    from news_reco.retrieval.lexical.bm25 import build_query_from_history
    from news_reco.retrieval.semantic.embeddings import build_query_embedding

    # Built ONCE for the whole table, not once per impression -- see
    # ArticleLookup's docstring for why (pandas .loc per-cell access
    # measured as a genuine bottleneck at MINDlarge_test/ebnerd_testset
    # scale).
    article_lookup = ArticleLookup.from_dataframe(articles)

    all_rows: list[dict] = []
    total_violations = 0

    for impression in impressions.itertuples(index=False):
        candidate_ids = candidates_by_impression.get(impression.impression_id, [])
        if not candidate_ids:
            continue

        history = list(impression.history)
        bm25_query = build_query_from_history(history, get_article_title)
        semantic_query_vector = build_query_embedding(history, get_article_embedding)

        clicked_by_article = clicked_lookup.get(impression.impression_id) if clicked_lookup is not None else None

        rows, violations = build_impression_rows(
            impression_id=impression.impression_id,
            candidate_ids=candidate_ids,
            history=history,
            impression_time=impression.time,
            article_lookup=article_lookup,
            get_article_embedding=get_article_embedding,
            bm25_index=bm25_index,
            semantic_index=semantic_index,
            bm25_query=bm25_query,
            semantic_query_vector=semantic_query_vector,
            popularity_lookup=popularity_lookup,
            availability_gate=availability_gate,
            gate_freshness=gate_freshness,
            clicked_by_article=clicked_by_article,
        )
        all_rows.extend(rows)
        total_violations += violations

    feature_names = list(COMMON_FEATURE_NAMES)
    if availability_gate is not None:
        feature_names.append(TEMPORAL_FEATURE_NAME)
    columns = ["impression_id", "article_id"] + feature_names + (["clicked"] if clicked_lookup is not None else [])

    if not all_rows:
        return pd.DataFrame(columns=columns), total_violations

    table = pd.DataFrame(all_rows)[columns]
    table = table.sort_values("impression_id", kind="stable").reset_index(drop=True)
    return table, total_violations
