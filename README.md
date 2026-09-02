# News Recommendation — CS4.406

Reproducible news-recommendation pipeline for **MIND-small** (English) and
**EB-NeRD** (Danish, `demo`/`small`/large-test bundles), covering both
Assignment 1 (lexical + semantic retrieval) and Assignment 2 (GBDT
learning-to-rank re-ranker over behavioral click-log features, currently
under active audit against the Assignment 2 spec). Built incrementally,
milestone by milestone — see [reports/engineering_log.md](reports/engineering_log.md)
for the full narrative of design decisions, alternatives considered, bugs
found and fixed, and the results behind every piece of this.

The core pipeline (schema, temporal split, feature store, retrieval,
evaluation) is **dataset-agnostic**: a thin per-dataset loader/adapter pair
maps each dataset's raw format into one common schema, and everything
downstream — splitting, feature store, BM25, embeddings, evaluation — runs
unmodified on both datasets.

## Setup

Requires Python 3.11+ (developed against 3.12.10).

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
```

MIND-small additionally requires a Hugging Face account with access
granted to the [MIND dataset repo](https://huggingface.co/datasets/yjw1029/MIND)
(gated — register and request access there first, then run
`.venv\Scripts\huggingface-cli login` once). EB-NeRD is a public S3 bucket
and needs no authentication.

## One-command rebuild

```bash
python scripts/pipeline/build_pipeline.py --config configs/mind_small.yaml
python scripts/pipeline/build_pipeline.py --config configs/ebnerd_demo.yaml
python scripts/pipeline/build_pipeline.py --config configs/ebnerd_small.yaml
```

Downloads the dataset, cleans it into the common schema, assigns the
temporal train/val/test split with leakage checks, and materializes the
feature store — end to end, from nothing but the config file. Works for
both datasets: which stage list (and, for EB-NeRD, which bundle) to run is
inferred from the config's own `dataset_name` field, not a separate flag.
Each stage is also independently runnable (see below) and independently
tested.

## Pipeline stages

Scripts are grouped by role under `scripts/` — `data/` (acquisition +
cleaning), `pipeline/` (orchestration + dataset-agnostic prep stages),
`evaluation/` (Component 1 evaluation), `submission/` (Component 1
Codabench submissions), `gbdt/` (Component 2). See "Project layout" below.

| Stage | Script | What it does |
|---|---|---|
| Download (MIND) | `scripts/data/download_mind_small.py` | Fetch MIND-small train/dev from the authenticated Hugging Face cache |
| Download (EB-NeRD) | `scripts/data/download_ebnerd.py --bundle {demo,small}` | Fetch and extract an EB-NeRD bundle from its public S3 bucket |
| Audit | `scripts/data/audit_mind_raw.py` | Report raw MIND file shapes/date-ranges/label sanity (no transformation) |
| Clean (MIND) | `scripts/data/clean_mind.py` | Raw TSVs → common schema (`articles`, `impressions`, `interactions` parquet tables) |
| Clean (EB-NeRD) | `scripts/data/clean_ebnerd.py --bundle {demo,small}` | Raw parquet files → the same common schema |
| Split | `scripts/pipeline/split_mind.py` | Assign temporal train/val/test split; verify boundaries and no future-click leakage. Dataset-agnostic despite the name — used unmodified for EB-NeRD |
| Feature store | `scripts/pipeline/build_feature_store.py` | Attach split labels + cold-start/warm flag; materialize the feature store. Also dataset-agnostic |
| Embeddings | `scripts/pipeline/compute_article_embeddings.py --model <name>` | Encode article text with a sentence-transformers model (default `all-mpnet-base-v2` for English; `paraphrase-multilingual-mpnet-base-v2` used for Danish/EB-NeRD) |
| Eval: random baseline | `scripts/evaluation/evaluate_random_baseline.py` | Sanity-check the evaluation harness against real data with random scores |
| Eval: BM25 recall | `scripts/evaluation/evaluate_bm25_recall.py --language {en,da}` | Recall@K for lexical retrieval alone |
| Eval: semantic recall | `scripts/evaluation/evaluate_semantic_recall.py` | Recall@K for dense retrieval alone |
| Eval: full | `scripts/evaluation/evaluate_full.py --split {val,test} --language {en,da} [--gate-freshness]` | BM25 + semantic + RRF hybrid, AUC/MRR/nDCG@k, beyond-accuracy metrics (diversity, novelty, coverage), cold-start/warm and head/tail slicing, paired bootstrap significance. `--gate-freshness` restricts open-corpus retrieval to articles published by each impression's own time (EB-NeRD only, since MIND has no publish timestamps) |
| MIND Codabench submission (baseline) | `scripts/submission/generate_mind_submission.py [--limit N]` | Semantic-only baseline submission |
| EB-NeRD Codabench submission (baseline) | `scripts/submission/generate_ebnerd_submission.py [--limit N] [--resume]` | Semantic-only baseline submission |

## Datasets and configs

| Config | Dataset | Scale |
|---|---|---|
| `configs/mind_small.yaml` | MIND-small | ~51k articles, ~157k train impressions |
| `configs/ebnerd_demo.yaml` | EB-NeRD demo | ~11.8k articles, ~25k val/test impressions |
| `configs/ebnerd_small.yaml` | EB-NeRD small | ~20.7k articles, ~477k impressions |

Every config carries `val_start`/`test_start`: both datasets ship only
train + official-validation, so official validation is treated as our
held-out test, and our val is carved from train's last calendar day (see
`reports/engineering_log.md` for how these cutoffs were verified against
each dataset's real timestamps rather than assumed).

## Component 2 (Assignment 2, in progress): GBDT re-ranker

Adds a `LGBMRanker` learning-to-rank stage on top of Component 1's BM25 +
semantic scores, entirely additive (no Component 1 file modified — enforced
by a dedicated test). Currently being audited against the actual Assignment
2 spec (see `reports/engineering_log.md`'s "Assignment 2, Component 2"
section) — some design points (candidate-generation K, official-baseline
reproduction, cold-start slicing) are open questions, not yet settled.

| Stage | Script |
|---|---|
| Build features + train | `scripts/gbdt/train_gbdt_reranker.py --config configs/gbdt_{mind_small,ebnerd_small}.yaml [--ablation NAME] [--rebuild-features]` |
| Evaluate vs. baseline | `scripts/gbdt/evaluate_gbdt_reranker.py --config <same> --split {val,test} [--ablation NAME]` |
| MIND Codabench submission | `scripts/gbdt/generate_mind_submission_gbdt.py [--limit N]` |
| EB-NeRD Codabench submission | `scripts/gbdt/generate_ebnerd_submission_gbdt.py [--limit N] [--resume]` |

Requires `lightgbm`/`scikit-learn` in addition to `requirements.txt` — see
`requirements_reranker.txt`.

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

156 tests, one file per module (118 Component 1 + 38 Component 2). Tests
that need a real downloaded dataset are skipped (not failed) if it isn't
present, so the suite stays runnable on a clean checkout.

## Project layout

```
configs/            YAML configs (paths, dataset choice, temporal split cutoffs)
data/                raw/ -> interim/ -> processed/ (all gitignored, regenerated by the pipeline)
src/news_reco/
  data/               per-dataset loaders + adapters -> common schema (MIND, EB-NeRD)
  splitting/          temporal train/val/test split + leakage checks
  features/           the feature store
  retrieval/
    lexical/            BM25 (English + Danish stopwords/tokenization)
    semantic/            embeddings (sentence-transformers), brute-force cosine search
    hybrid.py             reciprocal rank fusion
    availability.py        freshness gating for open-corpus retrieval
  evaluation/          metrics (AUC, MRR, nDCG@k, recall@k), bootstrap CIs, beyond-accuracy
                        metrics, head/tail + cold-start slicing, evaluation harness
  reranking/          Component 2: GBDT feature engineering + LGBMRanker training (gbdt.py)
scripts/             thin CLI entry points, grouped by role:
  data/                 acquisition + cleaning (download_*, clean_*, audit_mind_raw.py)
  pipeline/             orchestration + dataset-agnostic prep (build_pipeline.py, split_mind.py,
                        build_feature_store.py, compute_article_embeddings.py)
  evaluation/           Component 1 evaluation (evaluate_random_baseline/bm25_recall/
                        semantic_recall/full.py)
  submission/           Component 1 Codabench submissions (generate_{mind,ebnerd}_submission.py)
  gbdt/                 Component 2 (train/evaluate_gbdt_reranker.py, generate_*_submission_gbdt.py,
                        benchmark_retrieval_and_serving.py)
tests/               one test file per module
reports/
  engineering_log.md   running log of design decisions, observations, limitations
                        (Component 1 milestones, then a separate Component 2 section)
  design_note_c1.md    Component 1's final <=4 page design note
```

## Status

**Component 1 (Assignment 1): complete.** Both datasets end to end —
acquisition, cleaning, temporal split with leakage checks, feature store,
BM25 (English + Danish), semantic retrieval (English + multilingual), RRF
hybrid, beyond-accuracy metrics, cold-start/warm and head/tail slicing,
paired bootstrap significance, freshness gating, and both Codabench
submissions (generated, validated, and scored on the live leaderboards —
see the engineering log for the official-vs-local comparison).

**Component 2 (Assignment 2): audit complete, implementation mostly
done.** A GBDT re-ranker (both datasets, full ablation studies, gated-vs-
ungated comparison, cold-start/head-tail slicing, bootstrap CIs on every
reported metric including diversity/novelty/coverage, a combined
recall@K∈[100,200] + serving-latency + memory benchmark, and both
Codabench submissions) exists and is described above — see
`reports/engineering_log.md`'s "Assignment 2, Component 2" section for
the full audit trail and every result. Resolved: NRMS is one of two
allowed re-ranker choices (GBDT vs. a small neural ranker) — GBDT was
chosen, so the existing GBDT-vs-A1-semantic-baseline comparison already
satisfies "reproduce the baseline, then beat it." Remaining work:
recency-decay/session/dwell features (held pending a design-choice
confirmation — see Milestone 1), and the design-note PDF.

| Extra Component-2 tooling | Script |
|---|---|
| Recall@K + serving latency + memory benchmark | `scripts/gbdt/benchmark_retrieval_and_serving.py --config <gbdt config> [--n-requests N] [--top-k K]` |
| Gated (anti-gaming) EB-NeRD variant | `configs/gbdt_ebnerd_small_gated.yaml` |
