# News Recommendation — CS4.406 Assignment 1

Reproducible lexical + semantic retrieval pipeline for news recommendation,
built on two datasets: **MIND-small** (English) and **EB-NeRD** (Danish,
`demo` and `small` bundles). Built incrementally, milestone by milestone —
see [reports/engineering_log.md](reports/engineering_log.md) for the full
narrative of design decisions, alternatives considered, bugs found and
fixed, and the results behind every piece of this.

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

## One-command rebuild (MIND)

```bash
python scripts/build_pipeline.py --config configs/mind_small.yaml
```

Downloads MIND-small, cleans it into the common schema, assigns the
temporal train/val/test split with leakage checks, and materializes the
feature store — end to end, from nothing but the config file. Each stage
is also independently runnable (see below) and independently tested.

EB-NeRD doesn't have a chained one-command script yet (its extra `--bundle
{demo,small}` flag on the download/clean stages doesn't fit the current
`build_pipeline.py` shape) — run its stages individually as shown below.

## Pipeline stages

| Stage | Script | What it does |
|---|---|---|
| Download (MIND) | `scripts/download_mind_small.py` | Fetch MIND-small train/dev from the authenticated Hugging Face cache |
| Download (EB-NeRD) | `scripts/download_ebnerd.py --bundle {demo,small}` | Fetch and extract an EB-NeRD bundle from its public S3 bucket |
| Audit | `scripts/audit_mind_raw.py` | Report raw MIND file shapes/date-ranges/label sanity (no transformation) |
| Clean (MIND) | `scripts/clean_mind.py` | Raw TSVs → common schema (`articles`, `impressions`, `interactions` parquet tables) |
| Clean (EB-NeRD) | `scripts/clean_ebnerd.py --bundle {demo,small}` | Raw parquet files → the same common schema |
| Split | `scripts/split_mind.py` | Assign temporal train/val/test split; verify boundaries and no future-click leakage. Dataset-agnostic despite the name — used unmodified for EB-NeRD |
| Feature store | `scripts/build_feature_store.py` | Attach split labels + cold-start/warm flag; materialize the feature store. Also dataset-agnostic |
| Embeddings | `scripts/compute_article_embeddings.py --model <name>` | Encode article text with a sentence-transformers model (default `all-mpnet-base-v2` for English; `paraphrase-multilingual-mpnet-base-v2` used for Danish/EB-NeRD) |
| Eval: random baseline | `scripts/evaluate_random_baseline.py` | Sanity-check the evaluation harness against real data with random scores |
| Eval: BM25 recall | `scripts/evaluate_bm25_recall.py --language {en,da}` | Recall@K for lexical retrieval alone |
| Eval: semantic recall | `scripts/evaluate_semantic_recall.py` | Recall@K for dense retrieval alone |
| Eval: full | `scripts/evaluate_full.py --split {val,test} --language {en,da} [--gate-freshness]` | BM25 + semantic + RRF hybrid, AUC/MRR/nDCG@k, beyond-accuracy metrics (diversity, novelty, coverage), cold-start/warm and head/tail slicing, paired bootstrap significance. `--gate-freshness` restricts open-corpus retrieval to articles published by each impression's own time (EB-NeRD only, since MIND has no publish timestamps) |

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

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

109 tests, one file per module. Tests that need a real downloaded dataset
are skipped (not failed) if it isn't present, so the suite stays runnable
on a clean checkout.

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
  reranking/          reserved for Component 2
scripts/             thin CLI entry points, one per pipeline stage
tests/               one test file per module
reports/
  engineering_log.md   running log of design decisions, observations, limitations
  design_note_c1.md    final <=4 page design note (written at the end)
```

## Status

Component 1 covers both required datasets end to end: acquisition,
cleaning, temporal split with leakage checks, feature store, BM25 (English
+ Danish), semantic retrieval (English + multilingual), RRF hybrid,
beyond-accuracy metrics, cold-start/warm and head/tail slicing, paired
bootstrap significance, and freshness gating with a with-vs-without
comparison (EB-NeRD only). MIND-small and both EB-NeRD bundles (`demo` and
`small`) have full evaluation results in the engineering log. Remaining
work: the <=4 page design note and Codabench submission generation.
