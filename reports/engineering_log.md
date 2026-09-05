# Engineering log — Assignment 1, Component 1

Running notes kept per milestone: design decisions, observations, limitations,
and future improvements. This feeds the final ≤4-page design note; it is not
the design note itself.

---

## Milestone 0 — Environment & project scaffolding (2026-08-02)

**What was built**
- Installed Python 3.12.10 via winget (machine had no Python previously).
- Repo skeleton: `configs/`, `data/{raw,interim,processed}/`,
  `src/news_reco/{data,splitting,features,retrieval/{lexical,semantic},evaluation,reranking,utils}/`,
  `scripts/`, `tests/`, `notebooks/`, `reports/`.
- `pyproject.toml` (setuptools, src-layout) + `.venv` + `requirements.txt`
  (pandas, pyarrow, numpy, PyYAML, pytest) + editable install (`pip install -e .`).
- `src/news_reco/utils/config.py`: `load_config()` reads a YAML file, validates
  required keys, and resolves relative paths to absolute paths anchored at the
  project root (not the caller's cwd).
- `configs/mind_small.yaml`: first dataset config (paths + random_seed).
- `tests/test_config.py`: 4 tests, all passing.
- `git init` + `.gitignore` (excludes `data/{raw,interim,processed}/`, `.venv/`
  — nothing committed yet, no commit requested).

**Design decisions**
- **venv + pip over conda/Poetry** — least ceremony, matches the assignment's
  "avoid unnecessary abstractions" instruction; pinned `requirements.txt` is
  enough reproducibility at this project's scale. (Student's explicit choice.)
- **Dict-free config object**: `load_config()` returns a typed `PipelineConfig`
  dataclass rather than a raw dict, so a typo'd field name fails at load time
  with a clear message instead of a `KeyError` deep inside some later module.
- **Paths resolved relative to project root, not cwd**: a config file at
  `configs/<name>.yaml` treats its own grandparent directory as the project
  root. This is what makes `python scripts/whatever.py` behave identically
  whether launched from the repo root or from inside `scripts/`.

**Observations**
- The tool environment used to set this up only reports an integrated Intel
  UHD 770 GPU (`Win32_VideoController` / no `nvidia-smi`)
- `winget`-installed Python does not appear on `PATH` within an already-open
  shell session; each new shell invocation needs `$env:Path` refreshed from
  the Machine/User environment, or must call `.venv\Scripts\python.exe`
  directly by path.

**Limitations**
- `revisions/assignment-1-revision.md` (referenced by the course README for
  full evaluation gates and the viva/individual-effort protocol) is not
  available to the student and was not used in planning. If it surfaces later
  and contradicts anything here (e.g. specific leakage-check requirements),
  those decisions may need revisiting.
- No dataset has been downloaded yet; `data/raw/mind_small/` is an empty
  placeholder directory.

**Next**
- Milestone 1: MIND-small acquisition (student downloads manually — license
  acceptance required) + raw schema audit.

---

## Milestone 1 — MIND-small acquisition & raw schema audit (2026-08-02)

**What was built**
- Verified the real MIND-small download path by inspecting msnews.github.io's
  DOM directly: both buttons now point to a **gated Hugging Face repo**
  (`yjw1029/MIND`), not the Azure blob storage URLs from older tutorials —
  the Azure Open Datasets catalog entry for MIND has been retired (redirects
  to a generic docs page). Student created an HF account, requested access,
  and ran `huggingface-cli login` in their own terminal — token never seen
  or handled in this session.
- `scripts/download_mind_small.py`: downloads `MINDsmall_train.zip` /
  `MINDsmall_dev.zip` via `huggingface_hub.hf_hub_download` (reuses the
  cached CLI login, no explicit token handling in code) and extracts them.
- `src/news_reco/data/mind_loader.py`: `load_news()` / `load_behaviors()` —
  thin pandas readers with column names and dtypes confirmed by reading the
  raw files directly (not from memory of MIND's docs).
- `tests/test_mind_loader.py`: schema tests, skipped (not failed) if the
  raw data isn't present, so the test suite still runs green for anyone
  without the dataset downloaded.
- `scripts/audit_mind_raw.py`: prints shape, date range, and label-sanity
  checks per split.

**Design decisions**
- **Loaders leave `history`/`impressions`/entity-JSON as raw strings** rather
  than parsing them into structured columns at load time. Reason: exploding
  `impressions` into one row per (article, clicked) pair is a real modeling
  decision (Milestone 2's job — it defines the shape of the interaction
  table everything downstream depends on), not something to bury inside a
  generic "loader."
- **Tests skip rather than fail when data is absent**, so `pytest` stays a
  valid whole-repo verification command regardless of whether the (large,
  license-gated) dataset happens to be present on a given machine.

**Observations (facts, not assumptions — will drive Milestone 3)**
- MIND-small train: 156,965 impressions, 50,000 users, 51,282 articles, 17
  categories. Time range **2019-11-09 → 2019-11-14** (6 days).
- MIND-small dev: 73,152 impressions, same 50,000 users, 42,416 articles.
  Time range **2019-11-15 → 2019-11-15** (exactly 1 day) — dev is literally
  the calendar day immediately after train ends. This confirms train/dev are
  already temporally disjoint by construction; Milestone 3 will treat
  official `dev` as our held-out **test** set and carve **validation** from
  the tail of official `train`, rather than reshuffling the two pools.
- `abstract` is missing for ~5% of articles in both splits (2,666/51,282
  train, 2,021/42,416 dev) — cleaning (Milestone 2) needs an explicit
  decision on how BM25/embeddings handle a missing abstract, not a silent
  NaN-concatenation.
- ~2–3% of impressions have empty click history (`history` is NaN) — these
  are the cold-start cases the rubric's required slice (cold-start vs. warm)
  will use.
- Every sampled impression label is exactly "0" or "1" — no malformed rows
  found in the first 1,000 checked per split.

**Limitations**
- Label sanity check only samples the first 1,000 rows per split, not the
  full file (156,965 / 73,152 rows) — cheap now, full-file validation should
  happen once in Milestone 2's cleaning step, not repeated on every audit run.
- Entity embeddings (`entity_embedding.vec`) cover named entities only, not
  whole-article content — confirmed by inspection. MIND does not ship
  article-level embeddings the way EB-NeRD does; Milestone 7 (semantic
  retrieval) will need to compute our own sentence embeddings over
  title+abstract text for MIND specifically.

**Next**
- Milestone 2: cleaning & normalization into the common schema (articles
  table, interactions table), including an explicit missing-abstract policy.

---

## Milestone 2 — Cleaning & normalization into the common schema (2026-08-02)

**What was built**
- `src/news_reco/data/schema.py`: column definitions for three
  dataset-agnostic tables — `articles`, `impressions`, `interactions`. Every
  future dataset adapter (MIND now, EB-NeRD later) must produce exactly
  these columns, so everything downstream (splitting, features, retrieval,
  evaluation) is written once, not once per dataset.
- `src/news_reco/data/mind_adapter.py`: `build_articles_table()` (union +
  dedupe train/dev news.tsv by `news_id`, missing abstract → `""`, derives a
  `text` field = title+abstract) and `build_impressions_and_interactions()`
  (explodes each impression's candidate string into normalized
  `impressions` + `interactions` tables, avoiding repeating `history` on
  every candidate row).
- `tests/test_mind_adapter.py`: unit tests against small hand-crafted
  DataFrames with known-correct expected outputs (dedup, missing-abstract
  fallback, explode correctness) — independent of whether the real dataset
  is downloaded.
- `scripts/clean_mind.py`: runs the adapter against the real downloaded
  data and writes `data/interim/mind_small/{articles,impressions,interactions}.parquet`.

**Design decisions**
- **Freshness/"first seen" gating deferred to Milestone 6**, not built now.
  MIND's `news.tsv` has no publish-date column at all (only EB-NeRD ships
  real timestamps), so any freshness signal for MIND would have to be a
  proxy (e.g. earliest time an article appears in any impression) — that's
  a genuine judgment call tied to *how candidates get gated during
  retrieval*, not to *what "clean" data looks like*, so it belongs with
  candidate-generation logic (M6), not cleaning (M2). Flagging this now so
  it isn't forgotten, and because it directly touches the assignment's
  anti-gaming "no future-click leakage" requirement.
- **Missing abstract → empty string, article kept** (not dropped). ~5% of
  articles have no abstract; dropping them would shrink the retrievable
  catalog and silently hurt every method's recall ceiling over a data
  formatting quirk, not a real data quality problem.
- **`impressions` and `interactions` kept as two normalized tables**
  (joined by `impression_id`) instead of one denormalized table repeating
  `history` per candidate row — standard normalization, avoids repeating a
  dozens-of-items list across every candidate of the same impression.

**Observations**
- 65,238 unique articles after deduplication (28,460 duplicate rows removed
  across train ∪ dev — confirms the two splits' catalogs overlap heavily,
  as expected for a 6-day window).
- 230,117 impressions total (156,965 + 73,152, matches Milestone 1 exactly).
- 8,584,442 candidate rows in `interactions` (~37 candidates/impression on
  average).
- **Click rate: 4.05%** — matches published MIND baseline statistics from
  the literature, a useful external sanity check that nothing is
  systematically miscounted.
- Zero articles ended up with fully empty `text` (title alone always
  covers what abstract-fallback leaves).
- Referential integrity verified: every `interactions.article_id` exists in
  `articles.article_id` (no orphaned candidate references).

**Limitations**
- Parquet round-trips the `history` list column as a `numpy.ndarray`, not a
  Python `list`, once read back — functionally equivalent (`len()`,
  indexing, iteration all work) but anything doing an `== [...]` equality
  check against a re-loaded `history` value needs `.tolist()` first. Noted
  here so it doesn't cause a confusing test failure later.
- No freshness/publish-time signal for MIND yet (see design decision above).

**Next**
- Milestone 3: temporal train/val/test split. Per Milestone 1's finding —
  treat official MIND `dev` as **test**, carve **validation** from the tail
  of official `train` by time — plus leakage checks confirming no
  impression's candidates or history reference data from beyond its own
  cutoff.

---

## Milestone 3 — Temporal train/val/test split & leakage checks (2026-08-02)

**What was built**
- Extended `PipelineConfig`/`load_config()` with `val_start`/`test_start` —
  generic fields (every dataset needs a temporal split), not MIND-specific.
- `configs/mind_small.yaml`: `val_start = 2019-11-14 00:00:00`,
  `test_start = 2019-11-15 00:00:00` (student's choice: val = train's last
  calendar day, test = official dev).
- `src/news_reco/splitting/temporal_split.py`:
  - `assign_split()` — labels each impression train/val/test by timestamp.
  - `check_split_boundaries()` — asserts every label agrees with its
    timestamp (catches a bug in assignment/joining before it corrupts eval).
  - `check_no_future_click_leakage()` — independently verifies every
    impression's claimed `history` against the actual click log, separating
    genuinely unverifiable entries (predate our window) from real
    contradictions (claimed-prior click observed only later).
- `tests/test_temporal_split.py`: 6 tests, including two *deliberately
  broken* inputs (mislabeled split, a manufactured future-click) to prove
  the checks actually catch problems, not just pass trivially.
- `scripts/split_mind.py`: runs the full split + both checks against real
  data, writes `data/processed/mind_small/impression_splits.parquet`.

**Design decisions**
- **Val = train's last calendar day (Nov 14), not a percentage cut**
  (student's explicit choice) — mirrors the dataset's own day-boundary
  logic (dev is exactly one day), rather than an arbitrary percentage with
  no real-world referent.
- **Leakage check uses a documented tolerance (1%), not zero-tolerance
  hard failure.** Real run found a genuine, quantified signal — see below —
  that traces to a structural limitation of the dataset (a truncated
  observation window), not a pipeline bug. Blocking the entire pipeline on
  a fully-explained 0.06% rate would be zero-tolerance theater, not rigor;
  reporting the exact rate and mechanism *is* the rigor the assignment's
  anti-gaming clause asks for.
- **Tie timestamps (first_click_time == impression_time) are not
  violations.** MIND timestamps have 1-second resolution; a tie is
  consistent with the click having happened at or before that second.
  Discovered via the real-data run (see Observations), not anticipated
  up front — a good example of why "run it on real data" is part of the
  milestone, not optional.

**Observations**
- Split sizes: **train 126,695 / val 30,270 / test 73,152** impressions.
- Leakage check, initial run (`>=` comparison): 6,902 / 7,470,153 (0.092%)
  history entries flagged. Investigating *why* (rather than just fixing the
  count) found:
  - 2,738 of those were exact-second ties → not real violations → fixed by
    switching to strict `>`.
  - Of the remaining 4,164 genuine violations (0.056% of all history
    entries), **60% trace to a single article, N47020** — one popular
    article being re-clicked by many users after their history already
    referenced it. Only 154 unique articles are involved in any violation.
  - Mechanism: a user's *true* first click on an article can predate our
    6-7 day logging window (unobservable to us); if they click the same
    article again later, our "first observed click" is that later
    re-click, which can appear to postdate an earlier history reference to
    the same (unobservable, earlier) original click.
- 7,453,664 / 7,470,153 (99.8%) of history entries are simply unverifiable
  (article never observed as a click in our own logs at all) — expected
  and not a violation; MIND's history is known to extend before the
  logging window starts.

**Limitations**
- We cannot fully eliminate the 0.056% residual signal without data outside
  the dataset (true pre-window click times). It's disclosed, quantified,
  and small — appropriate to report as-is in the design note rather than
  engineer away.
- The 1% tolerance threshold is a judgment call, not derived from a formal
  procedure; chosen because the observed rate (0.056%) is ~18x below it
  and the mechanism is fully explained, not because 1% has independent
  justification.

**Next**
- Milestone 4: minimal feature store (article store + interaction log,
  id-addressable, split-aware).

---00000000000000000000000000000000000000000000000000000000000000000000000

## Milestone 4 — Minimal feature store (2026-08-02)

**What was built**
- `src/news_reco/features/store.py`: `FeatureStore` class — loads
  `articles`/`impressions`/`interactions`/`impression_splits` once, exposes
  `get_article_text()`, `get_impressions(split)`, `get_interactions(split)`,
  and computes `history_length`/`is_cold_start` per impression.
- `tests/test_feature_store.py`: 4 tests against real cleaned+split data.
- `scripts/build_feature_store.py`: materializes
  `data/processed/mind_small/impressions_features.parquet` and prints
  cold-start/warm counts per split (the rubric's required slice).

**Design decisions**
- **`is_cold_start` is defined per-impression** (history length at that
  impression's own time), not per-user from a user's full history. A
  user-level definition would use information from impressions that hadn't
  happened yet relative to an earlier one — exactly the kind of leakage
  Milestone 3 was built to prevent. Not treated as an open choice: given
  the temporal-integrity work already done, a static per-user definition
  would contradict it, not just be a stylistically different option.
- **`FeatureStore` loads everything into memory once**, rather than
  querying parquet files per call. Reasonable at MIND-small's scale
  (hundreds of MB); flagged as a real limitation below for the "breaks at
  10x" design-note question.

**A real bug found by testing, not hypothetical:** `test_split_partition_covers_every_impression_exactly_once`
failed on first run — val and test impression_ids were not disjoint.
Root cause: MIND's train and dev `behaviors.tsv` each number impressions
starting from **1 independently** (they were never meant to share an ID
space). Milestone 2's `build_impressions_and_interactions` concatenated
train+dev but kept MIND's raw `impression_id` unchanged, so 73,152 dev-origin
rows silently collided with train-origin rows sharing the same raw ID.

- **Why it stayed invisible through Milestones 2 and 3:** M2 only checked
  row *counts* (correct, since concatenation doesn't drop rows). M3's
  `assign_split` operates positionally on the DataFrame, not by ID value, so
  every row still got the correct time-based label despite the collision.
  The corruption only became externally visible in M4, because
  `FeatureStore` re-joins `impression_splits.parquet` back onto impressions
  **by `impression_id`** — with a non-unique key, that merge silently fans
  out into a many-to-many join, cross-contaminating train and dev rows.
- **Fix:** `mind_adapter.build_impressions_and_interactions` now mints a
  fresh `pd.RangeIndex`-based `impression_id` after concatenation instead of
  trusting MIND's file-local numbering.
- **Regression test added:** `test_mind_adapter.py`'s existing test was
  rewritten to deliberately reuse `impression_id=1` across both input
  frames (matching MIND's real structure) — the original version used
  non-colliding IDs (1 and 2) and could never have caught this. Lesson:
  synthetic test data has to be adversarial on the specific axis a bug can
  occur on, not just "different-looking."
- **Blast radius, quantified:** re-ran `clean_mind.py` and `split_mind.py`
  after the fix. Row counts and split sizes were unchanged (126,695 /
  30,270 / 73,152 — both time-based, unaffected by the ID bug), but the
  leakage-check numbers *did* shift slightly (4,164 → 3,964 genuine
  violations), because that check also joins interactions to impressions by
  `impression_id` — so some of Milestone 3's original click-log matches
  were themselves computed against the corrupted join. The Milestone 3 log
  entry above is now superseded by these corrected numbers.

**Observations**
- Cold-start rates by split: train 1.9% (2,466/126,695), val 2.5%
  (772/30,270), test 3.0% (2,214/73,152) — small but present in every
  split, enough volume for Milestone 8's required slice metrics.

**Limitations**
- `FeatureStore` loading everything into memory won't hold at EB-NeRD scale
  (600M+ impressions) — worth naming explicitly in the design note's "where
  this breaks at 10x" section, with a sketch of the fix (chunked/lazy
  loading, or push the joins into a real on-disk store).

**Next**
- Milestone 5: evaluation harness — explain and implement AUC, MRR,
  nDCG@5, nDCG@10, plus bootstrap CIs, verified on synthetic data with
  known-correct answers before touching real retrieval output.

---

## Milestone 5 — Evaluation harness: AUC, MRR, nDCG@k, bootstrap CIs (2026-08-02)

**What was built**
- `src/news_reco/evaluation/metrics.py`: `auc_score()` (Mann-Whitney
  rank-sum identity via `pandas.Series.rank()`, average-rank tie handling,
  no new dependency), `reciprocal_rank()`, `ndcg_at_k()`. All three return
  `None` when undefined for an impression rather than a misleading 0.
- `src/news_reco/evaluation/bootstrap.py`: `bootstrap_ci()` — vectorized
  resampling (one `(n_resamples, n)` index matrix, not a Python loop).
- `src/news_reco/evaluation/harness.py`: `evaluate_ranking()` (all 4
  metrics for one impression) + `aggregate()` (mean, 95% bootstrap CI,
  n_included/n_excluded per metric across impressions).
- `tests/test_metrics.py`, `tests/test_bootstrap.py`, `tests/test_harness.py`:
  19 tests total, all against hand-derived or independently-computed
  expected values, never against the function under test's own output.
- `scripts/evaluate_random_baseline.py`: runs the full harness against real
  val-split interactions with random scores, as an end-to-end sanity check.

**Design decisions**
- **AUC implemented via the rank-sum identity**, not a pairwise O(n²) loop
  or a new library dependency (e.g. scikit-learn). `pandas.Series.rank()`
  handles tied scores (average rank) for free, which matters here since
  tied scores will be routine once BM25/embedding scores arrive in
  Milestone 6-7, not a rare edge case.
- **nDCG's gain term is written as plain `rel`, not `2^rel - 1`.** For
  binary relevance the two are identical (`2^1-1=1`, `2^0-1=0`), so
  implementing the graded-relevance formula here would add complexity with
  zero behavioral difference. Noted explicitly rather than silently
  "simplified," since it would become a real decision the moment any future
  extension introduces graded relevance.
- **Undefined metrics return `None` and are excluded from aggregation**,
  never coerced to 0. An impression with no click has no meaningful
  "ranking quality" to score; folding it in as 0 would silently bias every
  aggregate metric downward in a way that has nothing to do with ranking
  quality.

**Verification approach**
- Metric tests use hand-computed exact fractions (AUC, MRR: 1.0, 0.0, 0.5)
  or, for nDCG (harder to trust by hand), an *independent* formula computed
  inline in the test via `math.log2` — never copied from the implementation.
  The nDCG test also asserts `ndcg@5 < ndcg@10` on a case constructed so a
  relevant item falls between ranks 5 and 10, proving the k-cutoff is
  actually exercised, not just present in the signature.
- Bootstrap tests check properties (reproducibility under a fixed seed, a
  constant array collapsing to a point interval, CI width growing with
  variance, CI bracketing the sample mean) rather than exact numbers, since
  a resampling procedure's output isn't hand-computable in the way a closed
  formula is.
- **Real-data sanity check**: ran the harness against all 30,270 val
  impressions (1,222,429 candidate rows) with uniform random scores.
  Result: **AUC = 0.4994, 95% CI [0.4959, 0.5028]** — tightly
  bracketing the theoretical 0.5 for random ranking, at real scale. This is
  independent evidence the implementation is correct beyond the small
  hand-crafted tests.

**Observations**
- Every one of MIND-small's 230,117 impressions has at least one click
  (verified directly: 0 impressions with `clicked` sum = 0) — 72% have
  exactly one click, the rest a long tail up to 10+. This explains why
  `n_excluded = 0` for every metric in the random-baseline run: nothing to
  exclude when every impression has at least one positive.
- MRR (0.2287) and nDCG@5/@10 (0.2026 / 0.2625) under random scoring are
  well above 0 as expected (there's always *some* chance of ranking the
  click near the top by luck, and the metrics reward exactly that by
  degree), giving a concrete "worse than this is broken" floor for
  Milestones 6-7's real retrieval methods to beat.

**Limitations**
- `aggregate()`'s bootstrap reuses one fixed seed across all four metrics
  in a run; fine for independent per-metric CIs (what C1 needs), but a
  paired comparison between two systems (C2's ablation requirement) will
  need the *same* resampled impression indices reused across both systems'
  metric arrays, not independently reseeded per metric -- a different
  function to write in Component 2, not a fix needed here.

**Next**
- Milestone 6: BM25 lexical retrieval — explain BM25, build an index over
  the article catalog, generate candidates, and get the first real (not
  random) numbers through this harness.

---

## Milestone 6 — BM25 lexical retrieval (2026-08-03)

**What was built**
- `src/news_reco/retrieval/lexical/bm25.py`: hand-built inverted index
  (`BM25Index`, Okapi BM25, k1=1.5/b=0.75, smoothed IDF), `tokenize()`
  (lowercase + regex + small stopword list), `build_query_from_history()`
  (concatenates history titles per the official assignment PDF's Q2.2
  spec, via an injected `get_title` callable rather than a FeatureStore
  dependency, so it's testable standalone).
- `recall_at_k()` added to `news_reco.evaluation.metrics` — deliberately
  method-agnostic (plain retrieved-ID list in, relevant-ID set in) so
  Milestone 7's embeddings reuse it unchanged.
- `FeatureStore.get_article_title()` — symmetry with `get_article_text()`.
- `tests/test_bm25.py`: 10 tests, all against hand-computed BM25 scores
  (via an independently-written formula in the test, never the module's
  own code) or explicit edge cases (empty query, zero term overlap).
- `scripts/evaluate_bm25_recall.py`: builds the index over the full
  65,238-article catalog, evaluates recall@{50,100,200} on all val
  impressions with non-empty history.

**Design decisions**
- **Hand-built inverted index, not a library** — not just a pedagogical
  preference: the assignment PDF's Q2.1 explicitly says "build an inverted
  index," which settles it per "follow the assignment when it's explicit."
- **Full history used for query construction, not capped to recent-N**
  (student's explicit choice, against my recommendation to cap at 50 to
  match MIND's own published baselines and the literal word "recently").
  Documented cost: some users have up to 558 history articles, producing
  queries with ~900+ unique terms and materially longer runtimes -- a
  real, accepted trade-off, not an oversight.
- **No time-gating of the candidate pool.** As flagged since Milestone 2:
  MIND's `news.tsv` has no publish-date field, so "only retrieve articles
  that existed by this impression's time" cannot be enforced for MIND.
  Retrieval searches the full known catalog; this is a documented
  limitation, directly relevant to the assignment's anti-gaming clause
  ("report metrics with and without features unavailable at serving
  time") -- there is no gating feature to report without, for MIND
  specifically. Revisit for EB-NeRD, which does have real timestamps.

**A real performance investigation, not just "make it faster"**
Initial run took ~1 hour for 29,498 val queries. Rather than guessing at a
fix, profiled first: `fit()` is fast (1.3s over 65K articles); the cost is
in `search()` -- mean history length is 33 articles (std 41, **max 558**),
producing queries with hundreds to ~900 unique terms, each requiring a
full postings-list scan. Sample profiling: ~67ms mean per query, scaling
with term count (900-term queries: ~150-170ms; 20-term queries: ~50-65ms).
Also discovered (empirically, not assumed) that MIND's `history` field is
a **static per-user snapshot** -- it does not grow across a user's later
impressions within our observed window (verified: three impressions from
the same user, hours apart, had byte-identical history) -- which produces
genuine query duplication (33% of val's non-cold-start impressions share
an exact history with another impression). Applied two behavior-preserving
optimizations: (1) precomputed IDF cache in `fit()` (previously recomputed
`math.log()` on every posting-list scan for the same term across every
query), (2) query-result memoization keyed by query text in the eval
script (exploits the 33% duplication), plus `itertuples()` instead of
`iterrows()`. **Verified correctness, not just speed**: re-ran after
optimizing and confirmed byte-identical recall@K results to the
unoptimized run -- these were performance fixes, not silent behavior
changes.

**Results (val split, n=29,498 non-cold-start impressions)**

| K | recall@K | 95% CI |
|---|---|---|
| 50 | 0.0119 | [0.0107, 0.0130] |
| 100 | 0.0241 | [0.0226, 0.0257] |
| 200 | 0.0393 | [0.0373, 0.0413] |

**Observations**
- Recall is low in absolute terms (~4% at K=200) but this reflects a
  genuinely hard setup, not a bug: retrieval competes across the *entire*
  65,238-article catalog spanning the whole collection window, with no
  freshness gating, purely on lexical similarity between a user's past
  reading and a candidate's title+abstract. A real production system would
  restrict candidates to currently-fresh articles first, which this
  pipeline cannot do for MIND (see design decisions above) -- this is
  likely the single largest factor depressing these numbers, and a
  concrete, measured consequence of a limitation flagged back in
  Milestone 2, not new information.
- 772 impressions excluded (empty history / cold-start, no query
  buildable) -- matches Milestone 4's cold-start count for val **exactly**,
  a useful cross-check that both pipelines agree.
- This gives Milestone 7 (semantic retrieval) a concrete number to beat,
  and Milestone 8 a starting point for the lexical-vs-semantic comparison
  the assignment's Q3.5 asks for.

**Limitations**
- Runtime remains significant for the highest-history-length users even
  after optimization; no fix was applied that would change results (per
  the student's explicit choice to keep full history), only fixes that
  preserve them exactly.
- `evaluate_bm25_recall.py` runs on **val**, not test -- consistent with
  not touching test until final evaluation (Milestone 8).

**Next**
- Milestone 7: semantic (embedding-based) retrieval -- explain embeddings
  and ANN search, compute article embeddings, build an index, retrieve
  candidates using mean-pooled history embeddings (per Q3.3), and compare
  against this milestone's BM25 recall@K numbers.

---

## Milestone 7 — Semantic (embedding-based) retrieval (2026-08-03)

**What was built**
- `src/news_reco/retrieval/semantic/embeddings.py`: `compute_embeddings()`
  (sentence-transformers wrapper, L2-normalized output; imports the
  library lazily so search-only code paths never need it installed),
  `EmbeddingIndex` (brute-force cosine similarity via one matrix multiply
  + `argpartition` top-k, fit() takes precomputed embeddings rather than
  raw text so the expensive encode step is decoupled from indexing),
  `build_query_embedding()` (mean-pools history article embeddings per
  Q3.3, `None` for empty history -- same cold-start handling shape as
  BM25's `build_query_from_history()`).
- `FeatureStore` extended with optional embedding loading (skipped
  gracefully if `article_embeddings.npz` doesn't exist -- every script
  written before this milestone keeps working unmodified) and
  `get_article_embedding()`.
- `scripts/compute_article_embeddings.py`: one-time encode of all 65,238
  articles' text.
- `scripts/evaluate_semantic_recall.py`: mirrors
  `evaluate_bm25_recall.py`'s exact protocol (same val impressions, same
  non-empty-history filter, same K values) for a fair comparison.
- `tests/test_embeddings.py`: 6 tests against hand-constructed 2D vectors
  with obviously-correct cosine similarities (parallel=1.0,
  orthogonal=0.0, opposite=-1.0), not real model output.

**Design decisions (student's explicit choices)**
- **all-mpnet-base-v2** (768-dim, ~420MB) over all-MiniLM-L6-v2 (384-dim,
  ~80MB) -- higher quality, accepted slower/larger cost, against my
  recommendation of the smaller model as a first pass.
- **Brute-force numpy over FAISS** -- at 65,238 vectors, an exact matrix
  multiply is both simpler and faster than standing up an approximate
  index; the assignment explicitly permits this at small scale.

**A real debugging investigation: torch would not import**
`sentence_transformers` (and therefore torch) crashed with a Windows "DLL
initialization routine failed" error -- but *only* when something that
imports pandas/pyarrow (e.g. `FeatureStore`) got imported into the process
first; `import torch` completely alone always worked. Investigated rather
than guessed:
1. First hypothesis (plausible-sounding, wrong): the classic
   Intel-OpenMP duplicate-runtime conflict (torch/scipy/scikit-learn each
   bundle their own copy). Applied the standard fix
   (`KMP_DUPLICATE_LIB_OK=TRUE`) -- **did not fix it**, including when set
   before numpy's own import. Left in place as a harmless no-cost safety
   net, but it wasn't the real cause.
2. Isolated systematically instead: `numpy + torch` -> fine.
   `pyarrow + torch` -> crashes. `pandas + torch` -> crashes. This
   pinpointed pyarrow's bundled native libraries as the actual conflict
   with torch's, not an OpenMP issue at all.
3. Tested whether import *order* mattered: `torch` (via
   `sentence_transformers`) imported **before** pyarrow/pandas -> works,
   every time, reversed order -> fails, every time (4/4 repeated trials).
4. **Fix applied where it actually matters**: every entry-point script
   that needs both (`compute_article_embeddings.py`) now imports
   `sentence_transformers` as its literal first import, before `numpy`,
   before `FeatureStore`, before anything else -- documented inline with
   the reasoning, since this ordering requirement isn't discoverable from
   any error message and would otherwise silently break the next script
   that happens to import things in the "natural" order.
   `evaluate_semantic_recall.py` needs no such fix -- it never imports
   torch at all, only numpy, precisely because `EmbeddingIndex.fit()` was
   designed to take precomputed embeddings rather than raw text.

**A second real bug, caught immediately by running the real pipeline**
`compute_article_embeddings.py` saved `article_ids` via
`.to_numpy()` without a dtype cast, producing a numpy `object` array
(pandas' default for string columns). `np.load()` correctly refuses to
load object arrays without `allow_pickle=True` (a deliberate safety
default -- arbitrary pickle deserialization is a real risk for untrusted
files). Fixed at the source (`.astype(str)` forces a fixed-width unicode
dtype, no pickle needed) and repaired the already-computed embeddings file
in place (`allow_pickle=True` used just once, on a file this pipeline
itself generated seconds earlier -- not an untrusted input) rather than
re-running the ~38-minute encode.

**Results (val split, n=29,498 -- identical protocol to Milestone 6)**

| K | BM25 recall@K | Semantic recall@K |
|---|---|---|
| 50 | 0.0119 [0.0107, 0.0130] | **0.0183** [0.0169, 0.0198] |
| 100 | 0.0241 [0.0226, 0.0257] | **0.0290** [0.0271, 0.0307] |
| 200 | 0.0393 [0.0373, 0.0413] | **0.0444** [0.0422, 0.0464] |

Semantic retrieval's independent 95% CIs don't overlap BM25's at any K --
suggestive that mean-pooled mpnet embeddings retrieve better candidates
than title+abstract BM25 here. **Important caveat, stated precisely
rather than overclaimed**: these are independent per-method CIs, not a
paired comparison. The two methods are evaluated on the *same* impressions,
so their errors are correlated; a proper paired bootstrap (resample
impression indices once, compute the recall *difference* per resample,
check whether that CI excludes zero) is the statistically correct way to
claim significance, not non-overlapping independent intervals. That's
exactly the paired-bootstrap machinery flagged as missing back in
Milestone 5's log -- Milestone 8 (full comparison) is where it belongs.

**Observations**
- Embedding computation: ~2 min per 1,000 articles' worth of throughput
  once the model was cached locally (first run included a ~420MB one-time
  model download that inflated an early timing estimate 6x -- worth
  timing the *encode step alone* on a cached model, not total wall time
  including download, when estimating).
- Semantic search itself: full val evaluation (29,498 queries against
  65,238 x 768-dim vectors) ran in ~2 minutes -- roughly 30x faster than
  BM25's optimized run, because a single matrix multiply per query beats
  scanning variable-length postings lists for the very long queries this
  project's history-cap choice (Milestone 6) produces.
- 772 excluded impressions again matches Milestone 4's cold-start count
  exactly -- same cross-check as Milestone 6, now confirmed for a second,
  independently-built retrieval method.

**Limitations**
- No paired significance test yet (see above) -- Milestone 8.
- English-only model choice; EB-NeRD (Danish) will need its own embedding
  model decision, not a reuse of all-mpnet-base-v2.
- The torch/pyarrow import-order requirement is undocumented upstream and
  fragile: any new script combining both dependencies must remember it.

**Next**
- Milestone 8: full offline evaluation -- BM25 vs. semantic vs. a simple
  hybrid, AUC/MRR/nDCG@5/nDCG@10 via the Milestone 5 harness (not just
  recall@K), beyond-accuracy metrics (diversity, novelty, coverage),
  cold-start/warm slice, and the paired bootstrap comparison this
  milestone's results are waiting on.

---

## Milestone 8 — Full offline evaluation: BM25 vs. semantic vs. hybrid (2026-08-03)

**What was built**
- `BM25Index.score_documents()` / `EmbeddingIndex.score_documents()`: score
  a *specific, fixed* candidate set (not open-corpus search) -- needed to
  compute AUC/MRR/nDCG over an impression's actual shown candidates, where
  every item has a known label, as opposed to open-corpus retrieval where
  most retrieved articles were never shown to the user and have no
  verifiable label at all.
- `news_reco.retrieval.hybrid`: `reciprocal_rank_fusion()` (combines
  ranked lists by position, `1/(k+rank)`, k=60 -- the standard RRF
  default) and `fuse_score_dicts()` (RRF over a fixed candidate set's local
  ranks, avoiding the need for full-catalog rank lookups).
- `news_reco.evaluation.beyond_accuracy`: `intra_list_diversity()` (mean
  pairwise embedding dissimilarity within a list), `NoveltyModel`
  (self-information from **train-only** click popularity, Laplace-smoothed),
  `CoverageTracker` (catalog coverage across many recommendation lists).
- `news_reco.evaluation.slicing.classify_head_tail()`: classifies an
  impression as head/tail by the *maximum* train-popularity among its
  clicked articles.
- `paired_bootstrap_ci()` added to `news_reco.evaluation.bootstrap` --
  resamples the *same* impression indices for both systems being compared
  (preserving pairing), giving a CI on the *difference*, not two
  independent CIs to eyeball for overlap.
- `scripts/evaluate_full.py`: orchestrates all of the above across BM25,
  semantic, and RRF hybrid, sliced by head/tail, with the cold-start slice
  reported separately (see below), plus the paired significance test.
- 28 new unit tests across `test_bm25.py`, `test_embeddings.py`,
  `test_beyond_accuracy.py`, `test_hybrid.py`, `test_slicing.py`,
  `test_bootstrap.py` -- 90 tests total, all passing.

**Design decisions**
- **AUC/MRR/nDCG computed over the impression's actual shown candidates**
  (via `score_documents()`), not over our own open-corpus top-K. This
  refines Milestone 0's original framing ("score our own retrieved top-K
  against ground truth") once Milestones 6-7's real recall@K numbers
  (~4% at K=200) made that approach impractical: most of an open-corpus
  top-K has no verifiable label (it was never shown to the user), so AUC
  computed that way would be mostly undefined or built on a fabricated
  "not clicked" assumption for articles that were never actually offered.
  Rescoring the real candidate set is well-defined for every impression
  and directly comparable to the official MIND/EB-NeRD leaderboard
  protocol. recall@K (open-corpus) and AUC/MRR/nDCG (fixed-candidate-set)
  now answer two different, complementary questions: "can this method find
  the right item in the whole catalog" vs. "can this method's score rank
  the right item within what was actually shown."
- **Hybrid via Reciprocal Rank Fusion**, not weighted score averaging
  (student's explicit choice) -- avoids needing to justify a normalization
  scheme for combining BM25's unbounded scores with cosine similarity's
  bounded range.
- **Head/tail slice added alongside cold-start/warm** (student's explicit
  choice) -- cold-start/warm was known in advance to be uninformative
  for a method-vs-method comparison (see below); head/tail was chosen
  specifically because it could show a real difference.
- **Novelty and the head/tail threshold are both computed from TRAIN
  clicks only**, never from val/test -- using the split being evaluated to
  define its own difficulty slice would be circular, the same principle
  already applied to `is_cold_start` (Milestone 4) and BM25's corpus
  statistics.

**Real, honest finding: cold-start/warm was exactly as degenerate as predicted**
0% cold-start coverage for all three methods on both val (772/30,270) and
test -- every method here requires non-empty history to build a query at
all (`build_query_from_history`/`build_query_embedding` both need at least
one history article). This was anticipated before implementing (see the
design-decision discussion before this milestone started) and is reported
as its own section rather than folded into "warm" results, since it's a
real, structural limitation worth naming plainly: **this pipeline has no
way to serve a cold-start user at all**, which a production system would
need a fallback for (e.g. popularity-based recommendations) -- explicitly
out of scope for C1, but a genuine gap worth flagging for the design note's
"where it breaks" discussion.

**Results (val split, n=29,498 evaluated, 772 cold-start excluded)**

| Method | AUC | MRR | nDCG@5 | nDCG@10 | recall@200 | diversity | novelty | coverage |
|---|---|---|---|---|---|---|---|---|
| BM25 | 0.5811 | 0.3091 | 0.2798 | 0.3383 | 0.0393 | 0.8436 | 17.68 | 0.3147 |
| Semantic | 0.6452 | 0.3469 | 0.3223 | 0.3825 | 0.0444 | 0.5399 | 17.46 | 0.3171 |
| Hybrid | 0.6343 | 0.3397 | 0.3140 | 0.3728 | **0.0491** | 0.6678 | 17.57 | **0.3452** |

recall@50/@100/@200 for BM25 and semantic here are **byte-identical** to
Milestones 6 and 7's standalone recall@K runs (0.0393 / 0.0444 at K=200) --
a strong cross-check that this larger, more complex script reproduces
prior results exactly while extending them, not a coincidence.

**Paired bootstrap: semantic vs. BM25 (AUC, all impressions, val)**
Mean difference (semantic − BM25) = **0.0641**, 95% CI **[0.0599, 0.0679]**
-- tight, entirely positive, so semantic is significantly better than
BM25, not just suggestively so via non-overlapping independent CIs
(Milestone 7's caveat). This is the rigorous version of that earlier,
hedged observation.

**Two genuinely interesting slice findings, reported with an honest
interpretation rather than a proven mechanism**
- **Hybrid has the best recall@100/@200 and the best catalog coverage of
  all three methods** (0.0491 / 0.3452 vs. semantic's 0.0444 / 0.3171) --
  RRF fusion combining two structurally different candidate pools
  surfaces more distinct relevant articles than either method alone, and
  touches more of the catalog doing it. Diversity (0.6678) also lands
  neatly between BM25's (0.8436) and semantic's (0.5399), as expected from
  averaging two very differently-behaved retrievers.
- **Tail impressions score higher on AUC/MRR/nDCG than head impressions,
  for all three methods** (e.g. semantic AUC: tail 0.6485 vs. head 0.6389;
  nDCG@10: tail 0.4263 vs. head 0.3015) -- **but recall@K shows the
  opposite pattern** (semantic recall@200: head 0.0608 vs. tail 0.0355).
  Plausible reading: popular ("head") articles are easier to *find* in the
  full catalog (recall@K) because they're well-represented and
  textually/topically typical, but once several similar popular articles
  are competing for the top rank within a small shown candidate set,
  distinguishing *the specific* correct one gets harder -- rarer articles
  are less "confusable" with their candidate-set neighbors once already
  in the running. Flagged explicitly as an interpretation, not a proven
  causal claim.
- Diversity and novelty are both fairly stable across head/tail within
  each method (e.g. BM25 diversity: head 0.8476 vs. tail 0.8414) -- the
  head/tail split mainly differentiates accuracy metrics, not
  beyond-accuracy ones, in this data.

**Limitations**
- `OPEN_CORPUS_POOL = 500` (top-N each method contributes to hybrid
  open-corpus fusion) is an approximation, not each method's full-catalog
  ranking -- chosen to keep runtime bounded; a document ranked below 500
  in both methods cannot enter the fused top-200 even if it would with a
  full ranking. Not expected to matter much in practice (RRF's own
  1/(k+rank) weighting already makes rank-500 contributions negligible),
  but not verified against the unbounded case.
- `BEYOND_ACCURACY_K = 10` for diversity/novelty/coverage is a reasonable,
  common recommendation-slate size, not a value specified by the
  assignment -- a judgment call, stated as such.
- Full-catalog time-gating remains absent for MIND (no publish-date field
  -- flagged since Milestone 2); every number in this milestone inherits
  that limitation.

**Test-split (held-out, final, one-time) results (n=70,938 evaluated, 2,214 cold-start excluded)**

| Method | AUC | MRR | nDCG@5 | nDCG@10 | recall@200 | diversity | novelty | coverage |
|---|---|---|---|---|---|---|---|---|
| BM25 | 0.5639 | 0.3048 | 0.2797 | 0.3426 | 0.0179 | 0.8428 | 17.68 | 0.4400 |
| Semantic | 0.6518 | 0.3592 | 0.3435 | 0.4028 | 0.0322 | 0.5385 | 17.47 | 0.4305 |
| Hybrid | 0.6281 | 0.3445 | 0.3242 | 0.3862 | 0.0274 | 0.6667 | 17.58 | 0.4636 |

**AUC/MRR/nDCG hold up on held-out data** -- semantic beats BM25 by
roughly the same margin as on val, and the paired bootstrap confirms it
again, tighter this time from the larger sample: mean AUC diff
(semantic − BM25) = **0.0880**, 95% CI **[0.0855, 0.0907]** (n=70,938).

**recall@K drops sharply from val to test for every method** (BM25
recall@200: 0.0393 -> 0.0179; semantic: 0.0444 -> 0.0322; hybrid:
0.0491 -> 0.0274) -- roughly halved. This was investigated, not just
reported: computed what fraction of each split's *actually-clicked*
articles were never seen (as a candidate or in anyone's history) anywhere
in train. Val: 48.5% novel. **Test: 77.4% novel.** Test day (official MIND
dev, Nov 15 -- one day further from train's Nov 9-13 window than val's
Nov 14) is dominated by articles that didn't exist yet when train was
collected. Both val and test's clicked articles *are* present in our
search index (the catalog unions train+dev news.tsv, Milestone 2), so this
isn't a coverage gap -- it's that content-based retrieval (BM25/semantic)
depends on textual/topical similarity between a user's **static** history
and the target article, and a brand-new article covering an event that
didn't exist when that history was formed has structurally less of that
signal to match against. This is exactly the "rapid news decay" effect
the assignment names as a reason temporal splitting and freshness handling
are instructive here -- and it shows up as a real, ~2x, well-explained
effect in this project's own numbers, not an abstract warning.

**Head/tail and diversity/novelty/coverage patterns replicate from val to
test**, with tighter CIs from the larger sample -- tail still outperforms
head on AUC/MRR/nDCG, head still outperforms tail on recall@K, hybrid
still wins on coverage (0.4636 vs. semantic's 0.4305 and BM25's 0.4400)
and sits between BM25 and semantic on diversity. Consistent replication on
genuinely held-out data is itself a useful piece of evidence that val's
patterns weren't artifacts of that particular split.

**Next**
- Component 1 substantially complete on MIND: pipeline, temporal split
  with leakage checks, feature store, evaluation harness, BM25, semantic
  retrieval, and full comparative evaluation are all built, tested, and
  run on real data through the held-out test split. **Correction**: the
  student flagged that the previous entry mischaracterized EB-NeRD as
  "stretch" -- the assignment PDF is explicit that both datasets are
  required (Q1: "Download raw data files for both MIND-small and EB-NeRD
  demo/small"), not just recommended sequencing. EB-NeRD was always second
  in *build order* (retire risk on the easier dataset first), never
  optional. Continuing with EB-NeRD now.

---

## Milestone 9 — EB-NeRD acquisition, adapter, and pipeline port (2026-08-04)

**What was built**
- `configs/ebnerd_demo.yaml`: same `PipelineConfig` shape as MIND's config.
- `scripts/download_ebnerd.py`: EB-NeRD is a **public S3 bucket, no auth**
  (confirmed via direct HTTP HEAD before writing any code) -- much simpler
  than MIND's gated Hugging Face flow. Downloaded `ebnerd_demo.zip`
  (20.5 MB).
- `src/news_reco/data/ebnerd_loader.py` / `ebnerd_adapter.py`: raw parquet
  loaders and the transform into the common schema (`articles`,
  `impressions`, `interactions` -- the exact same three tables MIND
  produces). 6 unit tests on hand-crafted synthetic data, including a
  regression test that deliberately reuses a raw `impression_id` across
  two "splits" (mirroring Milestone 4's MIND bug) even though a direct
  check found no actual collision in EB-NeRD's real IDs -- defensive by
  design, not by observed necessity.
- **Common schema extended with `published_time`** (nullable): MIND's
  adapter now explicitly fills `NaT` for every article (previously simply
  absent from the schema), and EB-NeRD's adapter fills real timestamps.
  Re-ran `clean_mind.py` and the full test suite to confirm this was
  purely additive -- 90/90 passing, unchanged row counts.
- `scripts/clean_ebnerd.py`, and **`scripts/split_mind.py` /
  `build_feature_store.py` reused completely unmodified** for EB-NeRD
  (just pointed `--config` at `ebnerd_demo.yaml`) -- direct payoff of the
  Milestone 0 architecture decision (dataset-agnostic core, dataset-specific
  adapter). These scripts' MIND-specific names are now a known
  inconsistency (they contain no MIND-specific logic at all); left as-is
  for now, flagged as a polish item, not worth the churn mid-build.

**Real structural differences from MIND, all verified before coding, not
assumed**
- `articles.parquet` is **one file shared across all splits** (not
  split-scoped like MIND's news.tsv) -- no union/dedup needed.
- `history` is a **separate per-user file, one per split**, and genuinely
  *updated* between splits (a shared user's train-split history predates
  and differs from their validation-split history) -- verified directly:
  same user, train history ends 2023-05-15, validation history for the
  same user ends 2023-05-20. MIND's history was static within the whole
  observed window; EB-NeRD's is current-as-of-each-split's-start.
- Candidates arrive as two parallel arrays (`article_ids_inview`,
  `article_ids_clicked`), not MIND's single `"id-label"` string.
- Articles carry a **real `published_time`**, spanning 2000-2023 (vs.
  MIND's 6-day window) -- genuine freshness gating becomes possible,
  unlike MIND (Milestone 10).
- No shipped test split (EB-NeRD's real held-out test is the Codabench
  competition's, unlabeled) -- same shape as MIND, same fix: official
  validation -> our test, tail of official train -> our val.
- **History is dramatically longer**: mean 288.6 articles (MIND: 33),
  max 1,459 (MIND: 558) -- flagged before running BM25, since query cost
  scales with history length.
- **Zero cold-start impressions** in this bundle (every user has >=5
  history items) -- a genuine cross-dataset difference from MIND's
  ~2.5-3% cold-start rate, not a bug; the cold-start slice will be
  structurally empty for EB-NeRD's demo bundle specifically.
- **Click rate is roughly double MIND's**: 8.62% vs. 4.05%.

**Split cutoffs** (verified from real data, not assumed): official train
spans 2023-05-18 07:00 -> 2023-05-25 07:00 (7 days); official validation
spans 2023-05-25 07:00 -> 2023-06-01 07:00 (the next 7 days, contiguous).
`val_start = 2023-05-24 07:00:00` (last calendar day of official train,
mirroring MIND's val = train's last day); `test_start = 2023-05-25
07:00:00` (all of official validation, used whole, like MIND's dev).

**Results**: train 21,318 / val 3,406 / test 25,356 impressions.
Leakage check: 0.0296% violation rate (even lower than MIND's 0.053%),
same 1% tolerance policy applied, same reasoning (truncated-observation-
window artifact, not a pipeline bug).

**Limitations**
- `impression_id` uniqueness was checked only for the demo bundle, not the
  full dataset -- the defensive fix (mint fresh IDs, never trust raw ones)
  makes this a non-issue either way, but the check itself doesn't
  generalize as a claim about the small/large bundles.

**Next**
- BM25 and semantic retrieval on EB-NeRD.

---

## Milestone 10 — BM25 (Danish) and semantic (multilingual) retrieval on EB-NeRD (2026-08-04)

**What was built**
- **Danish tokenization fix, caught before it caused silent damage**:
  `tokenize()`'s original character class (`[a-z0-9]+`) is ASCII-only and
  would have silently mangled Danish text -- "på" ("on"/"at", one of the
  most common Danish words) tokenizes to just "p", the "å" isn't dropped
  gracefully, it just breaks the match. Fixed by extending the pattern to
  `[a-zæøå0-9]+`. Caught while writing `DANISH_STOPWORDS`, before ever
  running it against real data -- an early draft of the stopword list used
  *romanized* approximations ("paa", "naar") that would never have matched
  anything, silently making the whole stopword list a no-op. Both are
  covered by new tests (`test_tokenize_preserves_danish_special_characters`,
  `test_tokenize_with_danish_stopwords_drops_danish_function_words`).
- `tokenize()` and `BM25Index` both take a `stopwords` parameter now
  (default: English `STOPWORDS`) rather than hardcoding one language --
  `evaluate_bm25_recall.py` gained a `--language {en,da}` flag selecting
  the right set, one script for both datasets.
- `compute_article_embeddings.py` gained a `--model` flag (was hardcoded
  to the English model) -- EB-NeRD's embeddings computed with
  `paraphrase-multilingual-mpnet-base-v2` (768-dim), the multilingual
  sibling of MIND's `all-mpnet-base-v2`, same methodology, same
  architecture, per the student's explicit choice to compute embeddings
  ourselves rather than use EB-NeRD's provided word2vec/BERT artifacts.
- `evaluate_semantic_recall.py` needed **zero changes** for EB-NeRD --
  embeddings are language-agnostic once computed, so the existing script
  just worked pointed at a different config.

**Performance, profiled before running anything long (lesson carried over
from Milestone 6)**: despite ~9x longer average queries than MIND, BM25
search is actually *faster* per query on EB-NeRD (~11ms mean vs. MIND's
~67ms) -- EB-NeRD's catalog (11,777 articles) is ~5.5x smaller than
MIND's (65,238), so postings lists are shorter even though queries touch
more terms; that dominates. Full val run (3,406 impressions) completed in
well under a minute, no background execution or caching optimizations
needed.

**Results (val split, n=3,406, 0 excluded -- no cold-start impressions in
this bundle)**

| Method | recall@50 | recall@100 | recall@200 |
|---|---|---|---|
| BM25 (Danish stopwords) | 0.0038 | 0.0103 | 0.0220 |
| Semantic (multilingual) | 0.0132 | 0.0232 | 0.0364 |

**Semantic beats BM25 at every K, replicating MIND's pattern on a
completely different dataset, language, and catalog scale** -- meaningful
cross-dataset evidence that this isn't a MIND-specific artifact of the
particular model/corpus, though a formal paired-bootstrap comparison
(Milestone 8's rigor) for EB-NeRD is still pending, alongside the AUC/MRR/
nDCG/beyond-accuracy/slice work Milestone 11 will bring over.

**Next**
- Freshness gating (real, not deferred, this time -- EB-NeRD has
  `published_time`) and the full evaluation harness (AUC/MRR/nDCG,
  beyond-accuracy, paired bootstrap) ported to EB-NeRD.

---

## Milestone 11 — Freshness gating and full evaluation on EB-NeRD (2026-08-04)

**What was built**
- `src/news_reco/retrieval/availability.py`: `AvailabilityGate` --
  precomputes a sorted (published_time, article_id) array once, then
  answers "which articles were published at or before time T" via binary
  search (`np.searchsorted`) rather than filtering the whole articles
  table per impression. Refuses to construct itself from a dataset with no
  published_time data at all (MIND) rather than silently gating nothing.
- `BM25Index.search()` and `EmbeddingIndex.search()` both gained an
  `eligible_doc_ids` parameter (default `None`, fully backward compatible
  -- every existing caller and test kept working unmodified). Ineligible
  documents are masked out *before* top-k selection (BM25: skipped during
  the postings scan; embeddings: scored `-inf` and excluded), so an
  ineligible document can never displace an eligible one, and `top_k` is
  capped to the eligible count so a small eligible pool doesn't pad the
  result with excluded entries. 8 new tests.
- `evaluate_full.py` gained `--language {en,da}` (BM25 stopwords) and
  `--gate-freshness` (restricts open-corpus retrieval to articles
  published by each impression's own time) -- one script now serves both
  the "with" and "without" runs the anti-gaming clause asks for, via a
  flag, not a separate script.

**Design decision: gating applies to open-corpus retrieval, not
candidate-set scoring.** `score_documents()` (used for AUC/MRR/nDCG over
an impression's *actual shown* candidates) is deliberately never gated --
those candidates were genuinely shown to a real user by the real EB-NeRD
production system, which by definition never shows an unpublished
article. Gating only matters for *our own* open-corpus `search()`, which
reaches into the whole 11,777-article catalog including articles that, at
a given impression's time, didn't exist yet.

**Design decision: query caching disabled under gating.** The existing
query-text cache (safe when ungated, since two impressions with identical
history get identical results) becomes unsafe once results depend on
*time* too -- the same history at two different impression times can have
different eligible sets. Rather than build a time-aware cache key,
disabled caching entirely when `--gate-freshness` is set: EB-NeRD's scale
makes every query fast enough uncached that the cache was a nice-to-have,
not a requirement (profiled: cached val run and uncached val run both
completed in well under a minute; the uncached full *test* run took over
5 minutes precisely because it's ~7.4x more impressions with no caching
at all, not because gating itself is expensive per query).

**Results (val split, n=3,406) -- the "with vs. without" comparison itself**

| Method | recall@200 (ungated) | recall@200 (gated) |
|---|---|---|
| BM25 | 0.0220 | **0.0255** |
| Semantic | 0.0364 | **0.0414** |
| Hybrid | 0.0326 | **0.0379** |

AUC/MRR/nDCG are **identical** between gated and ungated runs (by design
-- they score the real shown candidate set, already implicitly gated by
reality). Catalog coverage drops slightly under gating (BM25: 0.1114 ->
0.1096; semantic: 0.0477 -> 0.0468).

**The genuinely interesting finding: gating *improves* recall@K, it
doesn't just make retrieval "more honest."** This is not obvious --
restricting the candidate pool could easily have hurt recall by removing
options. Investigated rather than just reported: the catalog spans
2000-2023 (23 years), so at any given impression's time, a large fraction
of the "always eligible regardless of gating" articles are old, and a
meaningful number of *not-yet-published* articles can still score highly
on lexical/semantic similarity despite being impossible answers (the true
click, by construction, must already exist). Removing those impostor
candidates from competition for the top-K slots makes room for genuinely
eligible candidates -- including the true click -- to rise into it. In
other words: **the ungated numbers were quietly pessimistic**, diluted by
candidates that could never have been the right answer. This directly
answers the anti-gaming clause's "with and without" requirement with a
concrete, explained direction of effect, not just two numbers side by side.

**Paired bootstrap (semantic vs. BM25, AUC, val)**: mean diff 0.0270, 95%
CI [0.0127, 0.0407] -- excludes zero, semantic significantly better,
replicating MIND's finding on a completely different dataset and
language. (Unaffected by gating, per the design decision above -- AUC is
computed on the same fixed candidate set either way.)

**Head/tail pattern is less consistent here than on MIND**: BM25 and
hybrid both show *head* impressions scoring higher on AUC (BM25: head
0.5576 vs. tail 0.5145) -- the reverse of MIND's tail-favoring pattern --
while semantic shows next to no head/tail gap on AUC (0.5310 vs. 0.5450).
recall@K strongly favors head for all three methods, consistent with
MIND. Flagged as a genuine cross-dataset difference worth discussing in
the design note, not smoothed over to match MIND's story.

**Test-split (held-out, final) results, n=25,356**

| Method | recall@200 (ungated) | recall@200 (gated) |
|---|---|---|
| BM25 | 0.0257 | **0.0286** |
| Semantic | 0.0242 | **0.0268** |
| Hybrid | 0.0232 | **0.0262** |

The gating-improves-recall effect **replicates on genuinely held-out
data** for every method -- strong evidence this is a real, robust
phenomenon (impossible-candidate dilution in an open-corpus retrieval
setup with a decades-wide catalog), not an artifact of val's particular
impressions. AUC/MRR/nDCG are again identical gated vs. ungated, as
designed. Paired bootstrap (semantic vs. BM25, AUC, test, n=25,356): mean
diff 0.0117, 95% CI [0.0065, 0.0176] -- smaller margin than val's 0.0270
but still clearly significant.

**Cross-dataset comparison worth naming explicitly**: semantic
significantly beats BM25 on both MIND (AUC diff 0.0880, test) and EB-NeRD
(AUC diff 0.0117, test) -- same direction, very different magnitude.
Plausible reading, not yet verified further: EB-NeRD's multilingual model
and 23-year-wide catalog present a harder generalization problem than
MIND's English-only, 6-day catalog, or the smaller margin simply reflects
EB-NeRD's much smaller demo-bundle catalog compressing the gap between
methods. Worth a real investigation (like the val/test recall@K
investigation in Milestone 8) if time allows before the design note, not
asserted as settled here.

**Next**
- Component 1 now covers both required datasets end-to-end: pipeline,
  temporal split with leakage checks, feature store, evaluation harness,
  BM25 (English + Danish), semantic retrieval (English + multilingual),
  hybrid, beyond-accuracy metrics, head/tail and cold-start slicing, paired
  bootstrap significance, and -- for EB-NeRD specifically, since MIND
  structurally can't support it -- freshness gating with a real "with vs.
  without" comparison. Remaining C1 work: design note (<=4 pages),
  Codabench submission generation, EB-NeRD's ebnerd_small bundle (currently
  only the demo bundle has been run) if time allows before the deadline.

## Milestone 12: EB-NeRD small bundle -- scaling up from demo (in progress)

`ebnerd_small` is ~13x demo's article count and ~30x its interaction
volume -- close to the real Codabench submission scale, so results here
carry more weight than the demo bundle's for the design note.

- `configs/ebnerd_small.yaml` added. Verified the small bundle's raw
  `impression_time` range **before** trusting the demo config's cutoffs:
  train 2023-05-18 07:00:01 -> 2023-05-25 06:59:58, validation 2023-05-25
  07:00:02 -> 2023-06-01 06:59:59 -- byte-identical boundaries to demo (demo
  and small are both subsets of the same collection window, differing only
  in user count), so `val_start`/`test_start` carried over unchanged rather
  than re-derived blind.
- `download_ebnerd.py --bundle small` and `clean_ebnerd.py --bundle small`
  both needed **zero code changes** -- the `--bundle` parameter was
  already there from Milestone 9, another payoff of not hardcoding "demo"
  anywhere in the pipeline scripts.
- Clean output: 20,738 articles, 477,534 impressions, 5,514,689
  interactions, click rate 0.0869, 0 articles missing `published_time`.
- Split: train 200,328 / val 32,559 / test 244,647 impressions. Leakage
  check over 139.4M history entries: 0.0439% genuine future-click
  violations -- within tolerance, and notably *lower* than demo's rate,
  consistent with a larger, less sparse history population.
- Feature store: 0 cold-start impressions across all three splits (every
  user has some history) -- same shape as the demo bundle.
- Multilingual embeddings computed for all 20,738 articles
  (`paraphrase-multilingual-mpnet-base-v2`, ~7 min on CPU, batch size 64).

**Design decision: defer the full `evaluate_full.py` sweep.** Timed it
directly rather than guessing: 2,000 val impressions with
`--gate-freshness` took 82s (0.041s/impression), which extrapolates to
~22 min for full val (32,559) and ~2.5-3 hours for full test (244,647) --
*per pass*, and the with/without-gating comparison needs two passes each.
Flagged the ~5-6 hour total compute cost to the user rather than launching
it silently; decision was to hold off and run the full test-split sweep
in a later session, since this bundle is an enhancement beyond the
required demo-bundle results, not a blocker for the required deliverable.
Recall@K-only evaluation (`evaluate_bm25_recall.py`,
`evaluate_semantic_recall.py`) is comparatively cheap and remains a
candidate to run first next session, ahead of the full AUC/nDCG/bootstrap
sweep.

**Full `evaluate_full.py` sweep, run next session as planned** (val +
test, gated + ungated). Ran all four passes concurrently as separate
background processes rather than sequentially -- 20 logical cores
available, and each pass is CPU-bound but largely single-process, so
contention was minor. All four completed; test-ungated (which keeps query
caching) actually finished before test-gated despite starting at the same
time, consistent with the Milestone 11 profiling note that caching, not
gating itself, is what saves time at this scale.

**Val results (n=32,559)**

| Method | recall@200 ungated | recall@200 gated | AUC ungated |
|---|---|---|---|
| BM25 | 0.0140 | **0.0156** | 0.5225 |
| Semantic | 0.0283 | **0.0304** | 0.5400 |
| Hybrid | 0.0230 | **0.0255** | 0.5396 |

Paired bootstrap (semantic vs. BM25, AUC): mean diff 0.0175, 95% CI
[0.0129, 0.0222] -- significant.

**Test results (held-out, n=244,647)**

| Method | recall@200 ungated | recall@200 gated | AUC ungated |
|---|---|---|---|
| BM25 | 0.0183 | **0.0196** | 0.5205 |
| Semantic | 0.0146 | **0.0155** | 0.5344 |
| Hybrid | 0.0159 | **0.0169** | 0.5346 |

Paired bootstrap (semantic vs. BM25, AUC, test): mean diff 0.0139, 95% CI
[0.0121, 0.0156] -- significant.

**Both findings from the demo bundle replicate at ~13x the article count
and ~10x the impression count:**
1. Freshness gating improves recall@200 for every method, on both val and
   held-out test -- the impossible-candidate-dilution effect isn't a
   demo-bundle artifact.
2. Semantic significantly outperforms BM25 on AUC, on both val and test.
   The small-bundle test margin (0.0139) sits between the demo bundle's
   test margin (0.0117) and MIND's (0.0880) -- consistent with the
   Milestone 11 hypothesis that catalog/language difficulty, not dataset
   noise, drives the gap size, though still not confirmed by a targeted
   investigation.

One reversal worth flagging honestly, not smoothing over: on the *demo*
bundle's test split, semantic's recall@200 (0.0242) was *below* BM25's
(0.0257); on the *small* bundle's test split the same pattern holds even
more strongly (semantic 0.0146 vs. BM25 0.0183) -- semantic wins on
ranking quality (AUC) but loses to BM25 on open-corpus recall at this K,
on both bundles' test splits, even though semantic wins recall@200 on
*val* for both bundles. Plausible reading: test's candidate articles skew
toward very recent/novel content that lexical overlap with a user's
history (BM25) catches better than semantic similarity to older
history-embeddings does, especially before freshness gating removes the
even-newer distractors -- worth a real investigation before the design
note, not asserted as settled here.

**Next**: design note (<=4 pages), Codabench submission generation.

## build_pipeline.py: closing the EB-NeRD gap

`build_pipeline.py` (Milestone 4) was written for MIND only -- a hardcoded
4-stage list, and every subprocess call passed `--config` alone. It never
worked for EB-NeRD, since `download_ebnerd.py`/`clean_ebnerd.py` also need
a `--bundle {demo,small}` flag the script never supplied.

**Design decision: infer dataset family + bundle from the config's
`dataset_name`, not a new CLI flag.** `configs/ebnerd_small.yaml`'s
`dataset_name: ebnerd_small` already encodes both facts a stage-selector
needs ("this is EB-NeRD" and "the small bundle"); asking for a `--bundle`
flag on top would let the two answers disagree with no validation catching
it. `resolve_stages()` parses `dataset_name` (`mind_small` -> the MIND
stage list; `ebnerd_{demo,small}` -> the EB-NeRD stage list with `--bundle`
threaded through only to the download/clean stages; anything else ->
`ValueError` with the recognized values named) and keeps
`split_mind.py`/`build_feature_store.py` un-parameterized, since they're
already dataset-agnostic.

Verified two ways: `resolve_stages()` checked directly against all three
real configs plus two invalid `dataset_name` values (both raise with a
clear message); then a full real run of
`python scripts/build_pipeline.py --config configs/ebnerd_demo.yaml`
end-to-end (download -> clean -> split -> feature store) reproduced
**byte-identical stats** to the original Milestone 9 demo-bundle run
(11,777 articles; split 21,318/3,406/25,356; 0.0296% leakage rate) --
confirms both the new wiring is correct and the whole pipeline is
genuinely reproducible from raw download alone, not just from
already-cleaned intermediate files.

## Milestone 13: Codabench submission generation

The assignment requires registering for and submitting predictions to
both official leaderboards (MIND Competition, Codabench #13967; RecSys
2024 / EB-NeRD Challenge, Codabench #2469) -- not stretch work, a
mandatory deliverable, grading not on rank but on having submitted.
Researched both competitions' submission-guidelines pages directly (not
assumed) before writing any code.

**Format, confirmed from each competition's own page**: a zip containing
exactly one file (`prediction.txt` for MIND, `predictions.txt` for
EB-NeRD; no folders, no other files) with one line per impression:
`impression_id [rank1,rank2,...,rankN]`, ranks positional against the
candidate list in the source file's own order, 1-indexed, rank 1 = best.
Both competitions' real test sets are genuinely unlabeled (no click
ground truth locally available) -- the submission's only job is a
correctly formatted ranking, not a locally-verifiable accuracy number.

**Design decision: rank by semantic similarity, not BM25 or hybrid.**
Chose semantic because it's the method with significant, reproducible AUC
gains over BM25 in every logged comparison on both datasets (Milestones 8,
11, 12), and both competitions score AUC as a primary or the primary
metric. Considered hybrid (RRF); rejected since it never clearly beat
semantic on AUC in this project's own results, so it wasn't the option
the evidence actually supported.

**Official test sets are structurally different from train/dev/val**:
both ship the real candidate list with no click labels at all (MIND:
bare `impressions` column, no "-0"/"-1" suffix; EB-NeRD: no
`article_ids_clicked` column in the test behaviors file whatsoever).
Added `parse_unlabeled_behaviors()` to both `mind_adapter.py` and
`ebnerd_adapter.py` -- deliberately much thinner than
`build_impressions_and_interactions()` (nothing to explode into an
interactions table), and deliberately keeps the *raw* impression_id
rather than re-minting a global one, since the submission file must
reference exactly the IDs the official scorer expects -- silently
renumbering them here would corrupt every submission. New
`src/news_reco/submission.py` holds the dataset-agnostic pieces
(`scores_to_ranks`, `write_prediction_file`, `zip_prediction_file`), unit
tested against a worked example taken directly from EB-NeRD's own
submission-guidelines page.

**Real bug caught by the tests, not guessed**: `write_prediction_file`'s
first version opened the output file in plain text mode, which silently
writes CRLF line endings on Windows. A test asserting the zip's raw bytes
(not `Path.read_text()`, which normalizes newlines back to `\n` and would
have hidden this) caught `'1 [1]\r\n' != '1 [1]\n'`. Fixed with
`newline="\n"` on the file open. Worth naming: this would have shipped a
subtly malformed submission to a Unix-based scorer without ever failing a
test that used the wrong assertion method.

**Real bug reproduced, not just fixed once**: the embeddings-cache
`.npz` save hit the exact same object-array bug from Milestone 7
(`.astype(str).to_numpy()` keeps a pandas object dtype; `np.load(...,
allow_pickle=False)` then refuses it) -- reproduced here by writing the
two casts in the wrong order in a new script. Caught immediately on the
first real run (not a silent latent bug), after already having encoded
120,959 articles over 76 minutes. Repaired the cache file in place
(`allow_pickle=True` on a file this pipeline itself had just generated,
same reasoning as Milestone 7: self-generated and trusted, not an
untrusted external file) rather than re-encoding, and fixed the actual
bug in both submission scripts for the run that mattered.

**MIND submission**: MINDlarge_test (2,370,727 impressions, 120,959
articles) is on the same gated Hugging Face repo already authenticated
for MIND-small -- no new access needed. Catalog embeddings: 76 min
one-time. Ranking, once embeddings were cached: 545.8s for all 2.37M
impressions (0.23ms/impression) -- the per-impression scoring loop was
never the bottleneck. Output validated two ways: every sampled line is a
genuine 1..N permutation, and the file's raw bytes confirmed LF-only line
endings.

**EB-NeRD submission -- a real production-scale failure, not just a
slow run.** The official `ebnerd_testset` is far larger than anything
this project had handled: 13,536,710 impressions, an 807,677-row history
file, 125,541 articles. The first version of
`generate_ebnerd_submission.py` mirrored the MIND script's structure
(load everything into memory with pandas, build a Python history-ID list
per impression, accumulate every result before writing) and **crashed the
whole machine** -- not just the Python process -- twice, confirmed by the
user directly (system genuinely restarting mid-run, no other processes
competing for resources). Root-caused rather than just retried blindly:
`parse_unlabeled_behaviors()` built a *fresh* Python list of history
article-ID strings for *every one of 13.5M impressions*, even though
impressions massively share users (whose history doesn't change within
this snapshot) -- on the order of a billion small Python string objects
against a 16GB machine with under 10GB free. This is the same class of
mistake Milestone 6 (BM25 performance) had already surfaced once for a
different reason (recomputing instead of reusing what's shared across
queries), now at a scale where it took the OS down instead of just being
slow.

Fixed by redesigning around three changes, not by asking for a bigger
machine:
1. **Precompute one mean-pooled query embedding per user up front** (a
   flat `(807_677, 768)` float32 matrix, built once) instead of a raw
   history-ID list rebuilt per impression. Replaces up to ~13.5M list
   constructions with one pass over 807,677 users.
2. **Column-pruned the history read** (`user_id`, `article_id_fixed`
   only) -- history.parquet also carries `impression_time_fixed`,
   `scroll_percentage_fixed`, and `read_time_fixed` arrays this script
   never uses, and loading them was pure waste.
3. **Streamed behaviors.parquet in 200K-row batches** via
   `pyarrow.parquet.ParquetFile.iter_batches()` and wrote predictions
   incrementally (`write_prediction_file` already accepted an `Iterable`,
   so this needed no change to the shared helper) instead of
   materializing all 13.5M results before writing anything.

Verified safety before re-attempting the full run: a 3,000-impression
smoke test completed in ~100s with correct output, confirming the new
code path worked before committing machine time to it again. Full run:
97.6s to precompute all 807,677 users' query embeddings, then 333.4s
(0.02ms/impression, even faster per-impression than MIND) to stream-rank
and write all 13,536,710 impressions -- **no crash**, flat memory
throughout. Output validated the same way as MIND's: exact expected line
count, a 5,000-line random sample all verified as genuine permutations,
raw-byte LF-ending check.

**Official Codabench leaderboard results** (both submissions registered,
uploaded, and scored by the user):

| Competition | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| MIND (MINDlarge_test) | 0.6544 | 0.3212 | 0.3465 | 0.4023 |
| EB-NeRD (ebnerd_testset) | 0.5349 | 0.3413 | 0.3785 | 0.4579 |

**Compared against this project's own local validation, as a sanity
check on the whole pipeline** (not just the ranking code, but the
download -> catalog -> embed -> score -> format -> zip chain end to end):

| | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| MIND local (MIND-small test, n=70,938) | 0.6518 | 0.3592 | 0.3435 | 0.4028 |
| MIND official (MINDlarge_test) | 0.6544 | 0.3212 | 0.3465 | 0.4023 |
| EB-NeRD local (ebnerd_small test, n=244,647) | 0.5344 | 0.3402 | 0.3741 | 0.4546 |
| EB-NeRD official (ebnerd_testset) | 0.5349 | 0.3413 | 0.3785 | 0.4579 |

EB-NeRD's official score is essentially identical to the local number
across all four metrics (largest gap 0.0044, on nDCG@10) -- strong,
independent confirmation that the local evaluation harness, the
official-test adapter, and the scoring/ranking code are all correct and
consistent with each other, from a scorer this project has no control
over. MIND's AUC and nDCG both land within ~0.003 of the local number
too, but MRR is a real, honest gap (0.3212 official vs. 0.3592 local, a
drop of ~0.038) worth naming rather than smoothing over -- MRR is the
metric most sensitive to getting the single top-ranked item exactly
right, and MINDlarge_test's candidate lists come from the full ~161K
MIND catalog rather than MIND-small's ~51K, so a harder, larger, more
diverse candidate pool per impression is a plausible explanation, though
not independently verified here (would need re-running local evaluation
against MINDlarge's own train/dev, not MIND-small, to isolate catalog
scale from any other difference between the two data slices).

**Next**: the design note (<=4 pages), the last remaining Component 1
deliverable.

---

# Assignment 2, Component 2 — Learning from Click-Logs

Separate from Component 1's milestone numbering above. Covers: the
GBDT re-ranker built during the Component-1-to-Component-2 transition
(training/evaluation/ablations/Codabench submissions for both datasets),
and the subsequent formal audit against Assignment 2's actual written
spec. Milestone numbers below are this audit's own (0, 1, 2, ...), not a
continuation of Component 1's.

## GBDT re-ranker built prior to the formal audit (dates: 2026-08-29 to 2026-09-01)

Before the Assignment 2 PDF was reviewed against the implementation, a
full GBDT learning-to-rank re-ranker was built on top of Component 1,
entirely additive (no Component-1 file modified — enforced by a
`git diff --quiet HEAD` test on six protected files):

- `src/news_reco/reranking/gbdt.py`: 11 features (MIND) / 12 (EB-NeRD,
  +`article_age_hours`) across retrieval/content/behavioral/semantic-
  history/temporal families; `ExpandingTrainPopularity` (leakage-safe
  time-aware popularity for TRAIN rows) and `build_frozen_train_click_counts`
  (frozen popularity for VAL/TEST); `ArticleLookup` (dict-based, not
  `.loc`-based, article metadata cache); vectorized `batch_max_jaccard_overlap`
  / `batch_semantic_history_similarity` (needed once EB-NeRD's ~145-290
  average history length made the naive per-candidate Python loops a
  multi-hour bottleneck at MINDlarge_test/ebnerd_testset scale).
- `scripts/train_gbdt_reranker.py` / `evaluate_gbdt_reranker.py`: build
  cached feature tables, train `LGBMRanker` (LambdaRank, early-stopped on
  val nDCG), evaluate vs. the A1 semantic-retrieval score with paired
  bootstrap CI, confined-candidate diversity/novelty/coverage, head/tail
  slicing. `--ablation` flag supports `retrieval_only` /
  `retrieval_content` / `retrieval_behavioral` / `retrieval_semantic_history`
  / `full`.
- `scripts/generate_mind_submission_gbdt.py` / `generate_ebnerd_submission_gbdt.py`:
  official Codabench submissions, memory-safe streaming (lesson carried
  over from Component 1's EB-NeRD crash: never materialize a full feature
  table for millions of impressions at once).

**Results (MIND-small, held-out test, n=70,938):** GBDT beats the A1
semantic baseline on all 4 metrics, paired bootstrap CI excludes zero
throughout — AUC +0.0094 [+0.0082,+0.0107], MRR +0.0131 [+0.0116,+0.0147],
nDCG@5 +0.0122 [+0.0108,+0.0137], nDCG@10 +0.0122 [+0.0110,+0.0134].
Ablations (isolated, not cumulative, each = retrieval + exactly one other
family) showed retrieval-only and retrieval+behavioral are NOT
significant improvements (retrieval+behavioral is significantly *worse*
on AUC, -0.0025), despite `article_training_click_count` being the single
most-used feature by split count in the full model — content features
carry nearly all of the real gain, semantic-history adds a smaller but
significant increment, and the full model beats every subset on every
metric (real complementary interaction, not one dominant family).

**Results (EB-NeRD-small, held-out test, n=244,647):** much larger gain
— AUC +0.1283 [+0.1268,+0.1298], MRR +0.0819, nDCG@5 +0.1089, nDCG@10
+0.0845, all significant. `article_age_hours` (a signal MIND structurally
cannot have) is the single most-used feature in the full model.

**Two real bugs found and fixed during official-submission generation
(EB-NeRD, ebnerd_testset, 13,536,710 impressions):**
1. The official test set's own "beyond-accuracy" evaluation slice (its
   last 200,000 rows) all share the placeholder `impression_id=0`.
   Grouping by `impression_id` for batch efficiency silently merged/lost
   most of these 200,000 genuinely distinct rows, and sorting a batch
   that straddled the transition by `impression_id` value scrambled row
   order (all the `0`s sorted to the front). Fixed by grouping internally
   by raw row POSITION within each streamed batch instead of by the
   `impression_id` value, reattaching the real id only at output time.
   Confirmed against the already-submitted baseline's own
   `predictions.txt`, which independently proves the correct convention
   is one line per raw row, not one per unique id.
2. A resume mechanism based on "which impression_ids are already
   written" is unsound on this same dataset for the same reason (id=0
   can't identify a specific row). Replaced with a byte/line-count-based
   positional resume (`count_completed_rows` + `skip_rows`), which let a
   run interrupted at 98.5% (~40+ hours of compute) be recovered by
   reprocessing only the last ~216K rows instead of restarting.

Both submissions independently validated after the fact: MIND
(2,370,727 lines, exact ID-set match, valid permutations) and EB-NeRD
(13,536,710 lines, exact source-row-count match, 200,000 correctly
distinct beyond-accuracy lines, valid permutations, LF-only).

## Milestone 0 — Repository + GBDT audit against the actual Assignment 2 PDF (2026-09-01)

Read the official Assignment 2 PDF for the first time in this audit and
checked the GBDT work above against it point by point, rather than
assuming the work already done satisfies the spec. Two findings changed
the picture immediately:

1. **The "official/starter baseline" Q3.1 asks to reproduce is NRMS**
   (from the ebnerd-benchmark repo) **or "the MIND baseline"** — a neural
   model. Everything evaluated so far compares GBDT against Assignment
   1's own semantic-retrieval score, which is a different, easier
   comparison than the spec literally asks for. Not yet resolved —
   deferred to Milestone 4 (baseline audit) rather than assumed away.
2. **Candidate generation doesn't match "retrieve top-K ≈ 100–200."**
   Every script today reranks whatever candidate list the impression
   already contains (MIND ~36-40/impression, EB-NeRD ~11-12/impression)
   — never an actual open-corpus retrieval step. Neither dataset's
   natural candidate-set size is close to 100-200, which is evidence the
   assignment intends a real BM25/semantic top-K retrieval stage distinct
   from what the Codabench submission itself requires (rank exactly the
   shown candidates). Deferred to Milestone 3.

Also flagged, pending later milestones: cold-start impressions are
currently excluded entirely from GBDT train/eval (same structural
limitation as A1's own retrieval), and the user explicitly said to stop
before changing this; EB-NeRD's own ablation study (done for MIND) was
never run; diversity/novelty/coverage have no bootstrap CI (only the four
accuracy metrics do); serving/scale analysis (Q4) doesn't exist at all;
a design-note PDF and `reports/gbdt_reranker_experiment.md` are both
still unwritten.

## Milestone 1 — Feature audit (2026-09-01)

Every feature in `gbdt.py` traced against actual raw-schema availability
(not assumed from the assignment's wording) for both datasets. Full
table in the audit response; key finding:

**EB-NeRD's `history.parquet` has two arrays currently discarded by
Component 1's adapter**: `impression_time_fixed` and `read_time_fixed`
(parallel to the `article_id_fixed` array Component 1's adapter already
uses). These are exactly what recency-weighted exponential decay and a
genuine dwell-time signal need, and they're leak-safe for the same
reason `article_id_fixed` already is (they describe clicks strictly
before the current impression). Two ways to reach them: extend
Component 1's common schema (touches the adapter contract Assignment 2
says to build on top of, not rewrite), or read `history.parquet`'s raw
arrays a second time inside GBDT-only code, keyed by `user_id` (fully
additive, small duplication). Leaning toward the second, not yet
decided.

Separately, `behaviors.parquet` (the CURRENT impression) has `read_time`,
`scroll_percentage`, `session_id`, `next_read_time`, `next_scroll_percentage`.
`session_id` is genuinely serving-safe. `read_time`/`scroll_percentage`
on this table describe engagement during/after the impression being
predicted — likely leakage if used as an input feature, not yet proven
safe. `next_read_time`/`next_scroll_percentage` are unambiguously future
information and must never be used. MIND has none of this (no session
concept, no per-history-item timestamp at all) — a genuine dataset
limitation, not a code gap; the only honest move is to omit these
features for MIND, matching how `article_age_hours` is already handled.

Position bias: neither dataset has an explicit position field; only the
candidate list's own order is available, and whether that order reflects
true display position is unverified, not confirmed.

## Milestone 2 — Leakage audit (2026-09-01)

Traced every behavioral feature from raw data to model input, with
specific attention to the concern that a statistic computed "from the
train split" can still leak *within* train if an early impression uses
information from a later one.

**Expanding training popularity** (`ExpandingTrainPopularity.count_before`):
re-derived the tie-handling by hand from `np.searchsorted(times, time,
side="left")` — for a sorted ascending array, `side="left"` returns the
count of entries strictly LESS than `time`; an entry exactly equal to
`time` is not counted. This is exactly the required "equal timestamps
must not affect the current impression" behavior, confirmed against the
implementation, not assumed. Already covered by
`test_expanding_train_popularity_future_click_does_not_affect_earlier_impression`
and `test_expanding_train_popularity_same_timestamp_ties_are_excluded`.

**Call-site audit** (grepped every call site across
`train_gbdt_reranker.py`, `generate_mind_submission_gbdt.py`,
`generate_ebnerd_submission_gbdt.py`): every single popularity
computation uses `store.get_interactions("train")` — no call site
anywhere uses val/test interactions. `train_impressions` passed to
`ExpandingTrainPopularity` deliberately includes cold-start rows (a
cold-start user's click still counts toward that article's popularity;
only the *feature-row-building* population is warm-only, a separate,
non-leakage-related scope decision).

**Frozen val/test popularity**: safe because every train timestamp
precedes every val/test timestamp by the temporal split's own
construction (Component 1, already tested) — confirmed this still holds
after the popularity redesign, not re-derived from scratch.

**Article age / `AvailabilityGate`**: eligibility uses
`np.searchsorted(published_times, time, side="right")` — the OPPOSITE
tie convention from the popularity feature (`side="right"` includes an
article published at the exact same instant as eligible; the popularity
feature's `side="left"` excludes a click at the exact same instant).
This is not an inconsistency — they answer different questions
("has this article started existing yet" vs. "did this other click
already happen, strictly before this one") and the eligibility
convention is Component 1's own, already-tested, unmodified code. Age
values are never negative by construction, and a violation (published
after the impression) is NaN'd, never clipped to 0 — confirmed against
real measured violation rates (EB-NeRD-small: 0.03%/0.0006%/0.002%
across train/val/test; the official ebnerd_testset's beyond-accuracy
block shows a near-100% violation rate for a fully understood reason —
its 200,000 rows all share one frozen timestamp,
`2023-06-01 07:00:01`, one second into the test window, while its
candidate pool draws from articles published across the whole ~6-week
span).

**History timestamps**: every feature that reads `history` (title
overlap, category/subcategory match, semantic similarity) inherits
Component 1's own `check_no_future_click_leakage` guarantee
transitively — none of these features aggregate across *other*
impressions or users, so there is no additional within-train leakage
surface beyond what Component 1 already verified for the `history` field
itself.

**Systematic per-feature cross-impression sweep** (the main point of
this milestone, not just re-checking popularity): confirmed every other
feature computes from ONLY the current impression's own history +
candidate's own static attributes (title/category/embedding, treated as
time-invariant) — none of them aggregate across other impressions at
all. `article_training_click_count` is the ONLY feature with a
cross-impression aggregation pathway, and it was already the one fixed.
No new leakage issue found; no new test added (Milestone 2's instruction
was to add tests only if an issue is found).

**Session features**: not yet implemented, nothing to trace yet — when
implemented, must read only past-session context, never the current
impression's own outcome fields (see Milestone 1's `read_time`/
`scroll_percentage` finding).

## Milestone 3 — Candidate-generation audit (2026-09-01)

Traced precisely, both paths: `train_gbdt_reranker.py`'s
`candidates_by_impression = interactions.groupby("impression_id")["article_id"].apply(list).to_dict()`
and both `generate_*_submission_gbdt.py`'s
`dict(zip(parsed["impression_id"], parsed["candidates"]))` both take the
candidate list directly from the raw dataset's own logged impression —
no BM25/semantic search over the open corpus anywhere in the GBDT path.
Confirmed: current behavior is "rerank the impression's own shown
candidates" (option 2 in the assignment's own framing), not "retrieve
top-K from the catalog, then rerank" (option 1).

Measured candidate-set sizes confirm this doesn't match K≈100-200 as a
literal reading: MIND averages 36.5-40.4/impression (max 299),
MINDlarge_test 39.3 (max 300); EB-NeRD-small averages 11.1-12.0
(max 100), `ebnerd_testset` 11.7 (max 100, or exactly 250 for the
beyond-accuracy block).

Important context found while auditing: Component 1's own
`evaluate_full.py` already has real open-corpus top-K retrieval
(`OPEN_CORPUS_POOL=500`), but only for recall@K and beyond-accuracy —
never for AUC/MRR/nDCG, which are always scored against the impression's
fixed candidate set (the only place labels exist). So even Component 1
already treats "retrieval" and "labeled accuracy scoring" as separate
concerns; the GBDT has only ever used the second. Also: MIND/EB-NeRD's
own impression logs are themselves the output of an upstream (opaque)
candidate-generation process, and essentially every published result on
these benchmarks — including NRMS, the exact baseline Q3 names — evaluates
by reranking the given slate, not independent full-catalog retrieval.
Discarding the given candidates for open-corpus top-K would lose ground-
truth labels for most of that pool and would stop being the same
evaluation NRMS itself does.

Presented three options, not decided: (1) keep as-is, documented as
matching standard benchmark convention; (2) add a separate, clearly-
labeled experiment reusing Component 1's existing `search()`/RRF to
report retrieval-stage recall@K∈[100,200], without touching the core
GBDT pipeline; (3) replace the GBDT's candidate universe with open-corpus
top-K entirely, accepting a recall ceiling and breaking Codabench
submission validity. Leaning toward (2). Awaiting user decision before
Milestone 4.

**Decision (2026-09-01): option 2 approved** — add a separate, clearly-
labeled experiment reusing Component 1's existing `search()`/RRF to
report retrieval-stage recall@K∈[100,200], without touching the core
GBDT training/eval/submission pipeline. Deferred to Milestone 7
(implementation), per the audit-first process.

## Milestone 4 — Baseline audit (2026-09-01)

Checked the actual starter repo (`ebanalyse/ebnerd-benchmark`) rather
than assume its contents. Confirmed:

- **NRMS** (Wu et al. 2019) is the named baseline. `ebrec`'s own
  reproducibility script (`examples/reproducibility_scripts/ebnerd_nrms.py`)
  targets `ebnerd_small` directly — the exact bundle already downloaded
  here — with documented, turnkey hyperparameters
  (`--epochs 5 --bs_train 32 --history_size 20 --npratio 4
  --transformer_model_name FacebookAI/xlm-roberta-large ...`).
- **Not reproduced**: zero neural training code exists anywhere in this
  repository.
- **Real, concrete blockers, not hand-waved**: `ebrec`'s
  `pyproject.toml` pins `numpy>=1.24.0,<1.26.1` and
  `scikit-learn==1.4.0`, both incompatible with this project's
  `numpy==2.1.3` / `scikit-learn==1.9.0` (needed by LightGBM's sklearn
  wrapper) — requires a fully separate environment. This machine has no
  GPU (`nvidia-smi` absent, `torch.cuda.is_available()==False`), and
  `xlm-roberta-large` (~550M params) over 5 epochs is a real GPU-bound
  cost the assignment PDF itself anticipates (recommends Colab/Kaggle/
  Lightning AI). MIND has no equivalent repo named in the PDF; likely
  Microsoft's own `recommenders-team/recommenders` (credited by
  `ebrec`'s own README as the source its NRMS/LSTUR/NPA/NAML
  implementations were adapted from), not yet confirmed.

Not resolved: whether/how to actually run NRMS reproduction, given the
environment/compute mismatch with everything else in this session.
Flagged to the user rather than silently deferred into "Milestone 7:
implement fixes" without a decision on workflow (separate env, cloud
GPU, team-of-2 split).

**Resolution (2026-09-01): NRMS reproduction is NOT required.** User's
reading, accepted as authoritative: Q2's "Option A: GBDT... Option B: a
small neural ranker (e.g., NRMS-style...)" is an either/or choice of
RE-RANKER ARCHITECTURE (GBDT was chosen, fully built) — not a
requirement to also separately reproduce NRMS. Under this reading, Q3's
"reproduce the official/starter baseline, then beat it" is satisfied by
treating Assignment 1's own retrieval pipeline as that baseline (Q2
itself says to build on "Assignment 1's candidate generator"), which the
existing GBDT-vs-A1-semantic-baseline comparison, ablations, and paired
bootstrap CIs already do in full. Milestone 4 closes with no NRMS work
required — no separate environment, no cloud GPU, no MIND-side baseline
sourcing needed.

**Cross-reference check (2026-09-01), user re-quoted Q2/Q3 verbatim to
verify**: reading Q2 and Q3 together, not Q3 in isolation, strengthens
this resolution rather than just making it plausible. Q2.4 ("Report
AUC, MRR, nDCG@5, nDCG@10 before and after re-ranking") bakes a
before/after comparison into the re-ranker task itself — "before" =
retrieval-stage score, "after" = the trained re-ranker, exactly the
GBDT-vs-A1-semantic-baseline comparison already built. Q3.2's own
examples of "one principled change" mix NRMS-specific ideas ("better
user encoding, attention over history") with GBDT-feature-engineering
ideas ("category-aware features, freshness weighting") — meaning the
list is written to cover both Q2 tracks, which only makes sense if Q3's
"baseline" is read relative to whichever Q2 option a team picks: NRMS-
as-baseline for the neural track, A1's retrieval for the GBDT track.
Residual honest ambiguity: Q3.1's parenthetical literally names NRMS/
"the MIND baseline" as *the* example of "official/starter baseline," not
A1's retrieval, so a strict textual reading could still expect it. Given
the joint Q2/Q3 reading, the practical GPU/dependency blockers (this
session's own constraints, not universal), and the grading note's
explicit "never on leaderboard rank... pipeline correctness, system
design, ablation rigour, scale analysis" — treated as a well-justified,
defensible call, not a corner cut.

## Milestone 5 — Evaluation audit (2026-09-01)

Re-read `evaluate_gbdt_reranker.py` in full and verified: AUC/MRR/nDCG
reuse Component 1's `evaluate_ranking` unmodified; both systems share
one loop over one shared table (impressions/candidates/labels computed
once, not twice) so a divergent comparison is structurally impossible,
not just conventionally avoided; paired bootstrap confirmed genuinely
paired (`paired_values` appended as a tuple inside the per-impression
loop, `paired_bootstrap_ci` uses one shared resample-index array for
both arrays); confined-vs-open-corpus beyond-accuracy already clearly
documented in both the docstring and the printed runtime output;
head/tail slice population confirmed identical between systems (single
`slice_name` computed once per impression, before the per-system loop).

Two confirmed gaps: (1) cold-start impressions never reach this script
at all (filtered upstream by `warm_impressions()`), so there is
structurally no code path that could report them — not fixed here, per
the explicit instruction to stop before changing the evaluation
definition; proposed reporting count/percentage explicitly ("N
impressions, X%, structurally excluded, no ranking fabricated") as
additive, non-metric-changing reporting. (2) diversity/novelty/coverage
have no bootstrap CI at all (plain means only) — diversity/novelty are
per-impression values and can reuse Component 1's existing
`bootstrap_ci`/`paired_bootstrap_ci` mechanically; coverage is a
set-union statistic over the whole run, not a per-impression average,
and needs a new (small) bootstrap-over-impressions routine.

## Milestone 6 — Serving/scale audit (2026-09-01)

Grepped the whole codebase (`latency`, `benchmark`, `p50/p95/p99`,
`qps`, `tracemalloc`) before concluding anything — one hit, a docstring
reference to an ad-hoc timing number, not a reusable benchmark.
Confirmed: nothing exists.

Key distinction driving Milestone 7's design: every timing number
gathered so far (MIND ~5.5ms/impression, EB-NeRD-small ~11ms/impression,
`ebnerd_testset` 6.05ms clean vs. ~14-15ms actually observed) is BATCH
THROUGHPUT with one-time setup costs amortized across millions of
impressions — not the single-request p99 latency Q4.2 asks for. Not
reporting any invented p50/p95/p99 or QPS/cost numbers from data that
wasn't collected for that purpose.

Environment documented (static facts): CPU 13th Gen Intel Core i5-13500
(14 cores/20 threads), no GPU, 16,054 MB RAM. Catalog sizes: MIND-small
~65K articles, MINDlarge_test 120,959; EB-NeRD-small 20,738,
`ebnerd_testset` 125,541. History length ~30-42 (MIND) vs. ~145-292
(EB-NeRD) — already-observed root cause of EB-NeRD's much higher BM25
cost. K: neither dataset's natural candidate count (~36-40 MIND,
~11-12 EB-NeRD) matches the assignment's suggested 100-200.

10x-scaling conclusion grounded in direct evidence, not speculation:
BM25 scoring (cost scales with candidates x unique-query-terms, which
scales with history length) is what breaks first — this project already
lived through the consequence of that once (EB-NeRD's ~23-52 hour
official-submission run vs. MIND's ~3.6 hours for a comparable
impression count).

## Milestone 7 — Implementation (in progress, 2026-09-01)

Implementing the approved fixes from Milestones 3, 5, and 6. Order:
(1) cold-start slice reporting, (2) beyond-accuracy bootstrap CI,
(3) EB-NeRD ablation study (parallel to MIND's), (4) gated-vs-ungated
EB-NeRD comparison, (5) recall@K∈[100,200] retrieval experiment,
(6) serving/scale benchmark. Recency-decay/session/dwell features
(Milestone 1 finding) held pending one more explicit confirmation on
the history.parquet raw-array-read design choice, not implemented
silently.

**Cold-start slice reporting + beyond-accuracy bootstrap CI**: added to
`evaluate_gbdt_reranker.py`. Cold-start reports count/percentage
explicitly ("N impressions, X%, structurally excluded, no ranking
fabricated") without touching any existing metric. Diversity/novelty now
get both a per-system CI (`bootstrap_ci`, reused) and a paired delta CI
(`paired_bootstrap_ci`, reused) -- confirmed their per-impression lists
stay index-aligned between systems because a value is appended for both
systems under the exact same condition (shared candidate count -> shared
top-k length -> shared None-ness). Coverage needed genuinely new code
(`bootstrap_coverage_ci`, script-local): it's a set-union statistic over
the whole run, not a per-impression average, so it resamples at the
impression level (200 resamples, not the usual 1000, since this is
O(resamples x impressions x k) in pure Python -- at EB-NeRD test scale
that's still ~1-4 minutes, a one-time cost). Verified on MIND-small val:
all 156 tests still pass, output correct.

**EB-NeRD ablation study** (test split, n=244,647), run in parallel to
MIND's Milestone 7: retrieval_only (AUC -0.0014, not an improvement) and
retrieval_behavioral (AUC -0.0517, significantly worse) are NOT
improvements -- same qualitative pattern as MIND. retrieval_content
(+0.0084) and retrieval_semantic_history (+0.0086) are modest, real
improvements. Added a 5th, EB-NeRD-only ablation (`retrieval_temporal` =
retrieval + `article_age_hours`, raises KeyError on MIND by design since
that column doesn't exist there) after noticing none of the original
four could isolate the single most-used feature in the full model.
Result: retrieval_temporal alone reaches **AUC +0.1615 -- larger than
the full model's own +0.1283**, on every one of the four accuracy
metrics. This is a genuine, non-obvious, not-fully-explained finding
(possible causes: feature interference in tree splits, or the
different early-stopping iteration between the two models -- 25 vs.
42 -- not yet isolated) being reported honestly rather than smoothed
over. Also: retrieval_temporal's confined-candidate coverage is
significantly LOWER than baseline (-0.0113) despite higher diversity
and novelty -- freshness concentrates recommendations on recently-
published articles consistently enough that per-list quality improves
while the aggregate catalog footprint shrinks.

**GBDT gated-vs-ungated comparison (2026-09-02), Assignment 2 Q9
anti-gaming requirement**: new config `configs/gbdt_ebnerd_small_gated.yaml`
(gate_freshness: true, separate model_dir/feature_table_dir so it can't
share cache with the ungated run -- gating changes which candidates
survive per impression, not just their features). Gated feature build
confirmed internally consistent with Milestone 8's own logged violation
counts: dropped exactly 639 train rows and 61 test rows, matching the
ungated run's "639/61 temporal-availability violations" exactly.
Gated test results: GBDT AUC 0.6647 vs. ungated 0.6627 (semantic
baseline unchanged at 0.5344 either way) -- gating makes negligible
practical difference at this dataset's scale, because violations are
rare (0.002% of test rows). Reported as a genuine finding, not
downplayed: the anti-gaming check passing without incident is itself
useful information, not a null result to bury.

**Serving/scale benchmark + recall@K experiment, combined into one new
script** (`scripts/benchmark_retrieval_and_serving.py`) since both need
the same open-corpus BM25+semantic+RRF retrieval call -- building two
scripts that each called it separately would be exactly the duplication
the audit process said to avoid. MEASURED (this machine, this run) vs.
ESTIMATED (back-of-envelope) kept explicit throughout, never mixed.

Memory (tracemalloc, real allocated memory, not estimated) -- MIND-small:
FeatureStore 628.0 MB, BM25Index 66.8 MB, EmbeddingIndex 201.7 MB (cross-
validated: 65,238 articles x 768 dims x 4 bytes = 200.4 MB, matches
almost exactly), model 1.9 MB. EB-NeRD-small: FeatureStore 1417.9 MB,
BM25Index 20.0 MB, EmbeddingIndex 64.1 MB, model 2.1 MB.

Single-request latency (warm system, index/model load excluded, n=500
real sampled impressions each) -- MIND: retrieval p50=118ms/p99=258ms,
feature construction p50=8ms/p99=45ms, reranking p50=2ms/p99=4ms, total
p50=128ms/p99=305ms. EB-NeRD: retrieval p50=44ms/p99=88ms, feature
construction p50=35ms/p99=84ms, reranking p50=4ms/p99=4ms, total
p50=83ms/p99=172ms. One bug caught before trusting these numbers: the
benchmark initially passed availability_gate=None unconditionally,
which silently produced a feature row missing article_age_hours for
EB-NeRD and crashed at predict() -- fixed by building a real
AvailabilityGate whenever the trained model's own feature_names expects
it (try/except ValueError, same pattern as every other script, None for
MIND). Not a cosmetic bug: running with a wrong None would have measured
a smaller, unrepresentative feature set, silently skipping exactly the
feature the EB-NeRD ablation found most important.

Recall@150 (open-corpus BM25+semantic+RRF retrieval, K=150, within the
assignment's suggested 100-200 range): MIND 0.0188 (n=500, consistent
order of magnitude with Component 1's own recall@200 hybrid=0.0274),
EB-NeRD 0.0060 (n=500) -- notably lower, plausibly because EB-NeRD's
catalog is nearly double MIND's while its real candidate lists are much
more narrowly curated (~11-12 shown vs. ~36-40 for MIND), making broad
retrieval less likely to surface the specific article actually shown.
This satisfies the Milestone-3-approved option 2 (a separate, clearly-
labeled experiment demonstrating top-K retrieval) without touching the
core GBDT training/eval/submission pipeline at all.

ESTIMATED section explicitly separated and labeled: naive single-core
QPS at measured p99 (MIND ~3.3 req/s, EB-NeRD ~5.8 req/s, no batching/
parallelism modeled) -- cost/QPS and 10x-scaling discussion deferred to
the design note, since they need a stated cloud-pricing assumption this
script deliberately does not invent.

## Project structure reorganization (2026-09-02)

User-requested: `scripts/` had 22 files flat in one directory. Split
into `scripts/{data,pipeline,evaluation,submission,gbdt}/` by role
(acquisition+cleaning / orchestration+shared prep / Component 1 eval /
Component 1 submissions / Component 2). Riskier than a plain file move
in three specific ways, all checked and fixed before considering this
done:

1. Six scripts compute `PROJECT_ROOT = Path(__file__).resolve().parent.parent`
   -- assumes exactly one level of nesting below the project root. Moving
   them one level deeper required `.parent.parent.parent`. Verified by
   computing each fixed file's resolved PROJECT_ROOT programmatically and
   comparing against the real project root, not just "looks right."
2. `build_pipeline.py` dispatches other stage scripts via `subprocess`
   using a flat `SCRIPTS_DIR / stage_script` path -- broken by the move
   since stages now live in different subfolders. Fixed by renaming to
   `SCRIPTS_ROOT` (one level higher, scripts/ itself) and addressing each
   stage as `"<subfolder>/<script>.py"`. Verified by resolving every
   stage path for all three dataset configs and confirming each file
   exists on disk, not just that the script imports.
3. The git-diff-based Component-1-file-integrity test
   (`test_component1_files_are_unmodified_relative_to_git_head`) compared
   `git diff HEAD -- <path>`, which shows an uncommitted rename as a full
   delete+add (a false positive) rather than "no change." Redesigned to
   compare blob content directly via `git show HEAD:<old_path>` against
   the file's current content at its new path -- caught a real, separate
   bug in the process: this repo has `core.autocrlf=true`, so `git show`
   (raw blob, LF) and the checked-out working-tree file (CRLF) would
   have differed on EVERY file regardless of actual changes; fixed by
   normalizing CRLF->LF on both sides before comparing. Also had to
   explicitly carve out and disclose the one functionally-necessary
   exception: `generate_mind_submission.py` and
   `generate_ebnerd_submission.py`'s PROJECT_ROOT line legitimately
   differs (one more `.parent`) purely because they moved deeper, not
   because their behavior changed -- the test strips only that one line
   before comparing, and still requires 100% byte-identical content
   everywhere else.

Also updated: three other tests' skip-reason messages (stale script
paths, cosmetic), README (every script path reference, the pipeline
stages table, the project-layout tree), and two path references inside
`gbdt.py` (Component 2, mine to edit). Deliberately did NOT edit stale
path references inside `store.py`/`embeddings.py`'s error-message
strings (Component 1, protected) or historical entries earlier in this
log -- those describe what was true at the time and shouldn't be
retroactively rewritten.

Verified, not assumed: full test suite (156 passed), `--help` on one
script from each new subfolder, every `build_pipeline.py` stage path
resolved and confirmed to exist for all three dataset configs, every
fixed PROJECT_ROOT computation independently verified to resolve to the
true project root, and a real (not just import-level) end-to-end run of
`scripts/gbdt/evaluate_gbdt_reranker.py` from its new location against
real cached data.

## Recency-decay, dwell-time, and position-bias features implemented (2026-09-02)

Resolved the design decision held open since Milestone 1 (Design
Decision C): read EB-NeRD's `history.parquet` timestamp/read-time arrays
SEPARATELY in GBDT-only code (`load_ebnerd_history_detail`), not by
extending Component 1's adapter/schema. history.parquet is now read
twice across the two layers -- once by `FeatureStore` via the existing
adapter (article IDs only), once here (IDs + timestamps + read times) --
a small, deliberate duplication, not a bug.

**Correction to the Milestone 1 audit** while implementing: Q1.2's
"position bias" and "within-session click patterns" are two DIFFERENT
asks, and the audit had conflated them under one EB-NeRD-only umbrella.
Position bias -- a candidate's own rank in the impression's given
candidate list -- needs zero new data (it's already in `candidate_ids`'
own order) and is available for BOTH datasets, not EB-NeRD-only.
Implemented as `candidate_position_in_impression` (1-indexed, matching
the existing rank features' convention), added to `COMMON_FEATURE_NAMES`
and to the `retrieval_behavioral` ablation family. Not verified against
either dataset's documentation to reflect true on-screen display order
(vs. an arbitrary logging order) -- stated as a caveat, not asserted as
fact this repo doesn't have.

**New features (EB-NeRD only)**:
- `recency_weighted_semantic_similarity_to_history`
  (`RECENCY_FEATURE_NAME`): exponential-decay-weighted mean cosine
  similarity to history articles, `weight = 0.5 ** (age_hours /
  half_life_hours)`, half-life defaulted to 24h (one news cycle) --
  a plain, disclosed hyperparameter choice, not tuned.
  `batch_recency_weighted_semantic_similarity` vectorizes this the same
  way `batch_semantic_history_similarity` already does (one matmul +
  weighted sum per impression, not per candidate) -- necessary for the
  same reason: EB-NeRD's long average history.
- `mean_history_read_time_seconds` (`DWELL_FEATURE_NAME`): plain
  (not recency-weighted, deliberately, to avoid double-counting decay
  already captured by the similarity feature) mean of PAST read_time
  values. NaN read_time entries (EB-NeRD's own "not recorded"
  convention) are dropped, not averaged in as 0 -- a real zero-second
  read and a missing measurement are different facts, and treating them
  the same would understate genuine engagement.
- Both features are leak-safe for the same reason `article_id_fixed`
  already is: they describe a PAST click, not the current impression's
  outcome. `_ages_hours_strictly_before` re-derives the strictly-before
  filter defensively rather than trusting Component 1's own
  `check_no_future_click_leakage` invariant silently.
- Explicitly NOT implemented: within-session click patterns (the harder
  of Q1.2's three asks -- needs `session_id` threaded through, a
  cross-impression sequential-position stat, its own leakage design).
  Deferred, not silently dropped.

**Two real bugs caught before they could ship, both from the same root
cause** (a table-wide fixed column list vs. a per-row conditional
value) -- the same class of bug already hit once in the serving
benchmark's `AvailabilityGate` omission:
1. `build_feature_table`'s original `history_detail_by_user.get(user_id)`
   (no default) would return `None` for a user missing from the lookup,
   which would make `build_impression_rows` skip emitting
   `RECENCY_FEATURE_NAME`/`DWELL_FEATURE_NAME` for just that impression's
   rows -- while every OTHER row in the same table has those columns,
   breaking `pd.DataFrame(all_rows)[columns]` with a KeyError. Fixed by
   defaulting to `[]` (empty list, not None) when the table has opted
   into these features at all -- the empty-history-safe 0.0 defaults
   already built into the feature functions then apply correctly.
   Caught by a dedicated test
   (`test_build_feature_table_history_detail_missing_user_gets_zero_not_crash`)
   before it could surface at the next real training run, not after.
2. (Same pattern as the earlier benchmark-script fix, re-confirmed
   correctly generalized here.)

Verified on real EB-NeRD-small data before considering this done:
`load_ebnerd_history_detail` loaded 15,143 users' full history detail in
0.7s; spot-checked dtype handling (`np.datetime64`/`np.float32` from a
real parquet read, vs. `pd.Timestamp`/Python `float` in the hand-written
tests) -- comparisons and `np.mean` handle the mix transparently, no bug
found. Full test suite: 165 passed (9 new).

**Not yet done, held pending a decision**: retraining/re-evaluating any
model with these three new features actually included. That would
supersede a meaningful chunk of already-logged ablation and
gated-vs-ungated results (Milestones 7-8 and the anti-gaming
comparison), and costs real compute (a full EB-NeRD-small feature-table
rebuild was ~90 minutes the last time). Flagging to the user rather than
committing to that unilaterally.

**User approved retraining (2026-09-02).** Before launching any expensive
run, wired `history_detail_by_user` into `train_gbdt_reranker.py`
(`load_ebnerd_history_detail_by_split` -- verified against
`scripts/data/clean_ebnerd.py` directly, not assumed, that this
project's train split maps to EB-NeRD's raw "train" pair and both val
AND test map to raw "validation", since the small/demo bundles have no
separate raw "test" history file) and ran a small real-data smoke test
before committing to a full rebuild.

**Caught a real, serious bug during that smoke test, not a superficial
one**: the smoke test's first pass showed recency/dwell values as 0.0
for every single row -- structurally identical output to "no history
available," easy to wave off as plausible and move on. Investigated
instead of trusting it. Root cause: `news_reco.data.ebnerd_adapter`'s
common-schema `impressions.user_id` is a STRING (`.astype(str)`, already
established Component-1 convention), but `load_ebnerd_history_detail`'s
dict keys were the raw parquet `int64` `user_id` -- every
`history_detail_by_user.get(impression.user_id)` lookup was silently
missing, falling back to the empty-history default every time, with no
error anywhere. Confirmed directly (not inferred) by printing both
sides' actual types against real data: `'139836' <str>` vs.
`14241 <int>`. Fixed by casting to `str(user_id)` in
`load_ebnerd_history_detail`, matching the adapter's own convention
explicitly, with a comment explaining why. Also fixed the two new tests
that exercised this exact path (`test_build_feature_table_history_detail_*`)
-- both had used matching int types on BOTH sides of the lookup in their
fixtures, meaning they could never have caught this specific mismatch;
rewrote them with string user_ids to match the real contract. Re-ran the
smoke test after the fix: 97.6% of rows now get a genuinely non-zero
recency-weighted-similarity and dwell-time value (the remaining ~2.4%
are impressions whose history entries didn't survive the strictly-
before-impression-time filter, a real, expected case, not a bug).

Recorded here specifically because this is exactly the failure mode
this project's own established discipline (test against real data before
trusting a shape/crash-only check, hand-compute expected values,
investigate a suspiciously-clean result rather than accept it) exists to
catch -- and it worked, before any of the ~90-minute-scale retraining
runs below were launched on top of silently-wrong features.

**Full retraining results, both datasets (2026-09-04).** `full` ablation
retrained with the complete new feature set: `candidate_position_in_impression`
(both datasets) plus `recency_weighted_semantic_similarity_to_history` /
`mean_history_read_time_seconds` (EB-NeRD only, in addition to the
already-existing `article_age_hours`). MIND's feature-table cache was
already position-bias-free from before this session's feature work, so
its train/val/test tables were rebuilt from scratch (`--rebuild-features`);
EB-NeRD's train/val had already been rebuilt during the smoke-test/bug-fix
work above, and the run below only needed to rebuild EB-NeRD's test split
(the earlier attempt was interrupted mid-run by a session teardown --
confirmed via file timestamps that train/val were already complete and
correct before resuming with the same command minus `--rebuild-features`,
avoiding redoing ~40 minutes of good work).

MIND (mind_small, test, n=70,938; 12 features, best_iteration=40 --
unchanged in kind from before, just one more feature):

| | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| GBDT (full, retrained) | 0.6619 | 0.3716 | 0.3551 | 0.4149 |
| Semantic baseline | 0.6518 | 0.3592 | 0.3435 | 0.4028 |
| Paired bootstrap delta | +0.0100 [+0.0088,+0.0114] | +0.0124 [+0.0108,+0.0138] | +0.0116 [+0.0101,+0.0131] | +0.0121 [+0.0109,+0.0133] |

All four deltas significant (CI excludes zero). Beyond-accuracy: diversity
+0.0093 (higher, significant), novelty -0.1194 (lower, significant),
coverage +0.0015 [+0.0011,+0.0019] (higher, significant). Cold-start:
2,214/73,152 test impressions (3.03%) have empty history, structurally
excluded from both systems, reported not fabricated.

EB-NeRD (ebnerd_small, test, n=244,647; 15 features, **best_iteration=1**
-- flagged and investigated below before trusting):

| | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| GBDT (full, retrained) | 0.6693 | 0.4344 | 0.4910 | 0.5455 |
| Semantic baseline | 0.5344 | 0.3397 | 0.3739 | 0.4545 |
| Paired bootstrap delta | +0.1350 [+0.1333,+0.1366] | +0.0947 [+0.0930,+0.0963] | +0.1171 [+0.1153,+0.1188] | +0.0910 [+0.0896,+0.0924] |

All four deltas significant. Beyond-accuracy: diversity +0.0288 (higher,
significant), novelty +0.1622 (higher, significant -- notably the
*opposite* sign from MIND, and the opposite sign from the earlier
`retrieval_temporal` ablation's novelty result), coverage -0.0086
[-0.0098,-0.0077] (lower, significant). Cold-start: 0/244,647 (EB-NeRD-small's
test split has no empty-history impressions, consistent with earlier logging).

**EB-NeRD `best_iteration=1` investigated, not taken on faith.** The
previous full model (12 features, no `article_age_hours`/recency/dwell)
stopped at iteration 42; this one stopped at iteration 1 and never
improved across the full 20-round patience window. That is exactly the
kind of suspiciously-clean-looking result this project's discipline says
to investigate rather than report. Three checks, in order:
1. **Leakage check on the two new/newly-added structural features.**
   `candidate_position_in_impression` vs. `clicked`: corr -0.103, mean
   position 6.07 when clicked vs. 8.99 when not -- a real, modest
   position-bias effect (expected, already discussed in Milestone 1), not
   a near-deterministic leak. `article_age_hours` vs. `clicked`: corr
   -0.065, similarly modest. Neither is remotely close to the kind of
   correlation (near +-1) that would indicate the feature secretly encodes
   the label.
2. **New-feature sanity check.** `recency_weighted_semantic_similarity_to_history`
   has real spread (std 0.057, 2.14M unique values across 2.23M training
   rows) and a small positive correlation with clicks (+0.031);
   `mean_history_read_time_seconds` likewise has real spread (std 37.2s,
   13,745 unique values) and a small negative correlation (-0.011).
   Neither is degenerate/constant, and neither shows up as a leak.
3. **Consistency with an already-logged finding.** Milestone 7's EB-NeRD
   ablation study already found that `retrieval_temporal` (retrieval +
   `article_age_hours` alone, no other behavioral features) reaches AUC
   +0.1615 over baseline -- *larger* than the old 12-feature full model's
   own +0.1283 -- and flagged this as unexplained feature interference.
   This retrain's result is consistent with, not contradictory to, that
   finding: a full 15-feature model containing `article_age_hours` reaches
   +0.1350, still below the article-age-alone ablation's +0.1615 but
   clearly benefiting from it, and apparently finds most of that benefit
   in a single powerful first tree, after which LambdaRank's pairwise
   objective doesn't find further gains within 20 rounds on this
   validation set. This reads as the same feature-dominance phenomenon
   already on record, now more extreme with `candidate_position_in_impression`
   added alongside it (both are strong, monotonic-ish structural
   signals a single shallow tree can exploit early) -- not a new bug.

Conclusion: the retrained EB-NeRD full model is a genuine, significant
improvement over both the semantic baseline (+0.1350 AUC) and the
previous 12-feature full GBDT model (0.6693 vs. 0.6627 AUC, previously
logged), despite the unusual training dynamics. The `best_iteration=1`
fact is reported here rather than hidden because it is a real, if not
fully mechanistically explained, property of this run -- consistent with,
not contradicted by, every check performed.

**Not yet done**: EB-NeRD gated retrain (`configs/gbdt_ebnerd_small_gated.yaml`)
with the new features, and a decision on whether to refresh the 5+5
MIND/EB-NeRD ablation studies (Milestone 7) against the new feature set
or leave them as pre-position-bias/pre-recency-dwell numbers with an
explicit note. Both flagged to the user rather than assumed.

**User approved both (2026-09-04): proceed with EB-NeRD gated retrain and
ablation refresh.** Launched the EB-NeRD gated retrain (rebuild-features,
since the gated feature-table cache predates this session's feature work
and lives in a separate `feature_table_dir`) and the MIND 4-ablation
refresh in parallel (separate caches, no conflict); EB-NeRD's own 6-ablation
refresh deferred until the gated job frees up the machine, to avoid two
concurrent EB-NeRD-scale BM25/embedding-index builds contending for the
same cores. Confirmed via re-reading `train_gbdt_reranker.py` that ablation
training reuses the already-cached full-feature-set parquet and just
selects a column subset (`model_feature_names = ABLATIONS[name]`) --
ablation retraining itself is cheap; the cost is almost entirely each
ablation's own `evaluate_gbdt_reranker.py` bootstrap pass.

**MIND ablation refresh (test, n=70,938), all 4 non-full ablations now
include `candidate_position_in_impression` where structurally relevant
(added to `retrieval_behavioral`'s feature list):**

| Ablation | AUC | delta vs. baseline (95% CI) |
|---|---|---|
| retrieval_only | 0.6502 | -0.0017 [-0.0026,-0.0008] -- significantly worse |
| retrieval_content | 0.6602 | +0.0083 [+0.0072,+0.0095] -- significantly better |
| retrieval_behavioral | 0.6498 | -0.0021 [-0.0030,-0.0012] -- significantly worse |
| retrieval_semantic_history | 0.6539 | +0.0020 [+0.0009,+0.0033] -- significantly better |
| full (already logged above) | 0.6619 | +0.0100 [+0.0088,+0.0114] -- significantly better |

Same qualitative pattern as the pre-retrain MIND ablation study:
retrieval_only and retrieval_behavioral alone are not improvements over
the semantic baseline; retrieval_content and retrieval_semantic_history
are modest, real gains; full remains the best model. Adding
`candidate_position_in_impression` did not change this story -- it
shows up in `retrieval_behavioral`'s feature importances (47 splits, the
smallest of that ablation's 7 features) without flipping any ablation's
significance direction.

**EB-NeRD gated retrain (`configs/gbdt_ebnerd_small_gated.yaml`,
rebuild-features), 2026-09-04.** Gated feature-table cache
(`gbdt_features_gated/`) predates all of this session's feature work, so
a full rebuild was required (~2,166s/2,164s/2,929.9s = ~36/5/36 min for
train/val/test respectively). Anti-gaming consistency check: gated train
has exactly 639 fewer rows than the ungated retrain's train
(2,225,655 vs. 2,226,294) and gated test has exactly 61 fewer rows than
ungated test (2,928,881 vs. 2,928,942) -- matching the ungated run's own
logged "639/61 temporal-availability violations" exactly, on a
completely independent rebuild. Gating drops violating rows outright
rather than NaN-ing `article_age_hours`, so 0/0/0 violations are reported
for the gated tables themselves (expected -- there is nothing left to
violate after the drop).

Same `best_iteration=1` pattern reproduced, with a nearly identical
feature-importance collapse (`article_age_hours` 10, `candidate_position_
in_impression` 8, `bm25_rank_within_impression` 8, `semantic_rank_within_
impression` 2, `title_token_overlap_with_history` 1, `subcategory_match_
with_history` 1, every other feature 0 -- differing from the ungated
run's importances by at most 1-2 splits per feature). This is a useful,
unplanned confirmation of the earlier diagnosis: two feature tables built
independently (differing by <0.03% of rows) produced the same extreme
early-stopping behavior, which is much stronger evidence that this is a
genuine, reproducible property of the feature set + LambdaRank +
these hyperparameters than a single run could have shown -- not a
one-off artifact of a corrupted rebuild or a bad random seed.

Gated test results: GBDT AUC 0.6650 vs. semantic baseline 0.5344 (delta
+0.1307 [+0.1290,+0.1324], significant). Compared to the ungated
retrain's AUC 0.6693: gated is now 0.0043 *lower* than ungated -- the
opposite sign from the pre-retrain gated-vs-ungated comparison (which had
gated 0.0020 *higher*, AUC 0.6647 vs. 0.6627). Reported honestly rather
than reconciled into a false narrative: both differences are small
(0.002-0.004 AUC) relative to the >0.13 gap either system has over the
semantic baseline, and both are consistent with "gating makes a
negligible practical difference at this dataset's violation rate"
(0.002-0.03% of rows) -- a sign flip at this magnitude, on a model
dominated by a single feature (`article_age_hours`) that stops training
after one boosting round, is plausible small-sample/optimization noise,
not a systematic effect newly introduced by gating. All other metrics
track the same pattern as the ungated retrain (diversity +0.0290,
novelty +0.1631, coverage -0.0087, all significant; 0/244,647 cold-start
impressions, same as ungated).

**EB-NeRD ablation refresh (test, n=244,647), 6 ablations including the
new `retrieval_recency_dwell` (first time this ablation has ever been
trained/evaluated):**

| Ablation | AUC | delta vs. baseline (95% CI) | best_iter |
|---|---|---|---|
| retrieval_only | 0.5330 | -0.0014 [-0.0024,-0.0002] -- significantly worse | 1 |
| retrieval_content | 0.5428 | +0.0084 [+0.0071,+0.0097] -- significantly better | 2 |
| retrieval_behavioral | 0.4805 | -0.0539 [-0.0550,-0.0527] -- significantly worse | 2 |
| retrieval_semantic_history | 0.5392 | +0.0048 [+0.0033,+0.0064] -- significantly better | 33 |
| retrieval_temporal | 0.6959 | +0.1615 [+0.1600,+0.1631] -- significantly better | 25 |
| retrieval_recency_dwell (new) | 0.5325 | -0.0019 [-0.0031,-0.0007] -- significantly worse (AUC only) | 1 |
| full (15 features, already logged) | 0.6693 | +0.1350 [+0.1333,+0.1366] -- significantly better | 1 |

`retrieval_only`, `retrieval_content`, `retrieval_semantic_history`,
`retrieval_temporal` numbers are essentially unchanged from the
pre-retrain study (none of their feature lists gained a new feature this
session) -- `retrieval_temporal`'s +0.1615 matches the previously-logged
value exactly, a useful reproducibility check on the whole pipeline.
`retrieval_behavioral` (which gained `candidate_position_in_impression`)
is slightly *more* negative than before (-0.0539 vs. the earlier
-0.0517) -- same qualitative finding (extreme head/tail divergence: head
AUC 0.9138, tail AUC 0.4743, i.e. this ablation overfits hard to
`article_training_click_count`-style popularity signal that works very
differently across the two slices), not a new problem introduced by the
position feature.

**New, honest finding: `retrieval_recency_dwell` in isolation shows no
measurable ranking improvement.** AUC is significantly *worse* than the
semantic baseline (-0.0019), and MRR/nDCG@5/nDCG@10 show no significant
difference at all (CIs include zero). This does not contradict the
per-row correlation checks done during Milestone 7/implementation
(`recency_weighted_semantic_similarity_to_history` +0.031 corr with
clicks, `mean_history_read_time_seconds` -0.011) -- both are real but
very weak marginal signals, and evidently too weak to move a LambdaRank
model's ranking quality on their own when the model has nothing else
(no popularity, no freshness, no content overlap) to combine them with.
Their contribution inside the `full` model is entangled with the
dominant `article_age_hours` signal and cannot be cleanly isolated by
this ablation design alone. This is reported as a genuine limitation
of these two features on this dataset, not smoothed over: implementing
Q1.2's recency-decay/dwell-time asks (as the assignment requires) does
not, by itself, demonstrate a measurable accuracy improvement in
isolation -- their value (if any) is bundled into the full model's
already-reported significant gain over baseline.

Also worth naming again, now doubly confirmed (identical to the
pre-retrain finding): the full 15-feature model (AUC 0.6693) still
underperforms `retrieval_temporal` alone (AUC 0.6959) on this dataset --
i.e. feeding the model MORE information (adding 10 more features on top
of retrieval + article_age_hours) makes it slightly worse, not better.
Confirmed reproducible across two independent full-feature-table builds
(this session's retrain and the earlier pre-retrain run), so this is not
a fluke of one run -- genuine feature interference in how LambdaRank's
tree-building process handles `article_age_hours` when many other,
much-weaker features are also available to split on. Root cause not
further isolated (would need per-tree diagnostics or systematically
removing subsets of the "loser" features from `full` to find which
ones interfere) -- flagged as a real, unresolved, non-obvious limitation
of the current GBDT re-ranker on EB-NeRD, worth a sentence in the design
note rather than hidden.

**Status: both user-approved items (EB-NeRD gated retrain, MIND+EB-NeRD
ablation refresh) complete.** All models retrained/evaluated on the
full new feature set (position bias both datasets; +freshness, recency-
decay, dwell-time on EB-NeRD); every reported number above is from this
session's own real runs against real data, no numbers carried over
unchanged except where explicitly noted as an exact-match reproducibility
check (`retrieval_temporal`).
