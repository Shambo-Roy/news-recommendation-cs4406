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
  UHD 770 GPU (`Win32_VideoController` / no `nvidia-smi`) — this doesn't match
  the student's stated 8GB-VRAM desktop, so GPU availability must be
  re-verified directly at Milestone 7 (semantic retrieval), not assumed now.
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

---

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
