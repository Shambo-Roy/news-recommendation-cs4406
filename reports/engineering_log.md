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
