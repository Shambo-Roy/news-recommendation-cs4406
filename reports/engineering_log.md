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
