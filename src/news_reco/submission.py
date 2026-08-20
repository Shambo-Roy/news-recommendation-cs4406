"""Shared helpers for generating Codabench leaderboard submission files.

Both competitions score a full ranking of each impression's candidate list
against a hidden ground truth we never see -- there's no accuracy to check
locally, just a rank order to produce and format exactly as each
competition's submission guidelines specify. This module holds the two
pieces that are genuinely dataset-agnostic (turning per-candidate scores
into a 1-indexed rank list, and writing "id [ranks]" lines to a zip
containing nothing else); the MIND and EB-NeRD generation scripts differ
only in how they read their raw, unlabeled test files, which is dataset-
specific parsing that belongs in mind_adapter.py / ebnerd_adapter.py.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path


def scores_to_ranks(candidate_ids: list[str], scores: dict[str, float]) -> list[int]:
    """Convert per-candidate scores into a 1-indexed rank list, in candidate_ids' order.

    Rank 1 = highest score (the model's top pick). A candidate missing from
    `scores` is treated as score 0.0 (matches how EmbeddingIndex.score_documents
    handles a doc_id it doesn't recognize). Ties are broken by candidate_ids'
    original order (Python's sort is stable) -- neither competition documents
    a tie-breaking rule, so this is at least deterministic and reproducible.
    """
    order = sorted(range(len(candidate_ids)), key=lambda i: -scores.get(candidate_ids[i], 0.0))
    ranks = [0] * len(candidate_ids)
    for rank, original_index in enumerate(order, start=1):
        ranks[original_index] = rank
    return ranks


def write_prediction_file(rows: Iterable[tuple[object, list[int]]], txt_path: Path) -> None:
    """Write one "impression_id [rank1,rank2,...]" line per row, matching both
    competitions' documented format exactly (space before the bracket, comma-
    separated ranks, no spaces after commas).

    newline="\\n" forces LF line endings even on Windows -- the default text
    mode here would silently write CRLF, which the competitions' Unix-based
    scoring scripts were never verified to handle correctly.
    """
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        for impression_id, ranks in rows:
            rank_str = ",".join(str(r) for r in ranks)
            f.write(f"{impression_id} [{rank_str}]\n")


def zip_prediction_file(txt_path: Path, zip_path: Path) -> None:
    """Zip txt_path alone, with no enclosing folder -- both competitions'
    submission guidelines require the zip to contain nothing but the
    prediction file itself."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=txt_path.name)
