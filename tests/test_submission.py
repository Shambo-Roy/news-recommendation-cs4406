"""Unit tests for Codabench submission-file helpers (Milestone 14).

Hand-computed expected values throughout -- never trusting the
implementation's own output, consistent with every other module in this
project.

Run with: pytest tests/test_submission.py -v
"""

from __future__ import annotations

import zipfile

from news_reco.submission import scores_to_ranks, write_prediction_file, zip_prediction_file


def test_scores_to_ranks_orders_highest_score_first():
    # a=0.5, b=0.9, c=0.1 -- best-to-worst is b, a, c, i.e. ranks 2,1,3
    # in candidate_ids' own order [a, b, c].
    candidate_ids = ["a", "b", "c"]
    scores = {"a": 0.5, "b": 0.9, "c": 0.1}
    assert scores_to_ranks(candidate_ids, scores) == [2, 1, 3]


def test_scores_to_ranks_matches_codabench_worked_example():
    # Directly from the EB-NeRD submission guidelines page: candidates
    # [9798759, 9798604, 9777339, 9798829] rank to [3,2,4,1], meaning
    # best-to-worst order is 9798829, 9798604, 9798759, 9777339.
    candidate_ids = ["9798759", "9798604", "9777339", "9798829"]
    scores = {"9798759": 1.0, "9798604": 2.0, "9777339": 0.0, "9798829": 3.0}
    assert scores_to_ranks(candidate_ids, scores) == [3, 2, 4, 1]


def test_scores_to_ranks_missing_candidate_treated_as_zero():
    candidate_ids = ["a", "b"]
    scores = {"a": 0.5}  # "b" absent -- should score 0.0, so a (0.5) outranks it
    assert scores_to_ranks(candidate_ids, scores) == [1, 2]


def test_scores_to_ranks_ties_broken_by_original_order():
    candidate_ids = ["a", "b", "c"]
    scores = {"a": 0.5, "b": 0.5, "c": 0.5}
    # All tied -- stable sort keeps original order: a=1, b=2, c=3.
    assert scores_to_ranks(candidate_ids, scores) == [1, 2, 3]


def test_scores_to_ranks_single_candidate_is_always_rank_one():
    assert scores_to_ranks(["only"], {"only": -5.0}) == [1]


def test_write_prediction_file_matches_exact_documented_format(tmp_path):
    rows = [(139350, [3, 2, 4, 1]), (411733, [1, 3, 2])]
    out_path = tmp_path / "predictions.txt"
    write_prediction_file(rows, out_path)

    content = out_path.read_text(encoding="utf-8")
    assert content == "139350 [3,2,4,1]\n411733 [1,3,2]\n"


def test_zip_prediction_file_contains_only_the_one_file_no_folder():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        txt_path = tmp_dir / "predictions.txt"
        with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("1 [1]\n")
        zip_path = tmp_dir / "submission.zip"

        zip_prediction_file(txt_path, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert names == ["predictions.txt"]  # no folder prefix, no other files
            assert zf.read("predictions.txt").decode("utf-8") == "1 [1]\n"
