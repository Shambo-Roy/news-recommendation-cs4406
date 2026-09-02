"""Download and extract MIND's official Codabench test set (MINDlarge_test).

This is a separate download from configs/mind_small.yaml's train/dev --
MINDlarge_test is the real held-out set the MIND competition scores
submissions against on Codabench, hosted on the same gated Hugging Face
repo already authenticated for MIND-small (see download_mind_small.py's
docstring; `huggingface-cli login` must already have been run). It's a
different article catalog and scale from MIND-small (the full ~161k-article
MIND, not the 50k-user small sample), and ships with unlabeled impressions
(candidate lists with no click labels), so it's handled by its own
standalone script rather than folded into the small-bundle pipeline.

Usage:
    python scripts/download_mind_large_test.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO_ID = "yjw1029/MIND"
FILENAME = "MINDlarge_test.zip"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mind_large_test"


def main() -> None:
    print(f"Downloading {FILENAME} ...")
    cached_zip_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=FILENAME)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting into {RAW_DIR} ...")
    with zipfile.ZipFile(cached_zip_path) as zf:
        zf.extractall(RAW_DIR)

    print("\nDone. Contents:")
    for path in sorted(RAW_DIR.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {path.relative_to(RAW_DIR)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
