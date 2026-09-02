"""Download and extract EB-NeRD's official Codabench test set.

This is a separate download from configs/ebnerd_{demo,small}.yaml's train/
validation bundles -- ebnerd_testset is the real held-out set the RecSys
2024 challenge scores submissions against on Codabench. It ships unlabeled
(no `clicked` column) since that's what a genuine held-out test set means,
which is why it's handled by its own standalone script rather than folded
into clean_ebnerd.py: there's no temporal split or feature store to build
here, just raw files to fetch for the submission-generation script to read
directly.

Usage:
    python scripts/download_ebnerd_testset.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

BASE_URL = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com"
FILENAME = "ebnerd_testset.zip"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ebnerd_testset"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / FILENAME
    url = f"{BASE_URL}/{FILENAME}"

    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / total:.0%}", end="", flush=True)
    print()

    print(f"Extracting into {RAW_DIR} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)

    zip_path.unlink()

    print("\nDone. Contents:")
    for path in sorted(RAW_DIR.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {path.relative_to(RAW_DIR)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
