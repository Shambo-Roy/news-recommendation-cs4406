"""Download and extract the MIND-small dataset from its Hugging Face mirror.

MIND-small is hosted on a gated Hugging Face dataset repo (yjw1029/MIND), so
downloading requires an authenticated Hugging Face account that has been
granted access. Run `huggingface-cli login` yourself first, interactively, in
your own terminal -- this script never asks for or handles your token; it
relies on huggingface_hub picking up the credential huggingface-cli cached
locally on your machine.

Usage:
    python scripts/download_mind_small.py --config configs/mind_small.yaml
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from news_reco.utils.config import load_config

# The official MIND page (msnews.github.io) points both its "Training Set"
# and "Validation Set" buttons, under the MIND-small section, at this repo.
HF_REPO_ID = "yjw1029/MIND"
HF_FILENAMES = {
    "train": "MINDsmall_train.zip",
    "dev": "MINDsmall_dev.zip",
}


def download_and_extract_split(split_name: str, filename: str, raw_dir: Path) -> None:
    """Download one MIND-small split (train or dev) and unzip it in place.

    Args:
        split_name: "train" or "dev" -- used only to name the output folder.
        filename: The zip filename on the Hugging Face repo.
        raw_dir: Directory under which raw_dir/<split_name>/ is created.
    """
    print(f"Downloading {filename} ...")
    # hf_hub_download caches the file under the user's local HF cache and
    # returns the path to it. It reuses huggingface-cli's cached login token
    # automatically -- no credential is passed or read by this script.
    cached_zip_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=filename)

    split_dir = raw_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting into {split_dir} ...")
    with zipfile.ZipFile(cached_zip_path) as zf:
        zf.extractall(split_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    config.raw_dir.mkdir(parents=True, exist_ok=True)

    for split_name, filename in HF_FILENAMES.items():
        download_and_extract_split(split_name, filename, config.raw_dir)

    print("\nDone. Contents of raw_dir:")
    for path in sorted(config.raw_dir.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {path.relative_to(config.raw_dir)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
