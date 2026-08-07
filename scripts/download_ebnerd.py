"""Download and extract an EB-NeRD bundle from its public S3 bucket.

Unlike MIND (gated Hugging Face repo, Milestone 1), EB-NeRD is hosted on a
public S3 bucket with no authentication required -- confirmed via a direct
HTTP HEAD request before ever writing this script (200 OK, no login wall;
see reports/engineering_log.md).

Usage:
    python scripts/download_ebnerd.py --config configs/ebnerd_demo.yaml --bundle demo
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests

from news_reco.utils.config import PipelineConfig, load_config

BASE_URL = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com"
BUNDLE_FILENAMES = {
    "demo": "ebnerd_demo.zip",
    "small": "ebnerd_small.zip",
}


def download_and_extract(bundle: str, raw_dir: Path) -> None:
    filename = BUNDLE_FILENAMES[bundle]
    url = f"{BASE_URL}/{filename}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / filename

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

    print(f"Extracting into {raw_dir / bundle} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir / bundle)

    zip_path.unlink()  # don't leave the archive lying around alongside the extracted files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument("--bundle", choices=["demo", "small"], default="demo")
    args = parser.parse_args()
    config: PipelineConfig = load_config(args.config)

    download_and_extract(args.bundle, config.raw_dir)

    print("\nDone. Contents:")
    for path in sorted((config.raw_dir / args.bundle).rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {path.relative_to(config.raw_dir)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
