"""One-command pipeline rebuild: download -> clean -> split -> feature store.

Chains the already-independently-runnable, already-tested per-stage scripts
(Milestones 1-4) into a single command -- this is what "one command rebuilds
everything from raw files" means in practice. Each stage remains
independently runnable and independently tested; this just saves you from
remembering the order and re-typing --config four times.

Usage:
    python scripts/build_pipeline.py --config configs/mind_small.yaml

Note: the download step requires `huggingface-cli login` to have been run
once already (see scripts/download_mind_small.py's docstring) -- this
script does not perform that authentication itself.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

PIPELINE_STAGES = [
    "download_mind_small.py",
    "clean_mind.py",
    "split_mind.py",
    "build_feature_store.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()

    for stage_script in PIPELINE_STAGES:
        print(f"\n{'=' * 70}\n>>> Running {stage_script}\n{'=' * 70}", flush=True)
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / stage_script), "--config", args.config],
            check=False,
        )
        if result.returncode != 0:
            print(f"\n{stage_script} failed (exit code {result.returncode}) -- stopping pipeline.")
            sys.exit(result.returncode)

    print(f"\n{'=' * 70}\nPipeline complete.\n{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
