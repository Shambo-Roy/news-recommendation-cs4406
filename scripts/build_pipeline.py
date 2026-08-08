"""One-command pipeline rebuild: download -> clean -> split -> feature store.

Chains the already-independently-runnable, already-tested per-stage scripts
into a single command -- this is what "one command rebuilds everything from
raw files" means in practice. Each stage remains independently runnable and
independently tested; this just saves you from remembering the order (and,
for EB-NeRD, the extra --bundle flag) and re-typing --config every time.

Works for both datasets. Which stage list (and, for EB-NeRD, which bundle)
to run is inferred from the config file's own `dataset_name` field rather
than a separate CLI flag -- "ebnerd_small" already says both "this is
EB-NeRD" and "the small bundle", so asking for that information twice
would just invite the two answers to disagree.

Usage:
    python scripts/build_pipeline.py --config configs/mind_small.yaml
    python scripts/build_pipeline.py --config configs/ebnerd_demo.yaml
    python scripts/build_pipeline.py --config configs/ebnerd_small.yaml

Note: MIND's download step requires `huggingface-cli login` to have been
run once already (see scripts/download_mind_small.py's docstring) -- this
script does not perform that authentication itself. EB-NeRD's download
needs no authentication.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from news_reco.utils.config import load_config

SCRIPTS_DIR = Path(__file__).resolve().parent

MIND_STAGES: list[tuple[str, list[str]]] = [
    ("download_mind_small.py", []),
    ("clean_mind.py", []),
    ("split_mind.py", []),
    ("build_feature_store.py", []),
]


def ebnerd_stages(bundle: str) -> list[tuple[str, list[str]]]:
    # split_mind.py and build_feature_store.py are dataset-agnostic (Milestone
    # 9 payoff of the common-schema design) -- only the acquisition/cleaning
    # stages need to know which EB-NeRD bundle to fetch.
    return [
        ("download_ebnerd.py", ["--bundle", bundle]),
        ("clean_ebnerd.py", ["--bundle", bundle]),
        ("split_mind.py", []),
        ("build_feature_store.py", []),
    ]


def resolve_stages(dataset_name: str) -> list[tuple[str, list[str]]]:
    if dataset_name == "mind_small":
        return MIND_STAGES
    if dataset_name.startswith("ebnerd_"):
        bundle = dataset_name.removeprefix("ebnerd_")
        if bundle not in ("demo", "small"):
            raise ValueError(
                f"Unrecognized EB-NeRD bundle '{bundle}' in dataset_name '{dataset_name}' "
                "-- expected 'ebnerd_demo' or 'ebnerd_small'."
            )
        return ebnerd_stages(bundle)
    raise ValueError(
        f"Unrecognized dataset_name '{dataset_name}' -- expected 'mind_small', "
        "'ebnerd_demo', or 'ebnerd_small'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    stages = resolve_stages(config.dataset_name)

    for stage_script, extra_args in stages:
        print(f"\n{'=' * 70}\n>>> Running {stage_script}\n{'=' * 70}", flush=True)
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / stage_script), "--config", args.config, *extra_args],
            check=False,
        )
        if result.returncode != 0:
            print(f"\n{stage_script} failed (exit code {result.returncode}) -- stopping pipeline.")
            sys.exit(result.returncode)

    print(f"\n{'=' * 70}\nPipeline complete.\n{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
