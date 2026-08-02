"""Load and validate pipeline configuration from a YAML file.

Every module in this project reads its settings from a config file instead of
hardcoded paths, so the same code can run against a different dataset (MIND
vs. EB-NeRD) or a different machine just by pointing at a different YAML file.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

# Keys every config file must define. Listed explicitly (rather than inferred
# from the dataclass) so a missing key produces one clear error message here,
# instead of a confusing AttributeError deep inside some later module.
REQUIRED_KEYS = ("dataset_name", "raw_dir", "interim_dir", "processed_dir", "random_seed")


@dataclasses.dataclass
class PipelineConfig:
    """Typed view of a dataset's pipeline configuration.

    Path fields are absolute: load_config() resolves them relative to the
    project root, not the caller's current working directory, so a script
    behaves the same whether it's run from the repo root or from scripts/.
    """

    dataset_name: str
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    random_seed: int


def load_config(config_path: str | Path) -> PipelineConfig:
    """Read a YAML config file and return a validated PipelineConfig.

    Args:
        config_path: Path to a YAML file, e.g. "configs/mind_small.yaml".

    Returns:
        A PipelineConfig with all paths resolved to absolute paths.

    Raises:
        FileNotFoundError: if config_path does not exist.
        ValueError: if a required key is missing from the file.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    missing_keys = [key for key in REQUIRED_KEYS if key not in raw]
    if missing_keys:
        raise ValueError(f"Config file {config_path} is missing required keys: {missing_keys}")

    # A config file lives at <project_root>/configs/<name>.yaml, so the
    # project root is two levels up. Anchoring relative paths here (rather
    # than at the caller's cwd) is what makes the pipeline reproducible
    # regardless of which directory a script is invoked from.
    project_root = config_path.parent.parent

    return PipelineConfig(
        dataset_name=raw["dataset_name"],
        raw_dir=(project_root / raw["raw_dir"]).resolve(),
        interim_dir=(project_root / raw["interim_dir"]).resolve(),
        processed_dir=(project_root / raw["processed_dir"]).resolve(),
        random_seed=int(raw["random_seed"]),
    )
