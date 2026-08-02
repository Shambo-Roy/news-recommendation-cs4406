"""Verification tests for the config loader (Milestone 0).

Run with: pytest tests/test_config.py -v
"""

from pathlib import Path

import pytest

from news_reco.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIND_CONFIG = PROJECT_ROOT / "configs" / "mind_small.yaml"


def test_load_config_reads_all_fields():
    config = load_config(MIND_CONFIG)
    assert config.dataset_name == "mind_small"
    assert config.random_seed == 42
    assert config.val_start < config.test_start


def test_load_config_resolves_paths_to_absolute():
    config = load_config(MIND_CONFIG)
    assert config.raw_dir.is_absolute()
    assert config.raw_dir == PROJECT_ROOT / "data" / "raw" / "mind_small"


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(PROJECT_ROOT / "configs" / "does_not_exist.yaml")


def test_load_config_missing_key_raises(tmp_path):
    # Write a broken config under a fake "configs/" folder so load_config's
    # project-root resolution (config_path.parent.parent) still has a real
    # Path to compute against, even though nothing else in tmp_path exists.
    fake_configs_dir = tmp_path / "configs"
    fake_configs_dir.mkdir()
    broken_config = fake_configs_dir / "broken.yaml"
    broken_config.write_text("dataset_name: broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        load_config(broken_config)
