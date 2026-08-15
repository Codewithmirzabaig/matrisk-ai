"""Shared configuration and reproducibility utilities."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML configuration from the repository default or a supplied path."""
    config_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds used by the lightweight production baseline."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

