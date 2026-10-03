"""Configuration loader for the causal rideshare project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the project root directory.

    Walks up from the current working directory looking for ``configs/``
    as a heuristic.  Falls back to CWD.
    """
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "configs").is_dir():
            return parent
    return cwd


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML simulation config.

    Parameters
    ----------
    config_path : str or Path, optional
        Absolute or relative path to the YAML file.  Defaults to
        ``<project_root>/configs/simulation.yaml``.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "simulation.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)
