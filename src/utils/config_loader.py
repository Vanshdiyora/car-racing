"""YAML configuration loader with validation and defaults merging."""

from __future__ import annotations

import pathlib
from typing import Any

import yaml


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at root of {path}, got {type(data).__name__}")
    return data


def load_train_config(path: str = "configs/train_config.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_env_config(path: str = "configs/env_config.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_race_config(path: str = "configs/race_config.yaml") -> dict[str, Any]:
    return load_yaml(path)


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* (non-destructive)."""
    merged = base.copy()
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
