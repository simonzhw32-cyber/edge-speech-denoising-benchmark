"""Configuration loading shared by the script entry points."""

from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, _seen=None):
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    seen.add(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    parent = config.pop("extends", None)
    if parent:
        config = _merge(load_config(path.parent / parent, seen), config)
    return config


def phase1_entrypoint(task):
    import argparse

    parser = argparse.ArgumentParser(description=f"{task}: Phase 1 interface only")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("phase") != 1:
        parser.error("Only Phase 1 is implemented.")
    print(f"{task}: configuration loaded; implementation pending. No work executed.")
