"""Shared helpers: project paths, config loading, Objaverse download location."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(rel):
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def use_project_objaverse_dir(cfg):
    """Make the objaverse package store annotations + GLBs inside data/raw/objaverse
    instead of ~/.objaverse. Must be called before any objaverse.load_* call."""
    import objaverse

    d = project_path(cfg["data"]["raw_dir"])
    d.mkdir(parents=True, exist_ok=True)
    objaverse.BASE_PATH = str(d)
    objaverse._VERSIONED_PATH = str(d / "hf-objaverse-v1")
    return objaverse
