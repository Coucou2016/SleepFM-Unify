from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overlay onto a copy of base (overlay wins)."""
    out: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_config_path(path: str | Path, relative_to: Path | None = None) -> Path:
    path = Path(path)
    candidates = []
    if path.is_file():
        return path
    if relative_to is not None:
        candidates.append(relative_to / path)
    candidates.append(Path.cwd() / path)
    root = Path(__file__).resolve().parents[2]
    candidates.append(root / path)
    for cand in candidates:
        if cand.is_file():
            return cand
    return path


def load_config(path: str | Path) -> Dict[str, Any]:
    path = _resolve_config_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base", None)
    if base_ref:
        base_path = _resolve_config_path(base_ref, relative_to=path.parent)
        cfg = deep_merge(load_config(base_path), cfg)
    return cfg
