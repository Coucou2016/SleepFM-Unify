"""Validate SleepFM index.json + .npy epoch files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sleepfm.data.splits import assert_disjoint_splits, downstream_isolation_ok


def validate_dataset(data_dir: str | Path, strict_participants: bool = False) -> Tuple[bool, List[str]]:
    """
    Return (ok, messages). Raises no exception; callers decide on failure.
    """
    data_dir = Path(data_dir)
    messages: List[str] = []
    ok = True

    index_path = data_dir / "index.json"
    if not index_path.is_file():
        return False, [f"Missing {index_path}"]

    with open(index_path, encoding="utf-8") as f:
        payload = json.load(f)

    if "meta" not in payload or "splits" not in payload:
        ok = False
        messages.append("index.json must contain 'meta' and 'splits'")

    meta = payload.get("meta", {})
    splits = payload.get("splits", {})
    channels = meta.get("channels", {})
    clip_length = meta.get("clip_length")
    slices = meta.get("channel_slices", {})

    for split_name, entries in splits.items():
        for entry in entries:
            rel = entry.get("path")
            if not rel:
                ok = False
                messages.append(f"{split_name}: entry missing 'path'")
                continue
            fpath = data_dir / rel
            if not fpath.is_file():
                ok = False
                messages.append(f"Missing file: {fpath}")
                continue
            arr = np.load(fpath)
            expected_c = sum(channels.get(m, 0) for m in ("bas", "ecg", "respiratory"))
            if arr.ndim != 2:
                ok = False
                messages.append(f"{rel}: expected 2D array, got shape {arr.shape}")
            elif expected_c and arr.shape[0] != expected_c:
                ok = False
                messages.append(
                    f"{rel}: channel dim {arr.shape[0]} != expected {expected_c}"
                )
            elif clip_length and arr.shape[1] != clip_length:
                ok = False
                messages.append(
                    f"{rel}: time dim {arr.shape[1]} != meta clip_length {clip_length}"
                )
            for key in ("stage_id", "apnea"):
                if key not in entry:
                    ok = False
                    messages.append(f"{split_name}/{rel}: missing '{key}'")

    # Slice consistency
    if slices and channels:
        for mod, sl in slices.items():
            if mod in channels and sl[1] - sl[0] != channels[mod]:
                ok = False
                messages.append(f"channel_slices[{mod}] width mismatch vs channels[{mod}]")

    # Paper isolation: pretrain vs train/test
    try:
        iso = downstream_isolation_ok(data_dir)
        for name, passed in iso.items():
            if not passed:
                ok = False
                messages.append(f"Split isolation failed: {name}")
    except Exception as exc:
        ok = False
        messages.append(f"Split isolation check error: {exc}")

    if strict_participants and not meta.get("participant_level_splits"):
        ok = False
        messages.append("strict_participants: dataset lacks participant_level_splits in meta")

    if ok:
        messages.append("Dataset validation passed.")
    return ok, messages
