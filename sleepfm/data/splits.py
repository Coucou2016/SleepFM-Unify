"""Split integrity checks for SleepFM datasets (participant / epoch isolation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

DEFAULT_SPLIT_FRACTIONS: Dict[str, float] = {
    "pretrain": 0.70,
    "valid": 0.10,
    "train": 0.10,
    "test": 0.10,
}


def assign_participant_splits(
    participant_ids: Iterable[str],
    fractions: Optional[Mapping[str, float]] = None,
    seed: int = 42,
    split_order: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Map each participant_id to a split name (participant-level, disjoint).

    ``fractions`` values should sum to ~1. Every split with a positive fraction
    receives at least one participant when possible.
    """
    fractions = dict(fractions or DEFAULT_SPLIT_FRACTIONS)
    split_order = split_order or ["pretrain", "valid", "train", "test"]
    pids = sorted({str(p) for p in participant_ids})
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    names = [s for s in split_order if fractions.get(s, 0) > 0]
    extra = [s for s in fractions if s not in names and fractions[s] > 0]
    names.extend(extra)
    if not names:
        raise ValueError("No splits with positive fractions")
    n = len(pids)
    if n < len(names):
        raise ValueError(
            f"Need at least {len(names)} participants for splits {names}, got {n}"
        )

    remaining = n
    alloc: Dict[str, int] = {}
    for i, split in enumerate(names):
        leftover_splits = len(names) - i - 1
        if leftover_splits == 0:
            alloc[split] = remaining
            break
        want = max(1, int(round(float(fractions[split]) * n)))
        want = min(want, remaining - leftover_splits)
        alloc[split] = want
        remaining -= want

    mapping: Dict[str, str] = {}
    idx = 0
    for split in names:
        for _ in range(alloc[split]):
            mapping[pids[idx]] = split
            idx += 1
    return mapping


def load_index(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    with open(data_dir / "index.json", encoding="utf-8") as f:
        return json.load(f)


def entry_paths(entries: Iterable[dict]) -> Set[str]:
    return {e["path"] for e in entries}


def entry_participant_ids(entries: Iterable[dict]) -> Set[str]:
    ids: Set[str] = set()
    for e in entries:
        pid = e.get("participant_id")
        if pid is not None:
            ids.add(str(pid))
    return ids


def split_overlap(
    data_dir: str | Path,
    split_a: str,
    split_b: str,
    by: str = "path",
) -> Tuple[bool, Set[str]]:
    """
    Return (has_overlap, overlapping_ids) between two splits.

    by: "path" (epoch files) or "participant_id" (requires participant_id in index).
    """
    payload = load_index(data_dir)
    splits = payload["splits"]
    if split_a not in splits or split_b not in splits:
        raise KeyError(f"Missing split: {split_a!r} or {split_b!r}")

    if by == "path":
        a = entry_paths(splits[split_a])
        b = entry_paths(splits[split_b])
    elif by == "participant_id":
        a = entry_participant_ids(splits[split_a])
        b = entry_participant_ids(splits[split_b])
        if not a or not b:
            return False, set()
    else:
        raise ValueError(f"Unknown by={by!r}")

    overlap = a & b
    return bool(overlap), overlap


def assert_disjoint_splits(
    data_dir: str | Path,
    pairs: List[Tuple[str, str]],
    by: str = "path",
) -> None:
    for sa, sb in pairs:
        has, overlap = split_overlap(data_dir, sa, sb, by=by)
        if has:
            sample = sorted(overlap)[:5]
            raise AssertionError(
                f"Split leak: {sa!r} vs {sb!r} share {len(overlap)} {by}(s), e.g. {sample}"
            )


def downstream_isolation_ok(data_dir: str | Path) -> Dict[str, bool]:
    """Paper cohort separation: pretrain must not overlap train/test (paths + participants)."""
    checks = {}
    for other in ("train", "test"):
        checks[f"pretrain_vs_{other}_path"], _ = split_overlap(
            data_dir, "pretrain", other, by="path"
        )
        checks[f"pretrain_vs_{other}_path"] = not checks[f"pretrain_vs_{other}_path"]
        pid_overlap, _ = split_overlap(data_dir, "pretrain", other, by="participant_id")
        checks[f"pretrain_vs_{other}_participant"] = not pid_overlap
    checks["train_vs_test_path"], _ = split_overlap(data_dir, "train", "test", by="path")
    checks["train_vs_test_path"] = not checks["train_vs_test_path"]
    return checks
