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

# All pairs that must be disjoint for paper/strict evaluation.
PAPER_SPLIT_PAIRS: List[Tuple[str, str]] = [
    ("pretrain", "valid"),
    ("pretrain", "train"),
    ("pretrain", "test"),
    ("valid", "train"),
    ("valid", "test"),
    ("train", "test"),
]


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


def entry_night_ids(entries: Iterable[dict]) -> Set[str]:
    """Night / recording identifiers when present (composite with participant)."""
    ids: Set[str] = set()
    for e in entries:
        nid = e.get("night_id") or e.get("recording_id")
        if nid is None:
            continue
        pid = e.get("participant_id")
        if pid is not None:
            ids.add(f"{pid}::{nid}")
        else:
            ids.add(str(nid))
    return ids


def split_overlap(
    data_dir: str | Path,
    split_a: str,
    split_b: str,
    by: str = "path",
) -> Tuple[bool, Set[str]]:
    """
    Return (has_overlap, overlapping_ids) between two splits.

    by: "path", "participant_id", or "night_id" (night_id/recording_id composite).
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
    elif by == "night_id":
        a = entry_night_ids(splits[split_a])
        b = entry_night_ids(splits[split_b])
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


def _existing_pairs(data_dir: str | Path, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    payload = load_index(data_dir)
    splits = payload.get("splits", {})
    return [(a, b) for a, b in pairs if a in splits and b in splits]


def downstream_isolation_ok(data_dir: str | Path) -> Dict[str, bool]:
    """Full cohort separation across pretrain/valid/train/test (paths + participants + nights)."""
    checks: Dict[str, bool] = {}
    pairs = _existing_pairs(data_dir, PAPER_SPLIT_PAIRS)
    for sa, sb in pairs:
        for by in ("path", "participant_id", "night_id"):
            key = f"{sa}_vs_{sb}_{by}"
            try:
                has, _ = split_overlap(data_dir, sa, sb, by=by)
                # night_id: no overlap only matters when both sides have night ids
                checks[key] = not has
            except KeyError:
                checks[key] = True
    return checks


def assert_paper_isolation(
    data_dir: str | Path,
    *,
    strict: bool = True,
) -> Dict[str, bool]:
    """
    Verify path / participant / night isolation for all paper split pairs.

    When ``strict=True`` (paper / evaluate / paper-suite mode), raise
    ``RuntimeError`` on any leak instead of warning-and-continue.
    """
    checks = downstream_isolation_ok(data_dir)
    failed = [k for k, v in checks.items() if not v]
    if failed and strict:
        details = []
        for key in failed:
            # key like "train_vs_test_participant_id"
            parts = key.rsplit("_", 2)
            if len(parts) >= 3 and parts[-1] in ("path", "id") and parts[-2] in (
                "participant",
                "night",
            ):
                by = f"{parts[-2]}_{parts[-1]}" if parts[-1] == "id" else parts[-1]
                # Reconstruct pair from PAPER_SPLIT_PAIRS match
                by = "participant_id" if "participant" in key else (
                    "night_id" if "night" in key else "path"
                )
            else:
                by = "path"
            # Parse "a_vs_b_by"
            try:
                left, rest = key.split("_vs_", 1)
                # rest ends with _path / _participant_id / _night_id
                for suffix in ("_participant_id", "_night_id", "_path"):
                    if rest.endswith(suffix):
                        right = rest[: -len(suffix)]
                        by = suffix.lstrip("_")
                        break
                else:
                    right = rest
                has, overlap = split_overlap(data_dir, left, right, by=by)
                sample = sorted(overlap)[:5]
                details.append(f"{left} vs {right} ({by}): {len(overlap)} e.g. {sample}")
            except Exception as exc:
                details.append(f"{key}: {exc}")
        raise RuntimeError(
            "Split isolation leak (strict/paper mode): " + "; ".join(details)
        )
    return checks
