"""Split integrity checks for SleepFM datasets (participant / epoch isolation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

import numpy as np

# Isolation check value: True=pass, False=fail, "n/a"=field absent (not a silent pass).
IsolationStatus = Union[bool, str]

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
    *,
    require_ids: bool = False,
) -> Tuple[bool, Set[str]]:
    """
    Return (has_overlap, overlapping_ids) between two splits.

    by: "path", "participant_id", or "night_id" (night_id/recording_id composite).

    When ``require_ids=True`` and ``by == "participant_id"``, empty / missing
    participant_id sets on a non-empty split raise ``RuntimeError`` (fail-closed)
    instead of silently reporting no overlap.
    """
    payload = load_index(data_dir)
    splits = payload["splits"]
    if split_a not in splits or split_b not in splits:
        raise KeyError(f"Missing split: {split_a!r} or {split_b!r}")

    if by == "path":
        a = entry_paths(splits[split_a])
        b = entry_paths(splits[split_b])
    elif by == "participant_id":
        entries_a = splits[split_a]
        entries_b = splits[split_b]
        a = entry_participant_ids(entries_a)
        b = entry_participant_ids(entries_b)
        if require_ids:
            if entries_a and not a:
                raise RuntimeError(
                    f"Strict isolation: split {split_a!r} has {len(entries_a)} entries "
                    "but no participant_id fields (fail-closed)"
                )
            if entries_b and not b:
                raise RuntimeError(
                    f"Strict isolation: split {split_b!r} has {len(entries_b)} entries "
                    "but no participant_id fields (fail-closed)"
                )
            # Also fail if some entries are missing participant_id while others have it.
            missing_a = sum(1 for e in entries_a if e.get("participant_id") is None)
            missing_b = sum(1 for e in entries_b if e.get("participant_id") is None)
            if missing_a:
                raise RuntimeError(
                    f"Strict isolation: split {split_a!r} has {missing_a} entries "
                    "missing participant_id (fail-closed)"
                )
            if missing_b:
                raise RuntimeError(
                    f"Strict isolation: split {split_b!r} has {missing_b} entries "
                    "missing participant_id (fail-closed)"
                )
        elif not a or not b:
            return False, set()
    elif by == "night_id":
        a = entry_night_ids(splits[split_a])
        b = entry_night_ids(splits[split_b])
        # Missing night_id is not proof of isolation — callers must treat empty
        # sets as N/A (see downstream_isolation_ok), not as a silent PASS.
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
    *,
    require_ids: bool = False,
) -> None:
    for sa, sb in pairs:
        has, overlap = split_overlap(
            data_dir, sa, sb, by=by, require_ids=require_ids and by == "participant_id"
        )
        if has:
            sample = sorted(overlap)[:5]
            raise AssertionError(
                f"Split leak: {sa!r} vs {sb!r} share {len(overlap)} {by}(s), e.g. {sample}"
            )


def _existing_pairs(data_dir: str | Path, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    payload = load_index(data_dir)
    splits = payload.get("splits", {})
    return [(a, b) for a, b in pairs if a in splits and b in splits]


def downstream_isolation_ok(
    data_dir: str | Path,
    *,
    strict_participant_ids: bool = False,
) -> Dict[str, IsolationStatus]:
    """Full cohort separation across pretrain/valid/train/test (paths + participants + nights).

    Values are ``True`` (pass), ``False`` (fail/leak), or ``\"n/a\"`` when the
    checked id field is absent (night_id only — **not** a silent PASS).

    When ``strict_participant_ids=True``, missing ``participant_id`` fails the
    corresponding check (and ``assert_paper_isolation`` raises).
    """
    checks: Dict[str, IsolationStatus] = {}
    payload = load_index(data_dir)
    splits = payload.get("splits", {})
    pairs = _existing_pairs(data_dir, PAPER_SPLIT_PAIRS)
    for sa, sb in pairs:
        for by in ("path", "participant_id", "night_id"):
            key = f"{sa}_vs_{sb}_{by}"
            try:
                if by == "night_id":
                    a_ids = entry_night_ids(splits.get(sa, []))
                    b_ids = entry_night_ids(splits.get(sb, []))
                    if not a_ids or not b_ids:
                        checks[key] = "n/a"
                        continue
                has, _ = split_overlap(
                    data_dir,
                    sa,
                    sb,
                    by=by,
                    require_ids=strict_participant_ids and by == "participant_id",
                )
                checks[key] = not has
            except RuntimeError:
                if strict_participant_ids and by == "participant_id":
                    checks[key] = False
                else:
                    raise
            except KeyError:
                checks[key] = True
    return checks


def assert_paper_isolation(
    data_dir: str | Path,
    *,
    strict: bool = True,
) -> Dict[str, IsolationStatus]:
    """
    Verify path / participant / night isolation for all paper split pairs.

    When ``strict=True`` (paper / evaluate / paper-suite mode), raise
    ``RuntimeError`` on any leak **or** missing participant_id fields
    instead of warning-and-continue / silent True.

    ``night_id`` checks that are ``\"n/a\"`` (field absent) do **not** fail
    isolation; they are reported explicitly and skipped as proof of isolation.
    """
    checks = downstream_isolation_ok(data_dir, strict_participant_ids=strict)
    failed = [k for k, v in checks.items() if v is False]
    if failed and strict:
        details = []
        for key in failed:
            by = "path"
            try:
                left, rest = key.split("_vs_", 1)
                for suffix in ("_participant_id", "_night_id", "_path"):
                    if rest.endswith(suffix):
                        right = rest[: -len(suffix)]
                        by = suffix.lstrip("_")
                        break
                else:
                    right = rest
                try:
                    has, overlap = split_overlap(
                        data_dir,
                        left,
                        right,
                        by=by,
                        require_ids=by == "participant_id",
                    )
                    sample = sorted(overlap)[:5]
                    details.append(
                        f"{left} vs {right} ({by}): {len(overlap)} e.g. {sample}"
                    )
                except RuntimeError as exc:
                    details.append(f"{left} vs {right} ({by}): {exc}")
            except Exception as exc:
                details.append(f"{key}: {exc}")
        raise RuntimeError(
            "Split isolation leak (strict/paper mode): " + "; ".join(details)
        )
    return checks
