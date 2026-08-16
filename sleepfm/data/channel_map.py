"""YAML channel tables: map heterogeneous PSG leads onto BAS / ECG / respiratory slots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

MODALITY_ORDER = ("bas", "ecg", "respiratory")

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES = {
    "cinc2018": _REPO_ROOT / "configs" / "channels" / "cinc2018.yaml",
    "cinc": _REPO_ROOT / "configs" / "channels" / "cinc2018.yaml",
    "shhs": _REPO_ROOT / "configs" / "channels" / "shhs.yaml",
    "mesa": _REPO_ROOT / "configs" / "channels" / "mesa.yaml",
}


@dataclass
class ChannelTable:
    dataset: str
    target_fs: int = 256
    clip_seconds: int = 30
    target_channels: Dict[str, int] = field(
        default_factory=lambda: {"bas": 10, "ecg": 2, "respiratory": 7}
    )
    slots: Dict[str, List[List[str]]] = field(default_factory=dict)
    missing_slots: Dict[str, List[str]] = field(default_factory=dict)
    annotation_format: str = "auto"
    notes: str = ""
    access: str = ""

    def n_channels(self, modality: str) -> int:
        return int(self.target_channels[modality])

    def total_channels(self) -> int:
        return sum(self.n_channels(m) for m in MODALITY_ORDER)


def load_channel_table(path_or_dataset: str | Path) -> ChannelTable:
    """Load a channel table from a YAML path or a dataset alias (cinc2018/shhs/mesa)."""
    key = str(path_or_dataset).lower()
    if key in DEFAULT_TABLES:
        path = DEFAULT_TABLES[key]
    else:
        path = Path(path_or_dataset)
        if not path.is_file():
            cand = _REPO_ROOT / path
            path = cand if cand.is_file() else path
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    slots: Dict[str, List[List[str]]] = {}
    for mod in MODALITY_ORDER:
        rows = raw.get(mod, [])
        parsed: List[List[str]] = []
        for row in rows:
            if row is None:
                parsed.append([])
            elif isinstance(row, str):
                parsed.append([row])
            else:
                parsed.append([str(x) for x in row if x])
        slots[mod] = parsed
    return ChannelTable(
        dataset=str(raw.get("dataset", path.stem)),
        target_fs=int(raw.get("target_fs", 256)),
        clip_seconds=int(raw.get("clip_seconds", 30)),
        target_channels=dict(raw.get("target_channels") or {"bas": 10, "ecg": 2, "respiratory": 7}),
        slots=slots,
        missing_slots={k: list(v) for k, v in (raw.get("missing_slots") or {}).items()},
        annotation_format=str(raw.get("annotation_format", "auto")),
        notes=str(raw.get("notes") or "").strip(),
        access=str(raw.get("access") or "").strip(),
    )


def _norm_name(name: str) -> str:
    return "".join(ch for ch in name.lower().strip() if ch.isalnum())


def resolve_channel(available: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    """Return the first available channel matching any alias (exact, then fuzzy)."""
    if not aliases:
        return None
    by_norm = {_norm_name(a): a for a in available}
    by_lower = {a.lower().strip(): a for a in available}
    for alias in aliases:
        alias = str(alias).strip()
        if not alias:
            continue
        if alias in available:
            return alias
        low = alias.lower().strip()
        if low in by_lower:
            return by_lower[low]
        n = _norm_name(alias)
        if n and n in by_norm:
            return by_norm[n]
    for alias in aliases:
        n = _norm_name(str(alias))
        if not n:
            continue
        for avail_n, orig in by_norm.items():
            if n == avail_n or (len(n) >= 3 and (n in avail_n or avail_n in n)):
                return orig
    return None


def map_recording_channels(
    available: Sequence[str],
    table: ChannelTable,
) -> Tuple[Dict[str, List[Optional[str]]], List[str]]:
    """
    Map EDF/MAT channel names onto fixed SleepFM slots.

    Returns (mapping, warnings). mapping[mod][i] is a source name or None (pad).
    """
    mapping: Dict[str, List[Optional[str]]] = {}
    warnings: List[str] = []
    used = set()
    for mod in MODALITY_ORDER:
        n = table.n_channels(mod)
        slot_aliases = list(table.slots.get(mod, []))
        while len(slot_aliases) < n:
            slot_aliases.append([])
        chosen: List[Optional[str]] = []
        remaining = [n for n in available if n not in used]
        for i in range(n):
            hit = resolve_channel(remaining, slot_aliases[i])
            if hit is not None:
                used.add(hit)
                remaining = [n for n in available if n not in used]
            else:
                if slot_aliases[i]:
                    warnings.append(
                        f"{mod} slot {i}: no match for aliases {slot_aliases[i]} "
                        f"among {list(available)}"
                    )
            chosen.append(hit)
        mapping[mod] = chosen
    return mapping, warnings


def document_missing(table: ChannelTable) -> Dict[str, List[str]]:
    """Slots that are always padded, plus YAML-documented missing physiology."""
    documented = {k: list(v) for k, v in table.missing_slots.items()}
    for mod in MODALITY_ORDER:
        pads = [
            i
            for i, aliases in enumerate(table.slots.get(mod, []))
            if not aliases and i < table.n_channels(mod)
        ]
        if pads:
            documented.setdefault(mod, [])
            documented[mod].append(f"zero-padded slot indices {pads}")
    return documented
