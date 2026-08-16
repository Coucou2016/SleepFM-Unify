"""Night-level windows of SleepFM epochs (participant / recording sequences)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset



def night_key(entry: dict) -> Tuple[str, str]:
    pid = str(entry.get("participant_id") or "unknown")
    nid = entry.get("night_id") or entry.get("recording_id") or pid
    return pid, str(nid)


def epoch_sort_key(entry: dict) -> Tuple[int, str]:
    idx = entry.get("epoch_index")
    if idx is None:
        idx = entry.get("epoch")
    try:
        return int(idx), str(entry.get("path", ""))
    except (TypeError, ValueError):
        return 0, str(entry.get("path", ""))


def group_entries_by_night(entries: Sequence[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for entry in entries:
        groups[night_key(entry)].append(entry)
    for key in groups:
        groups[key] = sorted(groups[key], key=epoch_sort_key)
        for i, rec in enumerate(groups[key]):
            if rec.get("epoch_index") is None:
                rec = dict(rec)
                rec["epoch_index"] = i
                groups[key][i] = rec
    return dict(groups)


def night_summary_from_entries(
    entries: Sequence[dict],
    epoch_seconds: float = 30.0,
) -> dict:
    """Night placeholders from epoch labels in index.json.

    ``apnea_epoch_rate`` is apnea-positive **epochs per hour of recording**
    (binary apnea flags / hours). It is **not** clinical AHI (events/hour with
    AASM scoring). Key ``ahi`` is kept as a deprecated alias of the same value
    for backward-compatible probes; do not claim clinical AHI in papers.
    """
    n = len(entries)
    if n == 0:
        return {
            "n_epochs": 0,
            "apnea_epoch_rate": float("nan"),
            "ahi": float("nan"),
            "sleep_efficiency": float("nan"),
            "ahi_bin": -1,
            "ahi_definition": "placeholder_apnea_epoch_rate_not_clinical_ahi",
        }
    stages = [int(e.get("stage_id", 0)) for e in entries]
    apneas = [int(e.get("apnea", 0)) for e in entries]
    hours = max(n * float(epoch_seconds) / 3600.0, 1e-6)
    rate = float(sum(apneas) / hours)
    sleep_eff = float(sum(s != 0 for s in stages) / n)
    # Bins reuse common clinical AHI cut-points only as a coarse placeholder.
    if rate < 5:
        ahi_bin = 0
    elif rate < 15:
        ahi_bin = 1
    elif rate < 30:
        ahi_bin = 2
    else:
        ahi_bin = 3
    return {
        "n_epochs": n,
        "apnea_epoch_rate": rate,
        "ahi": rate,  # deprecated alias — not clinical AHI
        "sleep_efficiency": sleep_eff,
        "ahi_bin": ahi_bin,
        "ahi_definition": "placeholder_apnea_epoch_rate_not_clinical_ahi",
        "participant_id": entries[0].get("participant_id"),
        "night_id": night_key(entries[0])[1],
    }


def collate_night(batch: List[dict]) -> dict:
    skip = {"labels", "participant_id", "night_id"}
    keys = [k for k in batch[0].keys() if k not in skip and torch.is_tensor(batch[0][k])]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    if "labels" in batch[0] and batch[0]["labels"] is not None:
        out["stage_id"] = torch.stack([b["labels"]["stage_id"] for b in batch], dim=0)
        out["apnea"] = torch.stack([b["labels"]["apnea"] for b in batch], dim=0)
    out["participant_id"] = [b.get("participant_id") for b in batch]
    out["night_id"] = [b.get("night_id") for b in batch]
    return out


class NightSequenceDataset(Dataset):
    """Sliding windows of raw epochs grouped by participant/night."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "pretrain",
        window: int = 32,
        stride: Optional[int] = None,
        min_len: int = 2,
        return_labels: bool = False,
    ):
        data_dir = Path(data_dir)
        with open(data_dir / "index.json", encoding="utf-8") as f:
            payload = json.load(f)
        self.meta = payload["meta"]
        if split not in payload["splits"]:
            available = ", ".join(sorted(payload["splits"]))
            raise KeyError(f"Unknown split '{split}'. Available: {available}")
        self.entries = payload["splits"][split]
        self.data_dir = data_dir
        self.window = int(window)
        self.stride = int(stride or window)
        self.min_len = int(min_len)
        self.return_labels = return_labels
        slices = self.meta["channel_slices"]
        self.slices = {k: slice(v[0], v[1]) for k, v in slices.items()}
        self.groups = group_entries_by_night(self.entries)
        self.windows: List[Tuple[Tuple[str, str], int]] = []
        for key, recs in self.groups.items():
            n = len(recs)
            if n < self.min_len:
                continue
            if n <= self.window:
                self.windows.append((key, 0))
                continue
            start = 0
            while start < n:
                self.windows.append((key, start))
                if start + self.window >= n:
                    break
                start += self.stride

    def __len__(self) -> int:
        return len(self.windows)

    def _load_entry(self, entry: dict):
        arr = np.load(self.data_dir / entry["path"]).astype(np.float32)
        missing = set(entry.get("missing_modalities") or [])
        sample = {}
        present = []
        t = arr.shape[1]
        for name in ("bas", "ecg", "respiratory"):
            sl = self.slices[name]
            if name in missing:
                sample[name] = torch.zeros(sl.stop - sl.start, t, dtype=torch.float32)
                present.append(0.0)
            else:
                sample[name] = torch.from_numpy(arr[sl])
                present.append(1.0)
        sample["present_mask"] = torch.tensor(present, dtype=torch.float32)
        sample["labels"] = {
            "stage_id": int(entry.get("stage_id", 0)),
            "apnea": int(entry.get("apnea", 0)),
        }
        return sample

    def __getitem__(self, idx: int):
        key, start = self.windows[idx]
        recs = self.groups[key]
        chunk = recs[start : start + self.window]
        loaded = [self._load_entry(e) for e in chunk]
        l_true = len(loaded)
        pad_n = self.window - l_true
        if pad_n > 0:
            proto = loaded[-1]
            for _ in range(pad_n):
                loaded.append(
                    {
                        "bas": torch.zeros_like(proto["bas"]),
                        "ecg": torch.zeros_like(proto["ecg"]),
                        "respiratory": torch.zeros_like(proto["respiratory"]),
                        "present_mask": torch.zeros_like(proto["present_mask"]),
                        "labels": {"stage_id": 0, "apnea": 0},
                    }
                )
        sample = {
            "bas": torch.stack([x["bas"] for x in loaded], dim=0),
            "ecg": torch.stack([x["ecg"] for x in loaded], dim=0),
            "respiratory": torch.stack([x["respiratory"] for x in loaded], dim=0),
            "present_mask": torch.stack([x["present_mask"] for x in loaded], dim=0),
            "padding_mask": torch.zeros(self.window, dtype=torch.bool),
            "participant_id": key[0],
            "night_id": key[1],
        }
        if pad_n > 0:
            sample["padding_mask"][l_true:] = True
        if self.return_labels:
            sample["labels"] = {
                "stage_id": torch.tensor([x["labels"]["stage_id"] for x in loaded], dtype=torch.long),
                "apnea": torch.tensor([x["labels"]["apnea"] for x in loaded], dtype=torch.long),
            }
        return sample
