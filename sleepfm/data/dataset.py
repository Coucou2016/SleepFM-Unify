"""PyTorch dataset for multi-modal sleep epochs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset


def collate_multimodal(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    skip = {"labels", "participant_id", "night_id"}
    keys = [k for k in batch[0].keys() if k not in skip and torch.is_tensor(batch[0][k])]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    if "participant_id" in batch[0]:
        out["participant_id"] = [b.get("participant_id") for b in batch]
    return out


class SleepEpochDataset(Dataset):
    """Load BAS, ECG, respiratory tensors from index.json + .npy files."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        return_labels: bool = False,
        participant_ids: Optional[Iterable[str]] = None,
    ):
        data_dir = Path(data_dir)
        with open(data_dir / "index.json", encoding="utf-8") as f:
            payload = json.load(f)
        self.meta = payload["meta"]
        splits = payload["splits"]
        if split not in splits:
            available = ", ".join(sorted(splits))
            raise KeyError(
                f"Unknown split '{split}' in {data_dir / 'index.json'}. "
                f"Available: {available}"
            )
        self.entries = splits[split]
        if participant_ids is not None:
            allow: Set[str] = {str(p) for p in participant_ids}
            self.entries = [e for e in self.entries if str(e.get("participant_id")) in allow]
        self.data_dir = data_dir
        self.return_labels = return_labels
        slices = self.meta["channel_slices"]
        self.slices = {k: slice(v[0], v[1]) for k, v in slices.items()}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        arr = np.load(self.data_dir / entry["path"]).astype(np.float32)
        missing = {str(m) for m in (entry.get("missing_modalities") or [])}
        sample = {}
        present = []
        t = int(arr.shape[1])
        for name in ("bas", "ecg", "respiratory"):
            sl = self.slices[name]
            if name in missing:
                sample[name] = torch.zeros(sl.stop - sl.start, t, dtype=torch.float32)
                present.append(0.0)
            else:
                sample[name] = torch.from_numpy(arr[sl])
                present.append(1.0)
        sample["present_mask"] = torch.tensor(present, dtype=torch.float32)
        if self.return_labels:
            sample["labels"] = {
                "stage_id": int(entry["stage_id"]),
                "apnea": int(entry["apnea"]),
            }
        return sample
