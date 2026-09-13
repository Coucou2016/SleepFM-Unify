"""PyTorch dataset for multi-modal sleep epochs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset


def collate_multimodal(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    skip = {"labels", "participant_id", "night_id", "channel_mask"}
    keys = [k for k in batch[0].keys() if k not in skip and torch.is_tensor(batch[0][k])]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    if "participant_id" in batch[0]:
        out["participant_id"] = [b.get("participant_id") for b in batch]
    if "channel_mask" in batch[0] and isinstance(batch[0]["channel_mask"], dict):
        # Stack per-modality channel masks when present (optional schema).
        cm_out = {}
        for mod in ("bas", "ecg", "respiratory"):
            if mod in batch[0]["channel_mask"] and torch.is_tensor(batch[0]["channel_mask"][mod]):
                cm_out[mod] = torch.stack([b["channel_mask"][mod] for b in batch], dim=0)
        if cm_out:
            out["channel_mask"] = cm_out
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
        self.channels = self.meta.get("channels", {})

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        arr = np.load(self.data_dir / entry["path"]).astype(np.float32)
        missing = {str(m) for m in (entry.get("missing_modalities") or [])}
        sample = {}
        present = []
        t = int(arr.shape[1])
        channel_mask = {}
        entry_cm = entry.get("channel_mask") or {}
        for name in ("bas", "ecg", "respiratory"):
            sl = self.slices[name]
            n_ch = sl.stop - sl.start
            if name in missing:
                sample[name] = torch.zeros(n_ch, t, dtype=torch.float32)
                present.append(0.0)
                channel_mask[name] = torch.zeros(n_ch, dtype=torch.float32)
            else:
                sample[name] = torch.from_numpy(arr[sl])
                present.append(1.0)
                if name in entry_cm:
                    cm = entry_cm[name]
                    channel_mask[name] = torch.tensor(cm, dtype=torch.float32)
                    if channel_mask[name].numel() != n_ch:
                        channel_mask[name] = torch.ones(n_ch, dtype=torch.float32)
                else:
                    # Default: all channels present (zero-pad slots still marked 1 unless listed).
                    channel_mask[name] = torch.ones(n_ch, dtype=torch.float32)
                    pad_slots = (self.meta.get("missing_slots") or {}).get(name) or []
                    for slot in pad_slots:
                        if isinstance(slot, int) and 0 <= slot < n_ch:
                            channel_mask[name][slot] = 0.0
        sample["present_mask"] = torch.tensor(present, dtype=torch.float32)
        sample["channel_mask"] = channel_mask
        if self.return_labels:
            sample["labels"] = {
                "stage_id": int(entry["stage_id"]),
                "apnea": int(entry["apnea"]),
            }
        return sample
