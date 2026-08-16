from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class OSSEDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"OSSE data not found: {path}. Run scripts/fourdvarnet_generate_data.py first.")
        data = np.load(path)
        self.y_obs = torch.from_numpy(data["y_obs"])
        self.obs_mask = torch.from_numpy(data["obs_mask"])
        self.z_sst = torch.from_numpy(data["z_sst"])
        self.ssh = torch.from_numpy(data["ssh"])
        self.u = torch.from_numpy(data["u"])
        self.v = torch.from_numpy(data["v"])
        self.ssh_seq = torch.from_numpy(data["ssh_seq"])
        self.u_seq = torch.from_numpy(data["u_seq"])
        self.v_seq = torch.from_numpy(data["v_seq"])

    def __len__(self) -> int:
        return self.y_obs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "y_obs": self.y_obs[idx],
            "obs_mask": self.obs_mask[idx],
            "z_sst": self.z_sst[idx],
            "ssh": self.ssh[idx],
            "u": self.u[idx],
            "v": self.v[idx],
            "ssh_seq": self.ssh_seq[idx],
            "u_seq": self.u_seq[idx],
            "v_seq": self.v_seq[idx],
        }
