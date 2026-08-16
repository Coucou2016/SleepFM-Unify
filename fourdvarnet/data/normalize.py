"""Per-field normalization (train stats) for stable 4DVarNet training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormStats:
    ssh_mean: float
    ssh_std: float
    u_mean: float
    u_std: float
    v_mean: float
    v_std: float
    sst_mean: float
    sst_std: float

    @classmethod
    def from_train_samples(cls, samples: list[dict[str, np.ndarray]]) -> NormStats:
        ssh = np.concatenate([s["ssh_seq"].reshape(-1) for s in samples])
        u = np.concatenate([s["u_seq"].reshape(-1) for s in samples])
        v = np.concatenate([s["v_seq"].reshape(-1) for s in samples])
        sst = np.concatenate([s["z_sst"].reshape(-1) for s in samples])

        def _ms(a: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
            m = float(a.mean())
            s = float(a.std())
            return m, max(s, eps)

        sm, ss = _ms(ssh)
        um, us = _ms(u)
        vm, vs = _ms(v)
        tm, ts = _ms(sst)
        return cls(sm, ss, um, us, vm, vs, tm, ts)

    def to_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> NormStats:
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__})


def _norm(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def _denorm(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x * std + mean).astype(np.float32)


def apply_norm_sample(sample: dict[str, np.ndarray], stats: NormStats) -> dict[str, np.ndarray]:
    """Normalize a packed window; observation u/v channels stay zero (never observed)."""
    out = dict(sample)
    w = sample["ssh_seq"].shape[0]
    h, wi = sample["ssh_seq"].shape[1], sample["ssh_seq"].shape[2]

    ssh_seq = _norm(sample["ssh_seq"], stats.ssh_mean, stats.ssh_std)
    u_seq = _norm(sample["u_seq"], stats.u_mean, stats.u_std)
    v_seq = _norm(sample["v_seq"], stats.v_mean, stats.v_std)

    y = sample["y_obs"].reshape(w, 3, h, wi).copy()
    y[:, 0] = _norm(y[:, 0], stats.ssh_mean, stats.ssh_std)
    out["y_obs"] = y.reshape(w * 3, h, wi).astype(np.float32)
    out["ssh"] = _norm(sample["ssh"], stats.ssh_mean, stats.ssh_std)
    out["u"] = _norm(sample["u"], stats.u_mean, stats.u_std)
    out["v"] = _norm(sample["v"], stats.v_mean, stats.v_std)
    out["ssh_seq"] = ssh_seq
    out["u_seq"] = u_seq
    out["v_seq"] = v_seq
    out["z_sst"] = _norm(sample["z_sst"], stats.sst_mean, stats.sst_std)
    if "x_true" in sample:
        xt = np.concatenate(
            [ssh_seq[:, None], u_seq[:, None], v_seq[:, None]], axis=1
        ).transpose(0, 1, 2, 3).reshape(w * 3, h, wi)
        out["x_true"] = xt.astype(np.float32)
    return out
