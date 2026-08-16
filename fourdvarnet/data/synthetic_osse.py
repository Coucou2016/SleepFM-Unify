"""
Synthetic OSSE benchmark (Gulf-Stream-like QG + SQG SST-SSH synergy).

Faithful to Fablet et al. (2024) setup at reduced resolution when NATL60 is unavailable:
- 10°×10° domain, ~1/20° grid (configurable)
- Daily fields: SSH, SST, u, v
- Gappy along-track altimetry + DUACS-like smooth SSH
- Train/val/test splits aligned with paper calendar windows (scaled for short demos)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fourdvarnet.ops.physics import geostrophic_uv_from_ssh


@dataclass
class SyntheticOSSEConfig:
    n_lat: int = 48
    n_lon: int = 48
    n_days: int = 120
    dt_days: float = 1.0
    f_coriolis: float = 7.0e-5
    g: float = 9.81
    dx_deg: float = 0.05
    dy_deg: float = 0.05
    n_tracks: int = 4
    track_width: int = 1
    sst_noise: float = 0.05
    ssh_noise: float = 0.02
    ageostrophic_frac: float = 0.35
    seed: int = 42


def _make_grid(cfg: SyntheticOSSEConfig) -> tuple[np.ndarray, np.ndarray]:
    lon = np.linspace(-65.0, -55.0, cfg.n_lon, dtype=np.float32)
    lat = np.linspace(33.0, 43.0, cfg.n_lat, dtype=np.float32)
    return np.meshgrid(lon, lat)


def _qg_streamfunction(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    t: int,
    rng: np.random.Generator,
) -> np.ndarray:
    phase = 0.15 * t
    psi = (
        0.8 * np.sin(2 * np.pi * (lon2d + 60) / 8 + phase)
        * np.cos(2 * np.pi * (lat2d - 38) / 6)
        + 0.4 * np.sin(4 * np.pi * lon2d / 10 - 0.3 * t)
        + 0.3 * rng.standard_normal(lon2d.shape).astype(np.float32)
    )
    return psi.astype(np.float32)


def _sqg_sst(ssh: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
    """SST anomaly linked to SSH gradients (SQG-inspired linear synergy)."""
    dssh_dy = np.gradient(ssh, axis=0) / (lat2d[1, 0] - lat2d[0, 0] + 1e-6)
    dssh_dx = np.gradient(ssh, axis=1) / (lon2d[0, 1] - lon2d[0, 0] + 1e-6)
    return (ssh + 0.6 * dssh_dx - 0.4 * dssh_dy).astype(np.float32)


def _track_mask(
    shape: tuple[int, int],
    n_tracks: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=np.float32)
    for _ in range(n_tracks):
        if rng.random() < 0.5:
            col = rng.integers(0, w)
            mask[:, max(0, col - width) : min(w, col + width + 1)] = 1.0
        else:
            row = rng.integers(0, h)
            mask[max(0, row - width) : min(h, row + width + 1), :] = 1.0
    return mask


def generate_osse_sequence(cfg: SyntheticOSSEConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    lon2d, lat2d = _make_grid(cfg)
    n_t, h, w = cfg.n_days, cfg.n_lat, cfg.n_lon

    ssh = np.zeros((n_t, h, w), dtype=np.float32)
    sst = np.zeros((n_t, h, w), dtype=np.float32)
    u = np.zeros((n_t, h, w), dtype=np.float32)
    v = np.zeros((n_t, h, w), dtype=np.float32)
    ssh_obs = np.zeros((n_t, h, w), dtype=np.float32)
    obs_mask = np.zeros((n_t, h, w), dtype=np.float32)

    for t in range(n_t):
        psi = _qg_streamfunction(lon2d, lat2d, t, rng)
        # SSH anomaly ~O(0.05–0.3 m); avoids geostrophic blow-up from g/(f) scaling at demo resolution
        ssh_t = (psi * 0.08).astype(np.float32)
        ssh_t += cfg.ssh_noise * rng.standard_normal(ssh_t.shape).astype(np.float32)
        ssh[t] = ssh_t

        sst[t] = _sqg_sst(ssh_t, lon2d, lat2d) + cfg.sst_noise * rng.standard_normal(ssh_t.shape).astype(
            np.float32
        )

        ug, vg = _geostrophic_np(ssh_t, cfg)
        ua = cfg.ageostrophic_frac * rng.standard_normal(ssh_t.shape).astype(np.float32)
        va = cfg.ageostrophic_frac * rng.standard_normal(ssh_t.shape).astype(np.float32)
        u[t] = ug + ua
        v[t] = vg + va

        mask_t = _track_mask((h, w), cfg.n_tracks, cfg.track_width, rng)
        obs_mask[t] = mask_t
        ssh_obs[t] = np.where(mask_t > 0, ssh_t, 0.0)

    ssh_duacs = _duacs_smooth(ssh_obs, obs_mask)
    return {
        "ssh": ssh,
        "sst": sst,
        "u": u,
        "v": v,
        "ssh_obs": ssh_obs,
        "ssh_duacs": ssh_duacs,
        "obs_mask": obs_mask,
        "lon": lon2d.astype(np.float32),
        "lat": lat2d.astype(np.float32),
    }


def _geostrophic_np(ssh: np.ndarray, cfg: SyntheticOSSEConfig) -> tuple[np.ndarray, np.ndarray]:
    """Geostrophic currents on the OSSE grid (dx, dy in metres from degree spacing)."""
    import torch

    dx_m = cfg.dx_deg * 111e3 * np.cos(np.deg2rad(38.0))
    dy_m = cfg.dy_deg * 111e3
    t = torch.from_numpy(ssh[None, None].astype(np.float32))
    u_g, v_g = geostrophic_uv_from_ssh(
        t,
        f_coriolis=cfg.f_coriolis,
        g=cfg.g,
        dx=dx_m,
        dy=dy_m,
    )
    return u_g.squeeze().numpy(), v_g.squeeze().numpy()


def _duacs_smooth(ssh_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter, uniform_filter

        filled = ssh_obs.copy()
        for t in range(ssh_obs.shape[0]):
            m = mask[t] > 0
            if m.any():
                mean_val = filled[t][m].mean()
                filled[t][~m] = mean_val
            filled[t] = gaussian_filter(filled[t], sigma=2.0)
        return filled.astype(np.float32)
    except ImportError:
        from scipy.ndimage import uniform_filter

        return uniform_filter(ssh_obs, size=5).astype(np.float32)


def _split_indices(n_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper windows (Feb–Sep train, Jan–Feb val, Oct–Dec test) on synthetic timeline."""
    train = np.arange(int(0.15 * n_days), n_days)
    val = np.arange(int(0.08 * n_days), int(0.15 * n_days))
    test = np.arange(0, int(0.08 * n_days))
    return train, val, test


def generate_osse_split(
    cfg: SyntheticOSSEConfig,
    split: str,
    window: int = 7,
) -> list[dict[str, np.ndarray]]:
    data = generate_osse_sequence(cfg)
    train_idx, val_idx, test_idx = _split_indices(cfg.n_days)
    idx = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    samples = []
    for start in idx:
        if start + window > cfg.n_days:
            continue
        sl = slice(start, start + window)
        samples.append(_pack_window(data, sl, window))
    return samples


def _pack_window(data: dict[str, np.ndarray], sl: slice, window: int) -> dict[str, np.ndarray]:
    ssh_seq = data["ssh"][sl]
    u_seq = data["u"][sl]
    v_seq = data["v"][sl]
    ssh_duacs = data["ssh_duacs"][sl]
    ssh_obs = data["ssh_obs"][sl]
    mask = data["obs_mask"][sl]
    sst = data["sst"][sl]

    y_ssh = ssh_duacs.copy()
    y_ssh = np.where(mask > 0, ssh_obs, y_ssh)
    obs_mask = np.zeros((window, 3, *ssh_seq.shape[1:]), dtype=np.float32)
    obs_mask[:, 0] = np.maximum(mask, 0.25)
    obs_mask[:, 1] = 0.0
    obs_mask[:, 2] = 0.0

    y_flat = np.concatenate(
        [y_ssh[:, None], np.zeros_like(u_seq[:, None]), np.zeros_like(v_seq[:, None])], axis=1
    )
    y_flat = y_flat.transpose(0, 1, 2, 3).reshape(window * 3, *ssh_seq.shape[1:])

    obs_mask_flat = obs_mask.transpose(0, 1, 2, 3).reshape(window * 3, *ssh_seq.shape[1:])

    x_true_flat = np.concatenate(
        [ssh_seq[:, None], u_seq[:, None], v_seq[:, None]], axis=1
    ).transpose(0, 1, 2, 3).reshape(window * 3, *ssh_seq.shape[1:])

    return {
        "y_obs": y_flat.astype(np.float32),
        "obs_mask": obs_mask_flat.astype(np.float32),
        "z_sst": sst[-1:].astype(np.float32),
        "ssh": ssh_seq[-1:].astype(np.float32),
        "u": u_seq[-1:].astype(np.float32),
        "v": v_seq[-1:].astype(np.float32),
        "ssh_seq": ssh_seq.astype(np.float32),
        "u_seq": u_seq.astype(np.float32),
        "v_seq": v_seq.astype(np.float32),
        "x_true": x_true_flat.astype(np.float32),
    }


def save_osse_npz(cfg: SyntheticOSSEConfig, out_dir: str | Path, window: int = 7) -> Path:
    from fourdvarnet.data.normalize import NormStats, apply_norm_sample

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw: dict[str, list[dict[str, np.ndarray]]] = {}
    for split in ("train", "val", "test"):
        raw[split] = generate_osse_split(cfg, split, window=window)

    stats = NormStats.from_train_samples(raw["train"])
    for split in ("train", "val", "test"):
        normed = [apply_norm_sample(s, stats) for s in raw[split]]
        arrays = {k: np.stack([s[k] for s in normed], axis=0) for k in normed[0]}
        np.savez_compressed(out_dir / f"{split}.npz", **arrays)

    meta = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    meta.update(stats.to_dict())
    meta["window"] = window
    meta["lat_ref"] = 38.0
    np.savez(out_dir / "meta.npz", **{k: np.array(v) for k, v in meta.items()})
    return out_dir
