"""Evaluation metrics (RMSE, vector correlation, explained variance — paper Sec. 4.3 spirit)."""

from __future__ import annotations

import numpy as np
import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(((pred - target) ** 2).mean()).item())


def vector_correlation(
    u_p: torch.Tensor, v_p: torch.Tensor, u_t: torch.Tensor, v_t: torch.Tensor
) -> float:
    vp = torch.stack([u_p.flatten(), v_p.flatten()], dim=1)
    vt = torch.stack([u_t.flatten(), v_t.flatten()], dim=1)
    num = (vp * vt).sum()
    den = vp.norm() * vt.norm() + 1e-8
    return float((num / den).item())


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    var_t = target.var()
    if var_t < 1e-12:
        return 0.0
    mse = ((pred - target) ** 2).mean()
    return float((1.0 - mse / var_t).clamp(min=-1.0, max=1.0).item())


def evaluate_batch(outputs: dict, targets: dict) -> dict[str, float]:
    u_p, v_p = outputs["u_last"], outputs["v_last"]
    u_t, v_t = targets["u"], targets["v"]
    ssh_p, ssh_t = outputs["ssh_last"], targets["ssh"]
    return {
        "rmse_u": rmse(u_p, u_t),
        "rmse_v": rmse(v_p, v_t),
        "rmse_ssh": rmse(ssh_p, ssh_t),
        "vec_corr": vector_correlation(u_p, v_p, u_t, v_t),
        "ev_u": explained_variance(u_p, u_t),
        "ev_v": explained_variance(v_p, v_t),
        "ev_ssh": explained_variance(ssh_p, ssh_t),
    }


def _radial_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = field.shape
    f2 = np.fft.fft2(field - field.mean())
    psd = np.abs(np.fft.fftshift(f2)) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    radial = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)
    for ri in range(max_r + 1):
        m = r == ri
        if m.any():
            radial[ri] = psd[m].mean()
            counts[ri] = m.sum()
    k = np.arange(max_r + 1) / max(max_r, 1)
    return k, radial


def resolved_scale_days(
    pred: np.ndarray, truth: np.ndarray, dt_days: float = 1.0, threshold: float = 0.5
) -> float:
    """Heuristic spectral skill: first scale where co-spectrum ratio drops below threshold."""
    err = pred - truth
    _, ps_err = _radial_spectrum(err)
    _, ps_truth = _radial_spectrum(truth)
    ratio = 1.0 - ps_err / (ps_truth + 1e-8)
    ok = np.where(ratio >= threshold)[0]
    if len(ok) == 0:
        return float("nan")
    k_min = ok[0] / max(len(ok), 1)
    return float(dt_days / (k_min + 1e-3))


def spectral_metrics_summary(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_true: np.ndarray,
    v_true: np.ndarray,
    dt_days: float = 1.0,
) -> dict[str, float]:
    speed_p = np.sqrt(u_pred ** 2 + v_pred ** 2)
    speed_t = np.sqrt(u_true ** 2 + v_true ** 2)
    return {
        "lambda_t_u": resolved_scale_days(u_pred, u_true, dt_days),
        "lambda_t_v": resolved_scale_days(v_pred, v_true, dt_days),
        "lambda_t_speed": resolved_scale_days(speed_p, speed_t, dt_days),
    }
