"""Evaluate 4DVarNet reconstructions (RMSE, vector correlation, spectral scales)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from sleepfm.utils.config import load_config
from fourdvarnet.data.dataset import OSSEDataset
from fourdvarnet.eval.metrics import evaluate_batch, spectral_metrics_summary
from fourdvarnet.models.fourdvarnet import FourDVarNet, FourDVarNetConfig
from fourdvarnet.ops.physics import geostrophic_uv_from_ssh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fourdvarnet.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cpu"))
    window = int(cfg.get("window", 7))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FourDVarNet(FourDVarNetConfig(**ckpt.get("model_cfg", {})), n_timesteps=window).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(OSSEDataset(Path(cfg["data_dir"]) / "test.npz"), batch_size=4)
    all_stats = []
    u_list, v_list, ut_list, vt_list = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.enable_grad():
            out = model(batch["y_obs"], batch["obs_mask"], batch["z_sst"])
            all_stats.append(evaluate_batch(out, batch))
            u_list.append(out["u_last"].detach().cpu().numpy())
            v_list.append(out["v_last"].detach().cpu().numpy())
            ut_list.append(batch["u"].detach().cpu().numpy())
            vt_list.append(batch["v"].detach().cpu().numpy())

    mean_stats = {k: float(np.mean([s[k] for s in all_stats])) for k in all_stats[0]}
    u_p = np.concatenate(u_list, axis=0)[:, 0].mean(axis=0)
    v_p = np.concatenate(v_list, axis=0)[:, 0].mean(axis=0)
    u_t = np.concatenate(ut_list, axis=0)[:, 0].mean(axis=0)
    v_t = np.concatenate(vt_list, axis=0)[:, 0].mean(axis=0)
    spec = spectral_metrics_summary(u_p, v_p, u_t, v_t)

    print("=== 4DVarNet test metrics ===")
    for k, v in {**mean_stats, **spec}.items():
        print(f"  {k}: {v:.4f}")

    ssh = batch["ssh"]  # noqa: last batch — baseline geostrophic from SSH
    ug, vg = geostrophic_uv_from_ssh(
        ssh,
        f_coriolis=7e-5,
        dx=1.0,
        dy=1.0,
    )
    base = evaluate_batch({"u_last": ug, "v_last": vg, "ssh_last": ssh}, batch)
    print("=== SSH-geostrophic baseline (last batch) ===")
    for k, v in base.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
