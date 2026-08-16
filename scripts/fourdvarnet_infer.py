"""Run 4DVarNet inference on test OSSE split."""

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
from fourdvarnet.models.fourdvarnet import FourDVarNet, FourDVarNetConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fourdvarnet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/fourdvarnet/predictions.npz")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cpu"))
    window = int(cfg.get("window", 7))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = FourDVarNetConfig(**ckpt.get("model_cfg", {}))
    model = FourDVarNet(model_cfg, n_timesteps=window).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(
        OSSEDataset(Path(cfg["data_dir"]) / "test.npz"),
        batch_size=1,
        shuffle=False,
    )
    preds_u, preds_v, preds_ssh, true_u, true_v = [], [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.enable_grad():
            out = model(batch["y_obs"], batch["obs_mask"], batch["z_sst"])
            preds_u.append(out["u_last"].detach().cpu().numpy())
            preds_v.append(out["v_last"].detach().cpu().numpy())
            preds_ssh.append(out["ssh_last"].detach().cpu().numpy())
            true_u.append(batch["u"].detach().cpu().numpy())
            true_v.append(batch["v"].detach().cpu().numpy())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        u=np.concatenate(preds_u, axis=0),
        v=np.concatenate(preds_v, axis=0),
        ssh=np.concatenate(preds_ssh, axis=0),
        u_true=np.concatenate(true_u, axis=0),
        v_true=np.concatenate(true_v, axis=0),
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
