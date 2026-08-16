from __future__ import annotations



from pathlib import Path

from typing import Any, Dict



import numpy as np

import torch

from loguru import logger

from torch.utils.data import DataLoader

from tqdm import tqdm



from fourdvarnet.data.dataset import OSSEDataset

from fourdvarnet.models.fourdvarnet import FourDVarNet, FourDVarNetConfig





def _load_meta(data_dir: Path) -> dict[str, float]:

    meta_path = data_dir / "meta.npz"

    if not meta_path.exists():

        return {}

    meta = np.load(meta_path)

    return {k: float(meta[k]) if meta[k].ndim == 0 else meta[k] for k in meta.files}





def train_epoch(

    model: FourDVarNet,

    loader: DataLoader,

    optimizer: torch.optim.Optimizer,

    device: torch.device,

    loss_weights: dict[str, float] | None = None,

    grad_clip: float | None = None,

) -> Dict[str, float]:

    model.train()

    totals: Dict[str, float] = {}

    n = 0

    for batch in tqdm(loader, desc="train", leave=False):

        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(batch["y_obs"], batch["obs_mask"], batch["z_sst"])

        loss, stats = model.training_loss(out, batch, weights=loss_weights)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        if grad_clip is not None and grad_clip > 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        n += 1

        for k, v in stats.items():

            totals[k] = totals.get(k, 0.0) + v

    return {k: v / max(n, 1) for k, v in totals.items()}





def eval_epoch(

    model: FourDVarNet,

    loader: DataLoader,

    device: torch.device,

) -> Dict[str, float]:

    from fourdvarnet.eval.metrics import evaluate_batch



    model.eval()

    totals: Dict[str, float] = {}

    n = 0

    for batch in loader:

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.enable_grad():

            out = model(batch["y_obs"], batch["obs_mask"], batch["z_sst"])

        stats = evaluate_batch(out, batch)

        n += 1

        for k, v in stats.items():

            totals[k] = totals.get(k, 0.0) + v

    return {k: v / max(n, 1) for k, v in totals.items()}





def run_training(cfg: Dict[str, Any]) -> Path:

    device = torch.device(cfg.get("device", "cpu"))

    data_dir = Path(cfg["data_dir"])

    window = int(cfg.get("window", 7))

    meta = _load_meta(data_dir)



    train_ds = OSSEDataset(data_dir / "train.npz")

    val_ds = OSSEDataset(data_dir / "val.npz")

    pin = device.type == "cuda"

    train_loader = DataLoader(

        train_ds,

        batch_size=int(cfg.get("batch_size", 4)),

        shuffle=True,

        num_workers=int(cfg.get("num_workers", 0)),

        pin_memory=pin,

    )

    val_loader = DataLoader(

        val_ds,

        batch_size=int(cfg.get("batch_size", 4)),

        shuffle=False,

        pin_memory=pin,

    )



    dx_deg = float(meta.get("dx_deg", cfg.get("dx_deg", 0.05)))
    dy_deg = float(meta.get("dy_deg", cfg.get("dy_deg", 0.05)))
    lat_ref = float(meta.get("lat_ref", 38.0))
    dx_m = dx_deg * 111e3 * float(np.cos(np.deg2rad(lat_ref)))
    dy_m = dy_deg * 111e3

    model_cfg = FourDVarNetConfig(
        n_iter=int(cfg.get("n_iter", 8)),
        n_features=int(cfg.get("n_features", 20)),
        lstm_hidden=int(cfg.get("lstm_hidden", 64)),
        lambda_obs=float(cfg.get("lambda_obs", 1.0)),
        lambda_sst=float(cfg.get("lambda_sst", 0.5)),
        lambda_prior=float(cfg.get("lambda_prior", 0.1)),
        dx=float(cfg.get("dx", 1.0)),
        dy=float(cfg.get("dy", 1.0)),
        f_coriolis=float(cfg.get("f_coriolis", meta.get("f_coriolis", 7.0e-5))),
        dx_m=dx_m,
        dy_m=dy_m,
        ssh_mean=float(meta.get("ssh_mean", 0.0)),
        ssh_std=float(meta.get("ssh_std", 1.0)),
        u_mean=float(meta.get("u_mean", 0.0)),
        u_std=float(meta.get("u_std", 1.0)),
        v_mean=float(meta.get("v_mean", 0.0)),
        v_std=float(meta.get("v_std", 1.0)),
        residual_geo=bool(cfg.get("residual_geo", True)),
        alpha_uv_geo=float(cfg.get("alpha_uv_geo", 0.05)),
    )

    model = FourDVarNet(model_cfg, n_timesteps=window).to(device)

    lr = float(cfg.get("lr", 1e-3))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_weights = cfg.get("loss_weights")

    grad_clip = cfg.get("grad_clip_norm")



    out_dir = Path(cfg.get("output_dir", "outputs/fourdvarnet"))

    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    epochs = int(cfg.get("epochs", 20))



    for epoch in range(1, epochs + 1):

        tr = train_epoch(

            model, train_loader, optimizer, device, loss_weights=loss_weights, grad_clip=grad_clip

        )

        va = eval_epoch(model, val_loader, device)

        logger.info(

            f"epoch {epoch} train_loss={tr['loss']:.4f} "

            f"val_rmse_u={va['rmse_u']:.4f} vec_corr={va.get('vec_corr', 0):.3f}"

        )

        if va["rmse_u"] < best_val:

            best_val = va["rmse_u"]

            torch.save(

                {

                    "model": model.state_dict(),

                    "cfg": cfg,

                    "model_cfg": model_cfg.__dict__,

                    "meta": meta,

                },

                out_dir / "best.pt",

            )



    return out_dir / "best.pt"

