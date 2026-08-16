"""Contrastive pretraining for SleepFM."""

import argparse

import torch
from torch.utils.data import DataLoader

from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal
from sleepfm.data.night_dataset import NightSequenceDataset, collate_night
from sleepfm.models.sleepfm import MultiModalSleepFM
from sleepfm.models.temporal import NightTemporalEncoder
from sleepfm.training.trainer import PretrainTrainer
from sleepfm.utils.config import load_config
from sleepfm.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="SleepFM contrastive pretraining")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--mode", type=str, choices=["pairwise", "leave_one_out"], default=None)
    parser.add_argument("--unify", action="store_true", help="Enable SleepFM-Unify heads + mixed loss")
    parser.add_argument("--night", action="store_true", help="Enable night-level temporal windows")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    demo = cfg["demo"] if args.demo else {}
    data_dir = args.data_dir or cfg["data_dir"]
    output_dir = args.output_dir or cfg["output_dir"]
    batch_size = demo.get("batch_size", cfg["batch_size"])
    epochs = demo.get("epochs", cfg["epochs"])
    mode = args.mode or cfg["contrastive_mode"]

    unify_cfg = dict(cfg.get("unify") or {})
    if args.unify:
        unify_cfg["enabled"] = True
    temporal_cfg = dict(cfg.get("temporal") or {})
    if args.night:
        temporal_cfg["enabled"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | mode: {mode} | unify: {bool(unify_cfg.get('enabled'))}")

    if temporal_cfg.get("enabled"):
        window = int(temporal_cfg.get("window", 8))
        stride = temporal_cfg.get("stride")
        train_ds = NightSequenceDataset(data_dir, split="pretrain", window=window, stride=stride)
        val_ds = NightSequenceDataset(data_dir, split="valid", window=window, stride=stride)
        collate_fn = collate_night
    else:
        train_ds = SleepEpochDataset(data_dir, split="pretrain")
        val_ds = SleepEpochDataset(data_dir, split="valid")
        collate_fn = collate_multimodal

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    unify = bool(unify_cfg.get("enabled", False))
    model = MultiModalSleepFM(
        channels=cfg["channels"],
        embedding_dim=cfg["embedding_dim"],
        temperature_init=cfg["temperature_init"],
        unify=unify,
        shared_dim=unify_cfg.get("shared_dim"),
        private_dim=unify_cfg.get("private_dim"),
        downstream_space=unify_cfg.get("downstream_space", "concat"),
    )
    temporal_encoder = None
    if temporal_cfg.get("enabled"):
        d_model = int(unify_cfg.get("shared_dim", 256) if unify else cfg["embedding_dim"])
        temporal_encoder = NightTemporalEncoder(
            d_model=d_model,
            n_layers=int(temporal_cfg.get("n_layers", 2)),
            n_heads=int(temporal_cfg.get("n_heads", 4)),
            kind=str(temporal_cfg.get("model", "gru")),
            window=window,
        )

    loss_weights = dict(unify_cfg.get("loss_weights") or {}) if unify else None
    if temporal_encoder is not None:
        # Temporal head must participate in the loss; unify.yaml defaults temporal:0.
        loss_weights = dict(loss_weights or {})
        if not unify:
            if mode == "leave_one_out":
                loss_weights.setdefault("loo", 1.0)
                loss_weights.setdefault("pairwise", 0.0)
            else:
                loss_weights.setdefault("pairwise", 1.0)
                loss_weights.setdefault("loo", 0.0)
        if float(loss_weights.get("temporal", 0.0)) <= 0.0:
            loss_weights["temporal"] = float(temporal_cfg.get("loss_weight", 0.2))

    trainer = PretrainTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        lr_step_period=cfg["lr_step_period"],
        lr_gamma=cfg["lr_gamma"],
        output_dir=output_dir,
        contrastive_mode=mode,
        early_stopping_patience=cfg.get("early_stopping_patience"),
        loss_weights=loss_weights,
        modality_dropout=float(unify_cfg.get("modality_dropout", 0.0) if unify else 0.0),
        temporal_encoder=temporal_encoder,
        temporal_mask_prob=float(temporal_cfg.get("mask_prob", 0.15)),
        use_mixed_loss=unify or temporal_encoder is not None,
    )
    trainer.fit(epochs=epochs)
    print(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
