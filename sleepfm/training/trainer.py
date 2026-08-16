"""Contrastive pretraining loop (SleepFM baseline + SleepFM-Unify mixed loss)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sleepfm.models.sleepfm import ContrastiveMode, MultiModalSleepFM
from sleepfm.models.temporal import NightTemporalEncoder


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


class PretrainTrainer:
    def __init__(
        self,
        model: MultiModalSleepFM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        lr_step_period: int = 5,
        lr_gamma: float = 0.1,
        output_dir: str | Path = "outputs/pretrain",
        contrastive_mode: ContrastiveMode = "leave_one_out",
        early_stopping_patience: Optional[int] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        modality_dropout: float = 0.0,
        temporal_encoder: Optional[NightTemporalEncoder] = None,
        temporal_mask_prob: float = 0.15,
        use_mixed_loss: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.contrastive_mode = contrastive_mode
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = math.inf
        self.epochs_without_improvement = 0
        self.loss_weights = loss_weights
        self.modality_dropout = float(modality_dropout)
        self.temporal_encoder = temporal_encoder.to(device) if temporal_encoder is not None else None
        self.temporal_mask_prob = temporal_mask_prob
        self.use_mixed_loss = bool(
            use_mixed_loss
            or model.unify
            or (loss_weights is not None)
            or (self.modality_dropout > 0)
            or (self.temporal_encoder is not None)
        )

        params = list(self.model.parameters())
        if self.temporal_encoder is not None:
            params += list(self.temporal_encoder.parameters())
        self.optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_step_period, gamma=lr_gamma
        )

    def _compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        if self.use_mixed_loss:
            return self.model.pretrain_loss(
                batch,
                mode=self.contrastive_mode,
                loss_weights=self.loss_weights,
                modality_dropout=self.modality_dropout,
                temporal_encoder=self.temporal_encoder,
                temporal_mask_prob=self.temporal_mask_prob,
            )
        return self.model.contrastive_loss(batch, mode=self.contrastive_mode)

    def _step_epoch(self, loader: DataLoader, train: bool) -> float:
        if len(loader) == 0:
            raise RuntimeError(
                f"{'train' if train else 'val'} DataLoader is empty "
                "(dataset smaller than batch_size with drop_last, or no samples). "
                "Reduce --batch-size / config batch_size or grow the split."
            )
        if train:
            self.model.train()
            if self.temporal_encoder is not None:
                self.temporal_encoder.train()
        else:
            self.model.eval()
            if self.temporal_encoder is not None:
                self.temporal_encoder.eval()

        total_loss = 0.0
        n = 0
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
                batch = _move_batch(batch, self.device)
                loss, _logs = self._compute_loss(batch)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    with torch.no_grad():
                        self.model.temperature.clamp_(min=0.0)
                bs = int(batch["bas"].size(0))
                total_loss += loss.item() * bs
                n += bs

        return total_loss / max(n, 1)

    def fit(self, epochs: int) -> Dict[str, list]:
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(epochs):
            train_loss = self._step_epoch(self.train_loader, train=True)
            history["train_loss"].append(train_loss)
            val_loss = train_loss
            if self.val_loader is not None:
                val_loss = self._step_epoch(self.val_loader, train=False)
            history["val_loss"].append(val_loss)
            self.scheduler.step()

            print(
                f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint("best.pt", epoch, val_loss)
            else:
                self.epochs_without_improvement += 1

            self._save_checkpoint("checkpoint.pt", epoch, val_loss)

            if (
                self.early_stopping_patience is not None
                and self.epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping: no val improvement for "
                    f"{self.early_stopping_patience} epoch(s)"
                )
                break

        return history

    def _save_checkpoint(self, name: str, epoch: int, loss: float) -> None:
        ckpt = {
            "epoch": epoch,
            "loss": loss,
            "temperature": self.model.temperature.item(),
            "channels": dict(self.model.channels),
            "embedding_dim": self.model.embedding_dim,
            "contrastive_mode": self.contrastive_mode,
            "unify": self.model.unify,
            "shared_dim": self.model.shared_dim,
            "private_dim": self.model.private_dim,
            "downstream_space": self.model.downstream_space,
            "loss_weights": self.loss_weights,
            "modality_dropout": self.modality_dropout,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.temporal_encoder is not None:
            ckpt["temporal_state_dict"] = self.temporal_encoder.state_dict()
            ckpt["temporal_cfg"] = {
                "d_model": self.temporal_encoder.d_model,
                "kind": self.temporal_encoder.kind,
                "n_layers": self.temporal_encoder.n_layers,
                "n_heads": self.temporal_encoder.n_heads,
                "max_len": self.temporal_encoder.max_len,
                "window": getattr(self.temporal_encoder, "window", None),
            }
        torch.save(ckpt, self.output_dir / name)
