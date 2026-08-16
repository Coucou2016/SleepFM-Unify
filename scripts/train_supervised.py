"""Supervised baselines on concatenated modalities (paper comparison).

Models
------
* ``effnet`` — EffNetSupervised (all modalities concat), epoch-level, paper CNN baseline.
* ``seq`` / ``SeqStagingBaseline`` — lightweight 1D CNN + GRU/Transformer temporal head
  for fairer **night-level** staging comparison. **Not** a U-Sleep port.

Protocol (paper-aligned): train on ``train`` split only; report macro AUROC / AUPRC
on ``test`` (same metrics as the frozen-embedding linear probe).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal
from sleepfm.data.night_dataset import NightSequenceDataset, collate_night
from sleepfm.models.encoders import EffNetSupervised, SeqStagingBaseline
from sleepfm.utils.config import load_config
from sleepfm.utils.seed import set_seed


def _stack_modalities(signals: dict) -> torch.Tensor:
    return torch.cat([signals["bas"], signals["ecg"], signals["respiratory"]], dim=1)


def _collate_supervised(batch):
    signals = collate_multimodal(batch)
    labels = torch.tensor([b["labels"]["stage_id"] for b in batch], dtype=torch.long)
    return _stack_modalities(signals), labels


def _collate_seq(batch):
    night = collate_night(batch)
    stacked = torch.cat(
        [night["bas"], night["ecg"], night["respiratory"]],
        dim=2,
    )  # (B, L, C, T)
    if "stage_id" in night:
        y = night["stage_id"]
    else:
        y = torch.zeros(stacked.size(0), stacked.size(1), dtype=torch.long)
    pad = night.get("padding_mask")
    return stacked, y, pad


def _multiclass_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    n_classes = proba.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.ndim == 1 or y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin.reshape(-1, 1), y_bin.reshape(-1, 1)])
    try:
        macro_auroc = float(roc_auc_score(y_bin, proba, average="macro", multi_class="ovr"))
    except ValueError:
        macro_auroc = float("nan")
    try:
        macro_auprc = float(average_precision_score(y_bin, proba, average="macro"))
    except ValueError:
        macro_auprc = float("nan")
    return {"macro_auroc": macro_auroc, "macro_auprc": macro_auprc}


@torch.no_grad()
def _eval_effnet(model, loader, device) -> dict:
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        logits_all.append(logits.cpu())
        y_all.append(y)
    logits = torch.cat(logits_all, dim=0)
    y = torch.cat(y_all, dim=0).numpy()
    proba = torch.softmax(logits, dim=-1).numpy()
    return _multiclass_metrics(y, proba)


@torch.no_grad()
def _eval_seq(model, loader, device) -> dict:
    model.eval()
    logits_all, y_all = [], []
    for x, y, pad in loader:
        x = x.to(device)
        pad_t = pad.to(device) if pad is not None else None
        logits = model(x, padding_mask=pad_t)  # (B, L, C)
        b, length, c = logits.shape
        flat_logits = logits.reshape(b * length, c)
        flat_y = y.reshape(b * length)
        if pad is not None:
            keep = ~pad.reshape(-1).bool()
            flat_logits = flat_logits[keep]
            flat_y = flat_y[keep]
        logits_all.append(flat_logits.cpu())
        y_all.append(flat_y)
    if not logits_all:
        return {"macro_auroc": float("nan"), "macro_auprc": float("nan")}
    logits = torch.cat(logits_all, dim=0)
    y = torch.cat(y_all, dim=0).numpy()
    proba = torch.softmax(logits, dim=-1).numpy()
    return _multiclass_metrics(y, proba)


def main():
    parser = argparse.ArgumentParser(
        description="Supervised sleep-staging baselines (EffNetSupervised / SeqStagingBaseline)"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="effnet",
        choices=["effnet", "seq", "SeqStagingBaseline"],
        help="effnet = paper-style concat EffNet; seq = SeqStagingBaseline (not U-Sleep)",
    )
    parser.add_argument("--window", type=int, default=16, help="Night window for SeqStagingBaseline")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    demo = cfg["demo"] if args.demo else {}
    data_dir = args.data_dir or cfg["data_dir"]
    batch_size = args.batch_size or demo.get("batch_size", cfg["batch_size"])
    epochs = args.epochs or demo.get("epochs", 5)
    channels = cfg["channels"]
    in_ch = sum(channels[m] for m in ("bas", "ecg", "respiratory"))
    out_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/pretrain"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "SeqStagingBaseline" if args.model in ("seq", "SeqStagingBaseline") else "effnet"
    lr = args.lr if args.lr is not None else (1e-3 if model_name == "effnet" else 1e-3)

    if model_name == "effnet":
        train_ds = SleepEpochDataset(data_dir, split="train", return_labels=True)
        test_ds = SleepEpochDataset(data_dir, split="test", return_labels=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_supervised
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_supervised
        )
        model = EffNetSupervised(in_channel=in_ch, num_classes=5).to(device)
        # Paper-ish: SGD; Adam remains available via --lr with default Adam for stability on tiny demos
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            total, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
                total += loss.item() * y.size(0)
                n += y.size(0)
            print(f"Epoch {epoch + 1}/{epochs} loss={total / max(n, 1):.4f}")

        metrics = _eval_effnet(model, test_loader, device)
        ckpt_name = "supervised_effnet.pt"
    else:
        # Sequence baseline on night windows from the train split
        window = int(args.window)
        train_ds = NightSequenceDataset(
            data_dir, split="train", window=window, stride=max(1, window // 2), return_labels=True
        )
        test_ds = NightSequenceDataset(
            data_dir, split="test", window=window, stride=max(1, window // 2), return_labels=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=max(1, batch_size // 2), shuffle=True, collate_fn=_collate_seq
        )
        test_loader = DataLoader(
            test_ds, batch_size=max(1, batch_size // 2), shuffle=False, collate_fn=_collate_seq
        )
        model = SeqStagingBaseline(in_channel=in_ch, num_classes=5, temporal="gru").to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            total, n = 0.0, 0
            for x, y, pad in train_loader:
                x, y = x.to(device), y.to(device)
                pad_t = pad.to(device) if pad is not None else None
                opt.zero_grad()
                logits = model(x, padding_mask=pad_t)
                b, length, c = logits.shape
                flat_logits = logits.reshape(b * length, c)
                flat_y = y.reshape(b * length)
                if pad_t is not None:
                    keep = ~pad_t.reshape(-1).bool()
                    flat_logits = flat_logits[keep]
                    flat_y = flat_y[keep]
                if flat_y.numel() == 0:
                    continue
                loss = criterion(flat_logits, flat_y)
                loss.backward()
                opt.step()
                total += loss.item() * flat_y.size(0)
                n += flat_y.size(0)
            print(f"Epoch {epoch + 1}/{epochs} loss={total / max(n, 1):.4f}")

        metrics = _eval_seq(model, test_loader, device)
        ckpt_name = "seq_staging_baseline.pt"

    path = out_dir / ckpt_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channel": in_ch,
            "model": model_name,
            "channels": channels,
            "metrics_test": metrics,
        },
        path,
    )
    metrics_path = out_dir / f"{Path(ckpt_name).stem}_metrics.json"
    metrics_path.write_text(json.dumps({"model": model_name, "test": metrics}, indent=2), encoding="utf-8")
    print(f"Saved {path}")
    print(f"Test staging metrics ({model_name}):", json.dumps(metrics))
    print(
        "Note: SeqStagingBaseline is an in-repo sequence CNN+temporal head, "
        "not U-Sleep. Optional u-sleep install is unsupported here."
    )


if __name__ == "__main__":
    main()
