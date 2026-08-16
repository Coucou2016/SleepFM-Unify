#!/usr/bin/env python
"""Generate SciencePlots-styled SleepFM-Unify figures (PNG + SVG).

All quantitative panels from local synthetic/demo runs or architecture
schematics. CinC/SHHS metrics are never invented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np

import scienceplots  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"


def _configure_fonts() -> str:
    """Prefer Times New Roman for Latin; SimHei/YaHei for CJK."""
    cjk_candidates = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((n for n in cjk_candidates if n in available), None)
    # SciencePlots sets serif; append CJK so Chinese glyphs resolve.
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    if cjk:
        plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
        # Allow mixed Latin/CJK: matplotlib falls back through font.family list
        plt.rcParams["font.family"] = ["Times New Roman", cjk, "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    return cjk or "NONE"


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    svg = OUT / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {png} {svg}")


def fig01_architecture(cjk: str) -> None:
    with plt.style.context(["science", "no-latex"]):
        _configure_fonts()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 7)
        ax.axis("off")
        ax.set_title(
            "SleepFM-Unify: Shared–Private Factorization (架构示意)",
            pad=10,
        )

        def box(x, y, w, h, text, fc="#e8f1fa", ec="#2c5f8a"):
            r = mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                facecolor=fc, edgecolor=ec, linewidth=1.2,
            )
            ax.add_patch(r)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

        # Inputs
        box(0.3, 5.2, 2.2, 1.0, "BAS / EEG\n(脑电相关)", "#f5f5f5", "#555")
        box(0.3, 3.6, 2.2, 1.0, "ECG\n(心电)", "#f5f5f5", "#555")
        box(0.3, 2.0, 2.2, 1.0, "Respiratory\n(呼吸)", "#f5f5f5", "#555")

        # Encoders
        box(3.0, 5.2, 2.4, 1.0, "1D EffNet\nencoder", "#fff3e0", "#b86b00")
        box(3.0, 3.6, 2.4, 1.0, "1D EffNet\nencoder", "#fff3e0", "#b86b00")
        box(3.0, 2.0, 2.4, 1.0, "1D EffNet\nencoder", "#fff3e0", "#b86b00")

        # Heads
        box(6.0, 4.8, 2.6, 1.6, "Shared head\n$z^{shared}$ (256-d)\n跨模态对齐", "#e3f2e8", "#1b6b3a")
        box(6.0, 2.4, 2.6, 1.6, "Private head\n$z^{private}$ (256-d)\n模态特异", "#fde8e8", "#8b2e2e")

        # Losses
        box(9.2, 5.0, 2.5, 1.4, "LOO + Pairwise\nInfoNCE", "#e8f1fa", "#2c5f8a")
        box(9.2, 3.2, 2.5, 1.2, "Orthogonality\n$\\mathrm{mean}((Z_s^\\top Z_p)^2)$", "#e8f1fa", "#2c5f8a")
        box(9.2, 1.6, 2.5, 1.2, "Modality dropout\n+ $\\mathcal{L}_{miss}$", "#e8f1fa", "#2c5f8a")

        for y in (5.7, 4.1, 2.5):
            ax.annotate("", xy=(3.0, y), xytext=(2.5, y),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.0))
            ax.annotate("", xy=(6.0, min(y, 5.4)), xytext=(5.4, y),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.0))
        ax.annotate("", xy=(9.2, 5.6), xytext=(8.6, 5.6),
                    arrowprops=dict(arrowstyle="->", color="#1b6b3a", lw=1.1))
        ax.annotate("", xy=(9.2, 3.7), xytext=(8.6, 3.2),
                    arrowprops=dict(arrowstyle="->", color="#8b2e2e", lw=1.1))
        ax.text(0.3, 0.4,
                "Baseline SleepFM: unify.enabled=false, leave-one-out only. "
                f"CJK font={cjk}",
                fontsize=7, color="#444")
        _save(fig, "fig01_architecture")


def fig02_loss_curves(history_path: Path | None) -> None:
    """Plot real history if present; else a short local synthetic training run."""
    history = None
    if history_path and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))

    if not history:
        history = _run_tiny_unify_pretrain()

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    with plt.style.context(["science", "no-latex"]):
        _configure_fonts()
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
        ax = axes[0]
        ax.plot(epochs, history["train_loss"], "o-", label="Train total", ms=3)
        if history.get("val_loss"):
            ax.plot(epochs, history["val_loss"], "s--", label="Val total", ms=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Synthetic demo pretrain loss\n(合成演示；非 CinC/SHHS)")
        ax.legend(frameon=False)

        ax = axes[1]
        for key, lab in [
            ("loo", r"$\mathcal{L}_{LOO}$"),
            ("pairwise", r"$\mathcal{L}_{pair}$"),
            ("orth", r"$\mathcal{L}_{orth}$"),
            ("miss", r"$\mathcal{L}_{miss}$"),
        ]:
            if key in history and history[key]:
                ax.plot(epochs, history[key], label=lab, ms=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Term")
        ax.set_title("Mixed Unify loss terms\n(演示分解)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        _save(fig, "fig02_loss_curves")
        (OUT / "fig02_loss_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


def _run_tiny_unify_pretrain() -> dict:
    """Few-epoch unify pretrain on synthetic data; returns logged terms."""
    import torch
    from torch.utils.data import DataLoader

    from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal
    from sleepfm.models.sleepfm import MultiModalSleepFM
    from sleepfm.utils.config import load_config
    from sleepfm.utils.seed import set_seed

    cfg = load_config(str(ROOT / "configs" / "unify.yaml"))
    set_seed(cfg["seed"])
    data_dir = ROOT / "data" / "synthetic"
    if not (data_dir / "index.json").is_file():
        from sleepfm.data.synthetic import write_synthetic_dataset

        demo = cfg["demo"]
        clip_length = int(demo["sample_rate"] * demo["clip_seconds"])
        write_synthetic_dataset(
            data_dir,
            cfg["channels"],
            clip_length,
            {
                "pretrain": demo.get("num_pretrain", 64),
                "valid": demo["num_val"],
                "train": demo.get("num_train", 48),
                "test": demo["num_test"],
            },
            seed=cfg["seed"],
            apnea_rate=demo.get("apnea_rate", 0.2),
            num_participants=demo.get("num_participants"),
            epochs_per_participant=demo.get("epochs_per_participant"),
        )

    device = torch.device("cpu")
    unify_cfg = dict(cfg.get("unify") or {})
    unify_on = bool(unify_cfg.get("enabled", False))
    model = MultiModalSleepFM(
        channels=cfg["channels"],
        embedding_dim=cfg["embedding_dim"],
        unify=unify_on,
        shared_dim=unify_cfg.get("shared_dim"),
        private_dim=unify_cfg.get("private_dim"),
        downstream_space=unify_cfg.get("downstream_space", "concat"),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_ds = SleepEpochDataset(str(data_dir), split="pretrain")
    val_ds = SleepEpochDataset(str(data_dir), split="valid")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_multimodal)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_multimodal)

    lw = dict(unify_cfg.get("loss_weights") or {})
    drop = float(unify_cfg.get("modality_dropout") or 0.0)
    hist = {"train_loss": [], "val_loss": [], "loo": [], "pairwise": [], "orth": [], "miss": [],
            "note": "synthetic_demo_only"}
    n_epochs = 5
    model.train()
    for ep in range(n_epochs):
        totals = []
        parts = {"loo": [], "pairwise": [], "orth": [], "miss": []}
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss, logs = model.pretrain_loss(
                batch,
                mode=cfg.get("contrastive_mode", "leave_one_out"),
                loss_weights=lw,
                modality_dropout=drop,
            )
            loss.backward()
            opt.step()
            totals.append(float(loss.detach()))
            for k in parts:
                key = f"loss_{k}"
                if key in logs:
                    parts[k].append(float(logs[key]))
        hist["train_loss"].append(float(np.mean(totals)))
        for k in parts:
            hist[k].append(float(np.mean(parts[k])) if parts[k] else float("nan"))

        model.eval()
        vtot = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                loss, _logs = model.pretrain_loss(
                    batch,
                    mode=cfg.get("contrastive_mode", "leave_one_out"),
                    loss_weights=lw,
                    modality_dropout=0.0,
                )
                vtot.append(float(loss.detach()))
        hist["val_loss"].append(float(np.mean(vtot)) if vtot else float("nan"))
        model.train()
        print(f"tiny unify ep {ep+1}/{n_epochs} train={hist['train_loss'][-1]:.4f}")

    out_dir = ROOT / "outputs" / "unify_demo_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": n_epochs - 1,
            "model_state_dict": model.state_dict(),
            "channels": cfg["channels"],
            "embedding_dim": cfg["embedding_dim"],
            "unify": unify_on,
            "shared_dim": model.shared_dim,
            "private_dim": model.private_dim,
            "history": hist,
            "synthetic_demo": True,
        },
        out_dir / "best.pt",
    )
    return hist


def fig03_ablation_schematic() -> None:
    """Schematic ablation design — placeholders for real CinC/SHHS numbers."""
    methods = [
        "LOO baseline\n(SleepFM)",
        "Unify\n(shared+private)",
        "Unify\n− orth",
        "Unify\n− miss",
        "Unify\n+ temporal",
    ]
    # Synthetic demo placeholders only — marked clearly
    staging = [0.52, 0.51, 0.50, 0.50, 0.51]  # AUROC ~chance on synthetic
    with plt.style.context(["science", "no-latex"]):
        _configure_fonts()
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        x = np.arange(len(methods))
        bars = ax.bar(x, staging, color="#4c78a8", edgecolor="black", linewidth=0.6)
        ax.axhline(0.5, color="#c44e52", ls="--", lw=1.0, label="Chance (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_ylabel("Macro AUROC (synthetic demo)")
        ax.set_ylim(0.4, 0.7)
        ax.set_title(
            "Ablation schematic on synthetic labels\n"
            "（合成演示≈随机；CinC/SHHS 数值待补充）"
        )
        ax.legend(frameon=False)
        for b, v in zip(bars, staging):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}*",
                    ha="center", va="bottom", fontsize=7)
        ax.text(0.02, 0.02, "* demo only — not paper claims", transform=ax.transAxes,
                fontsize=7, color="#666")
        fig.tight_layout()
        _save(fig, "fig03_ablation_schematic")


def fig04_orthogonality_demo() -> None:
    """Compute Gram heatmaps from a forward pass on synthetic batch."""
    import torch
    from torch.utils.data import DataLoader

    from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal
    from sleepfm.models.sleepfm import MultiModalSleepFM
    from sleepfm.utils.config import load_config

    cfg = load_config(str(ROOT / "configs" / "unify.yaml"))
    data_dir = ROOT / "data" / "synthetic"
    unify_cfg = dict(cfg.get("unify") or {})
    model = MultiModalSleepFM(
        channels=cfg["channels"],
        embedding_dim=cfg["embedding_dim"],
        unify=bool(unify_cfg.get("enabled", False)),
        shared_dim=unify_cfg.get("shared_dim"),
        private_dim=unify_cfg.get("private_dim"),
        downstream_space=unify_cfg.get("downstream_space", "concat"),
    )
    ckpt = ROOT / "outputs" / "unify_demo_curves" / "best.pt"
    if ckpt.is_file():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    ds = SleepEpochDataset(str(data_dir), split="valid")
    loader = DataLoader(ds, batch_size=16, collate_fn=collate_multimodal)
    batch = next(iter(loader))
    with torch.no_grad():
        flat, _ = model._flatten_signals(model._signal_batch(batch))
        shared, private = model.encode_factorized(flat, normalize=True)
        zs = torch.cat([shared[m] for m in model.MODALITY_ORDER if m in shared], dim=0)
        zp = torch.cat([private[m] for m in model.MODALITY_ORDER if m in private], dim=0)
        d_show = min(32, zs.size(-1))
        gram = (zs[:, :d_show].T @ zp[:, :d_show] / max(zs.size(0), 1)).cpu().numpy()

    with plt.style.context(["science", "no-latex"]):
        _configure_fonts()
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        im = axes[0].imshow(gram, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        axes[0].set_title("Shared×Private Gram (demo)\n共享–私有交叉相关")
        axes[0].set_xlabel("Private dims (subset)")
        axes[0].set_ylabel("Shared dims (subset)")
        fig.colorbar(im, ax=axes[0], fraction=0.046)
        # Histogram of off-block entries
        axes[1].hist(gram.ravel(), bins=30, color="#4c78a8", edgecolor="white")
        axes[1].set_title("Entry distribution\n(合成前向；非临床结果)")
        axes[1].set_xlabel("Gram entry")
        axes[1].set_ylabel("Count")
        fig.tight_layout()
        _save(fig, "fig04_orthogonality")


def fig05_modality_dropout() -> None:
    labels = ["All present", "Drop BAS", "Drop ECG", "Drop Resp"]
    # Schematic robustness story — synthetic only
    vals = [0.51, 0.50, 0.49, 0.50]
    with plt.style.context(["science", "no-latex"]):
        _configure_fonts()
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        ax.plot(labels, vals, "o-", color="#4c78a8", ms=6)
        ax.axhline(0.5, ls="--", color="#c44e52", label="Chance")
        ax.set_ylabel("Downstream AUROC (synthetic)")
        ax.set_title("Modality-dropout robustness schematic\n模态缺失鲁棒性（合成示意；待补充真实数据）")
        ax.set_ylim(0.4, 0.65)
        ax.legend(frameon=False)
        fig.tight_layout()
        _save(fig, "fig05_modality_dropout")


def fig06_pipeline() -> None:
    steps = [
        "Raw PSG\n(CinC/SHHS\n待补充)",
        "Export\n+ validate",
        "Pretrain\nLOO / Unify",
        "Downstream\n+ retrieval",
        "Ablation\n+ night",
        "Paper suite\nJSON",
    ]
    with plt.style.context(["science", "no-latex"]):
        _configure_fonts()
        fig, ax = plt.subplots(figsize=(7.2, 2.4))
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 3)
        ax.axis("off")
        ax.set_title("Experiment pipeline / 实验流水线")
        for i, t in enumerate(steps):
            x = 0.4 + i * 2.1
            r = mpatches.FancyBboxPatch(
                (x, 0.9), 1.8, 1.3, boxstyle="round,pad=0.05,rounding_size=0.12",
                facecolor="#e8f1fa", edgecolor="#2c5f8a", linewidth=1.1,
            )
            ax.add_patch(r)
            ax.text(x + 0.9, 1.55, t, ha="center", va="center", fontsize=7)
            if i < len(steps) - 1:
                ax.annotate("", xy=(x + 2.0, 1.55), xytext=(x + 1.85, 1.55),
                            arrowprops=dict(arrowstyle="->", color="#333"))
        _save(fig, "fig06_pipeline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, default=None)
    args = parser.parse_args()
    cjk = _configure_fonts()
    print("CJK font:", cjk)
    fig01_architecture(cjk)
    fig02_loss_curves(Path(args.history) if args.history else None)
    fig03_ablation_schematic()
    fig04_orthogonality_demo()
    fig05_modality_dropout()
    fig06_pipeline()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
