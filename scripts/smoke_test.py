"""End-to-end smoke test: synthetic data, pretrain, downstream eval."""

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal
    from sleepfm.data.synthetic import write_synthetic_dataset
    from sleepfm.eval.downstream import (
        build_embedding_matrix,
        evaluate_apnea,
        evaluate_sleep_staging,
        train_logistic_regression,
    )
    from sleepfm.models.sleepfm import MultiModalSleepFM
    from sleepfm.training.trainer import PretrainTrainer
    from sleepfm.utils.config import load_config
    from sleepfm.utils.seed import set_seed
    from torch.utils.data import DataLoader

    cfg = load_config(ROOT / "configs" / "default.yaml")
    set_seed(cfg["seed"])
    demo = cfg["demo"]
    channels = cfg["channels"]
    clip_length = int(demo["sample_rate"] * demo["clip_seconds"])

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        out_dir = Path(tmp) / "outputs"
        write_synthetic_dataset(
            data_dir,
            channels,
            clip_length,
            {
                "pretrain": demo.get("num_pretrain", demo.get("num_train", 64)),
                "valid": demo["num_val"],
                "train": demo.get("num_train", 48),
                "test": demo["num_test"],
            },
            seed=cfg["seed"],
            apnea_rate=demo.get("apnea_rate", 0.15),
            num_participants=demo.get("num_participants"),
            epochs_per_participant=demo.get("epochs_per_participant"),
        )

        from sleepfm.data.splits import assert_disjoint_splits

        assert_disjoint_splits(
            data_dir,
            [("pretrain", "train"), ("pretrain", "test"), ("train", "test")],
            by="path",
        )
        if demo.get("num_participants"):
            assert_disjoint_splits(
                data_dir,
                [("pretrain", "train"), ("pretrain", "test")],
                by="participant_id",
            )

        device = torch.device("cpu")
        model = MultiModalSleepFM(
            channels=channels,
            embedding_dim=cfg["embedding_dim"],
            temperature_init=cfg["temperature_init"],
        )

        # Forward pass
        ds = SleepEpochDataset(data_dir, split="pretrain")
        batch = collate_multimodal([ds[0], ds[1]])
        batch = {k: v.to(device) for k, v in batch.items()}
        z = model.encode(batch)
        assert all(z[m].shape[-1] == cfg["embedding_dim"] for m in z)
        loss, _ = model.contrastive_loss(batch, mode="leave_one_out")
        assert loss.ndim == 0

        train_loader = DataLoader(
            ds,
            batch_size=demo["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=collate_multimodal,
        )
        val_loader = DataLoader(
            SleepEpochDataset(data_dir, split="valid"),
            batch_size=demo["batch_size"],
            collate_fn=collate_multimodal,
        )

        trainer = PretrainTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=cfg["lr"],
            output_dir=out_dir,
            contrastive_mode="leave_one_out",
            lr_step_period=100,
        )
        trainer.fit(epochs=demo["epochs"])

        ckpt = out_dir / "best.pt"
        model = MultiModalSleepFM.from_checkpoint(str(ckpt), device="cpu")

        ds_cfg = cfg["downstream"]
        train_ds = SleepEpochDataset(
            data_dir, split=ds_cfg.get("train_split", "train"), return_labels=True
        )
        test_ds = SleepEpochDataset(
            data_dir, split=ds_cfg.get("test_split", "test"), return_labels=True
        )
        X_tr, y_tr = build_embedding_matrix(model, train_ds, device, demo["batch_size"])
        X_te, y_te = build_embedding_matrix(model, test_ds, device, demo["batch_size"])

        clf = train_logistic_regression(X_tr, y_tr["stage_id"], cfg["downstream"])
        metrics = evaluate_sleep_staging(clf, X_te, y_te["stage_id"])
        assert "macro_auroc" in metrics

        unify_model = MultiModalSleepFM(
            channels=channels,
            embedding_dim=64,
            unify=True,
            shared_dim=32,
            private_dim=32,
        )
        u_loss, u_logs = unify_model.pretrain_loss(
            batch,
            loss_weights={
                "loo": 1.0,
                "pairwise": 0.5,
                "orth": 0.1,
                "miss": 0.5,
                "temporal": 0.0,
            },
            modality_dropout=1.0,
        )
        assert torch.isfinite(u_loss)
        u_loss.backward()

        print("SMOKE TEST PASSED")
        print(f"  embedding shapes: {[v.shape for v in z.values()]}")
        print(f"  contrastive loss: {loss.item():.4f}")
        print(f"  staging metrics: {metrics}")
        print(f"  unify mixed loss: {u_loss.item():.4f} ({', '.join(k for k in u_logs if k.startswith('loss_'))})")


if __name__ == "__main__":
    main()
