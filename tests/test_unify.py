"""SleepFM-Unify: shared/private shapes, orthogonality, missing-modality LOO."""

import math

import torch

from sleepfm.models.sleepfm import MultiModalSleepFM, orthogonality_loss
from sleepfm.training.trainer import PretrainTrainer
from torch.utils.data import DataLoader

from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal


def _batch(bs=8, length=64, channels=None):
    channels = channels or {"bas": 10, "ecg": 2, "respiratory": 7}
    return {
        "bas": torch.randn(bs, channels["bas"], length),
        "ecg": torch.randn(bs, channels["ecg"], length),
        "respiratory": torch.randn(bs, channels["respiratory"], length),
        "present_mask": torch.ones(bs, 3),
    }


def test_unify_shared_private_shapes():
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    model = MultiModalSleepFM(
        channels=channels, embedding_dim=32, unify=True, shared_dim=16, private_dim=16
    )
    model.eval()
    batch = _batch(6, 48, channels)
    shared, private = model.encode_factorized(batch)
    for name in ("bas", "ecg", "respiratory"):
        assert shared[name].shape == (6, 16)
        assert private[name].shape == (6, 16)
        assert torch.allclose(shared[name].norm(dim=-1), torch.ones(6), atol=1e-4)
    down = model.encode(batch, space="downstream")
    assert down["bas"].shape[-1] == 32
    contrastive = model.encode(batch, space="shared")
    assert contrastive["bas"].shape[-1] == 16


def test_orthogonality_finite():
    torch.manual_seed(0)
    s = torch.randn(12, 16)
    p = torch.randn(12, 16)
    loss = orthogonality_loss(s, p)
    assert loss.ndim == 0
    assert math.isfinite(float(loss.item()))
    assert float(loss.item()) >= 0.0


def test_unify_contrastive_uses_shared_only():
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    model = MultiModalSleepFM(
        channels=channels, embedding_dim=32, unify=True, shared_dim=8, private_dim=24
    )
    loss, meta = model.contrastive_loss(_batch(8, 64, channels), mode="leave_one_out")
    assert loss.ndim == 0
    assert math.isfinite(float(loss.item()))
    assert meta["mode"] == "leave_one_out"


def test_loo_with_missing_modality():
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    model = MultiModalSleepFM(channels=channels, embedding_dim=16)
    batch = _batch(8, 48, channels)
    batch["present_mask"][:, 2] = 0  # drop respiratory
    loss, meta = model.contrastive_loss(batch, mode="leave_one_out")
    assert math.isfinite(float(loss.item()))
    assert meta["num_modalities"] == 2
    # never crash if a key is absent
    del batch["respiratory"]
    batch["present_mask"] = torch.tensor([[1.0, 1.0, 0.0]] * 8)
    loss2, _ = model.contrastive_loss(batch, mode="leave_one_out")
    assert math.isfinite(float(loss2.item()))


def test_mixed_loss_backward():
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    model = MultiModalSleepFM(
        channels=channels, embedding_dim=16, unify=True, shared_dim=8, private_dim=8
    )
    model.train()
    loss, logs = model.pretrain_loss(
        _batch(8, 48, channels),
        mode="leave_one_out",
        loss_weights={"loo": 1.0, "pairwise": 0.5, "orth": 0.1, "miss": 0.5, "temporal": 0.0},
        modality_dropout=1.0,
    )
    assert math.isfinite(float(loss.item()))
    loss.backward()
    assert model.encoders["bas"].stage1.weight.grad is not None
    assert model.proj_shared["bas"].weight.grad is not None
    assert "loss_loo" in logs


def test_l_miss_respects_present_mask():
    """Naturally missing modalities must not enter the remaining mean for L_miss."""
    torch.manual_seed(0)
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    model = MultiModalSleepFM(
        channels=channels, embedding_dim=16, unify=True, shared_dim=8, private_dim=8
    )
    model.train()
    batch = _batch(8, 48, channels)
    # Mark respiratory naturally missing; force-drop another via dropout path
    batch["present_mask"][:, 2] = 0.0
    # Call pretrain_loss with miss weight; monkeypatch dropout to always drop ecg
    orig = model._apply_modality_dropout

    def force_drop(present_mask, batch_size, device, p):
        if present_mask is None:
            present_mask = torch.ones(batch_size, 3, device=device)
        present_mask = present_mask.clone()
        present_mask[:, 1] = 0.0  # drop ecg
        return present_mask, "ecg"

    model._apply_modality_dropout = force_drop
    loss, logs = model.pretrain_loss(
        batch,
        mode="leave_one_out",
        loss_weights={"loo": 0.0, "pairwise": 0.0, "orth": 0.0, "miss": 1.0, "temporal": 0.0},
        modality_dropout=1.0,
    )
    model._apply_modality_dropout = orig
    assert math.isfinite(float(loss.item()))
    assert logs.get("dropped_modality") == "ecg"
    assert "loss_miss" in logs


def test_empty_loader_raises():
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    model = MultiModalSleepFM(channels=channels, embedding_dim=16)
    empty = DataLoader([], batch_size=2)
    trainer = PretrainTrainer(
        model=model,
        train_loader=empty,
        val_loader=None,
        device=torch.device("cpu"),
        output_dir="outputs/_empty_loader_test",
    )
    try:
        trainer._step_epoch(empty, train=True)
        assert False, "expected RuntimeError for empty loader"
    except RuntimeError as exc:
        assert "empty" in str(exc).lower()


def test_mixed_loss_single_encode_and_backward():
    """All mixed-loss terms must reuse one backbone encode per step."""
    from sleepfm.models.temporal import NightTemporalEncoder

    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    model = MultiModalSleepFM(
        channels=channels, embedding_dim=16, unify=True, shared_dim=8, private_dim=8
    )
    model.train()
    calls = {"n": 0}
    orig = model.encode_backbone

    def counted(batch):
        calls["n"] += 1
        return orig(batch)

    model.encode_backbone = counted
    batch = {
        "bas": torch.randn(2, 4, 4, 48),
        "ecg": torch.randn(2, 4, 2, 48),
        "respiratory": torch.randn(2, 4, 3, 48),
        "present_mask": torch.ones(2, 4, 3),
        "padding_mask": torch.zeros(2, 4, dtype=torch.bool),
    }
    enc = NightTemporalEncoder(d_model=8, n_layers=1, kind="gru")
    loss, logs = model.pretrain_loss(
        batch,
        mode="leave_one_out",
        loss_weights={"loo": 1.0, "pairwise": 0.5, "orth": 0.1, "miss": 0.5, "temporal": 0.2},
        modality_dropout=1.0,
        temporal_encoder=enc,
    )
    assert calls["n"] == 1
    assert math.isfinite(float(loss.item()))
    loss.backward()
    assert model.encoders["bas"].stage1.weight.grad is not None
    assert model.proj_shared["bas"].weight.grad is not None
    assert enc.out.weight.grad is not None
    assert "loss_loo" in logs
    assert "loss_temporal" in logs


def test_unify_checkpoint_roundtrip(tmp_path):
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    model = MultiModalSleepFM(
        channels=channels, embedding_dim=16, unify=True, shared_dim=8, private_dim=8
    )
    path = tmp_path / "unify.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "channels": channels,
            "embedding_dim": 16,
            "unify": True,
            "shared_dim": 8,
            "private_dim": 8,
            "temperature": 0.0,
        },
        path,
    )
    loaded = MultiModalSleepFM.from_checkpoint(str(path))
    assert loaded.unify
    assert loaded.shared_dim == 8
    model.eval()
    loaded.eval()
    batch = _batch(4, 32, channels)
    with torch.no_grad():
        a = model.encode(batch)["bas"]
        b = loaded.encode(batch)["bas"]
    assert torch.allclose(a, b, atol=1e-5)


def test_baseline_checkpoint_still_loads(tmp_path):
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    model = MultiModalSleepFM(channels=channels, embedding_dim=16, unify=False)
    path = tmp_path / "base.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "channels": channels,
            "embedding_dim": 16,
            "temperature": 0.0,
        },
        path,
    )
    loaded = MultiModalSleepFM.from_checkpoint(str(path))
    assert not loaded.unify


def test_unify_trainer_one_step(tiny_data_dir):
    device = torch.device("cpu")
    model = MultiModalSleepFM(
        channels={"bas": 10, "ecg": 2, "respiratory": 7},
        embedding_dim=32,
        unify=True,
        shared_dim=16,
        private_dim=16,
    )
    loader = DataLoader(
        SleepEpochDataset(tiny_data_dir, split="pretrain"),
        batch_size=4,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_multimodal,
    )
    trainer = PretrainTrainer(
        model,
        loader,
        None,
        device,
        output_dir=tiny_data_dir / "out",
        use_mixed_loss=True,
        loss_weights={"loo": 1.0, "pairwise": 0.2, "orth": 0.05, "miss": 0.2, "temporal": 0.0},
        modality_dropout=0.5,
        lr_step_period=100,
    )
    hist = trainer.fit(epochs=1)
    assert math.isfinite(hist["train_loss"][0])
    assert (tiny_data_dir / "out" / "best.pt").is_file()
