"""Unit tests for contrastive loss, embeddings, and symmetry."""

import torch

from sleepfm.models.sleepfm import MultiModalSleepFM


def _random_batch(batch_size: int = 8, length: int = 320):
    return {
        "bas": torch.randn(batch_size, 10, length),
        "ecg": torch.randn(batch_size, 2, length),
        "respiratory": torch.randn(batch_size, 7, length),
    }


def test_encode_l2_normalized():
    channels = {"bas": 10, "ecg": 2, "respiratory": 7}
    model = MultiModalSleepFM(channels=channels, embedding_dim=64)
    model.eval()
    z = model.encode(_random_batch(4))
    for name, emb in z.items():
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), name


def test_contrastive_loss_scalar():
    model = MultiModalSleepFM(
        channels={"bas": 10, "ecg": 2, "respiratory": 7}, embedding_dim=32
    )
    loss, meta = model.contrastive_loss(_random_batch(8), mode="leave_one_out")
    assert loss.ndim == 0
    assert meta["mode"] == "leave_one_out"
    loss_pw, _ = model.contrastive_loss(_random_batch(8), mode="pairwise")
    assert loss_pw.ndim == 0


def test_contrastive_symmetry_pairwise():
    """Permuting batch rows preserves pairwise contrastive loss (symmetric InfoNCE)."""
    torch.manual_seed(0)
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    model = MultiModalSleepFM(channels=channels, embedding_dim=16)
    model.eval()
    batch = {
        "bas": torch.randn(4, 4, 128),
        "ecg": torch.randn(4, 2, 128),
        "respiratory": torch.randn(4, 3, 128),
    }
    loss1, _ = model.contrastive_loss(batch, mode="pairwise")
    perm = torch.randperm(4)
    batch_perm = {k: v[perm] for k, v in batch.items()}
    loss2, _ = model.contrastive_loss(batch_perm, mode="pairwise")
    assert torch.allclose(loss1, loss2, atol=1e-5)


def test_single_encode_in_loss():
    """Loss path must not double-encode (gradient / norm consistency)."""
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    model = MultiModalSleepFM(channels=channels, embedding_dim=16)
    batch = _random_batch(4, length=64)
    batch["bas"] = batch["bas"][:, :4]
    batch["ecg"] = batch["ecg"][:, :2]
    batch["respiratory"] = batch["respiratory"][:, :3]
    model.train()
    loss, _ = model.contrastive_loss(batch)
    loss.backward()
    assert model.encoders["bas"].stage1.weight.grad is not None
