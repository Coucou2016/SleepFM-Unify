"""Night-level dataset, temporal encoder, and placeholder labels."""

import math

import torch

from sleepfm.data.night_dataset import (
    NightSequenceDataset,
    collate_night,
    group_entries_by_night,
    night_summary_from_entries,
)
from sleepfm.eval.night import (
    epoch_sequence_kappa,
    night_embedding_table,
    night_eval_pack,
    probe_night_tasks,
)
from sleepfm.models.sleepfm import MultiModalSleepFM
from sleepfm.models.temporal import NightTemporalEncoder, temporal_losses


def test_group_and_summary():
    entries = [
        {"path": "a.npy", "participant_id": "P1", "epoch_index": 1, "stage_id": 2, "apnea": 1},
        {"path": "b.npy", "participant_id": "P1", "epoch_index": 0, "stage_id": 0, "apnea": 0},
        {"path": "c.npy", "participant_id": "P2", "epoch_index": 0, "stage_id": 4, "apnea": 0},
    ]
    groups = group_entries_by_night(entries)
    assert list(groups[("P1", "P1")])[0]["path"] == "b.npy"
    summary = night_summary_from_entries(groups[("P1", "P1")], epoch_seconds=30.0)
    assert summary["n_epochs"] == 2
    assert math.isfinite(summary["ahi"])
    assert 0.0 <= summary["sleep_efficiency"] <= 1.0


def test_night_sequence_dataset(tiny_data_dir):
    ds = NightSequenceDataset(tiny_data_dir, split="pretrain", window=2, stride=2, min_len=2)
    assert len(ds) >= 1
    item = ds[0]
    assert item["bas"].ndim == 3  # (L, C, T)
    assert item["bas"].shape[0] == 2
    assert item["present_mask"].shape == (2, 3)
    batch = collate_night([ds[0], ds[min(1, len(ds) - 1)]])
    assert batch["bas"].ndim == 4


def test_temporal_encoder_and_losses():
    enc = NightTemporalEncoder(d_model=16, n_layers=1, kind="gru")
    x = torch.randn(3, 6, 16)
    y = enc(x)
    assert y.shape == (3, 6, 16)
    loss, logs = temporal_losses(enc, x, mask_prob=0.5)
    assert math.isfinite(float(loss.item()))
    loss.backward()
    tr = NightTemporalEncoder(d_model=16, n_layers=1, n_heads=4, kind="transformer")
    z = tr(x)
    assert z.shape == x.shape


def test_gru_temporal_respects_padding_mask():
    """Pad positions must not change outputs on valid timesteps (pack_padded)."""
    torch.manual_seed(0)
    enc = NightTemporalEncoder(d_model=8, n_layers=1, kind="gru")
    enc.eval()
    x = torch.randn(2, 5, 8)
    pad = torch.zeros(2, 5, dtype=torch.bool)
    pad[0, 3:] = True
    pad[1, 4:] = True
    with torch.no_grad():
        y_pad = enc(x, padding_mask=pad)
        y_trim0 = enc(x[0:1, :3], padding_mask=None)
        y_trim1 = enc(x[1:2, :4], padding_mask=None)
    assert torch.allclose(y_pad[0, :3], y_trim0[0], atol=1e-5)
    assert torch.allclose(y_pad[1, :4], y_trim1[0], atol=1e-5)


def test_night_eval_helpers(tiny_data_dir):
    model = MultiModalSleepFM(
        channels={"bas": 10, "ecg": 2, "respiratory": 7}, embedding_dim=16
    )
    device = torch.device("cpu")
    X_tr, s_tr, _ = night_embedding_table(model, str(tiny_data_dir), "train", device, batch_size=4)
    X_te, s_te, _ = night_embedding_table(model, str(tiny_data_dir), "test", device, batch_size=4)
    assert X_tr.ndim == 2
    metrics = probe_night_tasks(X_tr, s_tr, X_te, s_te)
    assert "ahi_bin" in metrics
    assert "sleep_efficiency" in metrics


def test_night_eval_temporal_and_kappa(tiny_data_dir):
    model = MultiModalSleepFM(
        channels={"bas": 10, "ecg": 2, "respiratory": 7},
        embedding_dim=16,
        unify=True,
        shared_dim=8,
        private_dim=8,
    )
    enc = NightTemporalEncoder(d_model=8, n_layers=1, kind="gru")
    device = torch.device("cpu")
    pack_tr = night_eval_pack(
        model, str(tiny_data_dir), "train", device, batch_size=4, temporal_encoder=enc
    )
    pack_te = night_eval_pack(
        model, str(tiny_data_dir), "test", device, batch_size=4, temporal_encoder=enc
    )
    assert pack_tr["used_temporal"]
    assert pack_tr["X"].shape[1] == 8
    X_mean, _, _ = night_embedding_table(model, str(tiny_data_dir), "train", device, batch_size=4)
    assert X_mean.shape[1] == 48
    kappa = epoch_sequence_kappa(
        pack_tr["X_epochs"], pack_tr["y_stage"], pack_te["X_epochs"], pack_te["y_stage"]
    )
    assert "cohen_kappa" in kappa or "note" in kappa or "error" in kappa
    metrics = probe_night_tasks(
        pack_tr["X"], pack_tr["summaries"], pack_te["X"], pack_te["summaries"]
    )
    assert "ahi_bin" in metrics


def test_temporal_from_checkpoint_roundtrip(tmp_path):
    enc = NightTemporalEncoder(d_model=8, n_layers=1, kind="gru", window=4)
    path = tmp_path / "with_temporal.pt"
    torch.save(
        {
            "temporal_state_dict": enc.state_dict(),
            "temporal_cfg": {
                "d_model": enc.d_model,
                "kind": enc.kind,
                "n_layers": enc.n_layers,
                "n_heads": enc.n_heads,
                "max_len": enc.max_len,
                "window": enc.window,
            },
        },
        path,
    )
    loaded = NightTemporalEncoder.from_checkpoint(str(path))
    assert loaded is not None
    assert loaded.d_model == 8
    assert loaded.window == 4
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        assert torch.allclose(enc(x), loaded(x), atol=1e-5)
    assert NightTemporalEncoder.from_checkpoint({"model_state_dict": {}}) is None
