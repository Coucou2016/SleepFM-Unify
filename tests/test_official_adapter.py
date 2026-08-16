"""Tests for official SleepFM checkpoint adapter."""

from pathlib import Path

import torch

from sleepfm.models.encoders import EffNet
from sleepfm.models.official_adapter import (
    apply_official_weights,
    build_model_from_official,
    convert_official_to_ours,
    is_official_checkpoint,
    save_converted_checkpoint,
)
from sleepfm.models.sleepfm import MultiModalSleepFM


def _fake_official_ckpt(channels=None, embedding_dim: int = 32):
    """Tiny fake official checkpoint (same EffNet layout, small dims)."""
    channels = channels or {"bas": 5, "ecg": 1, "respiratory": 3}
    # Build real EffNet state dicts at reduced width via in_channel only
    # (full EffNet depth — OK for unit test size)
    blobs = {}
    mapping = {
        "bas": "sleep_stages_state_dict",
        "ecg": "ekg_state_dict",
        "respiratory": "respiratory_state_dict",
    }
    for our, key in mapping.items():
        enc = EffNet(in_channel=channels[our], embedding_dim=embedding_dim)
        # Simulate DataParallel prefix on one modality
        sd = enc.state_dict()
        if our == "ecg":
            sd = {f"module.{k}": v for k, v in sd.items()}
        blobs[key] = sd
    return {
        **blobs,
        "temperature": 0.25,
        "epoch": 1,
    }


def test_is_official_checkpoint():
    ckpt = _fake_official_ckpt()
    assert is_official_checkpoint(ckpt)
    native = {
        "model_state_dict": {"encoders.bas.stage1.weight": torch.zeros(1)},
        "channels": {"bas": 10, "ecg": 2, "respiratory": 7},
    }
    assert not is_official_checkpoint(native)


def test_convert_maps_aliases_and_module_prefix():
    ckpt = _fake_official_ckpt()
    # Alias forms
    ckpt["resp_state_dict"] = ckpt.pop("respiratory_state_dict")
    converted, report = convert_official_to_ours(ckpt)
    assert report.source_keys["respiratory"] == "resp_state_dict"
    assert report.source_keys["bas"] == "sleep_stages_state_dict"
    assert report.source_keys["ecg"] == "ekg_state_dict"
    assert any(k.startswith("encoders.bas.") for k in converted)
    assert any(k.startswith("encoders.ecg.") for k in converted)
    assert not any(k.startswith("module.") for k in converted)
    assert report.inferred_channels == {"bas": 5, "ecg": 1, "respiratory": 3}
    assert report.temperature == 0.25


def test_load_matching_channels(tmp_path):
    ckpt = _fake_official_ckpt({"bas": 5, "ecg": 1, "respiratory": 3}, embedding_dim=32)
    path = tmp_path / "official.pt"
    torch.save(ckpt, path)
    model, report = build_model_from_official(
        path, channels={"bas": 5, "ecg": 1, "respiratory": 3}, embedding_dim=32
    )
    assert report.ok
    assert len(report.skipped_shape) == 0
    assert float(model.temperature.item()) == 0.25
    # Forward works
    model.eval()
    z = model.encode(
        {
            "bas": torch.randn(2, 5, 128),
            "ecg": torch.randn(2, 1, 128),
            "respiratory": torch.randn(2, 3, 128),
        }
    )
    assert z["bas"].shape == (2, 32)


def test_shape_mismatch_reported(tmp_path):
    ckpt = _fake_official_ckpt({"bas": 5, "ecg": 1, "respiratory": 3}, embedding_dim=32)
    path = tmp_path / "official.pt"
    torch.save(ckpt, path)
    # Paper channel counts → stage1 mismatches
    model = MultiModalSleepFM(
        channels={"bas": 10, "ecg": 2, "respiratory": 7}, embedding_dim=32
    )
    report = apply_official_weights(model, ckpt, strict=False)
    assert report.skipped_shape
    assert any("stage1.weight" in s for s in report.skipped_shape)


def test_from_checkpoint_auto_detect(tmp_path):
    ckpt = _fake_official_ckpt(embedding_dim=32)
    path = tmp_path / "best.pt"
    torch.save(ckpt, path)
    model = MultiModalSleepFM.from_checkpoint(str(path))
    assert set(model.channels) == {"bas", "ecg", "respiratory"}
    assert model.channels["bas"] == 5


def test_save_converted_checkpoint(tmp_path):
    ckpt = _fake_official_ckpt(embedding_dim=32)
    src = tmp_path / "official.pt"
    dst = tmp_path / "native.pt"
    torch.save(ckpt, src)
    report = save_converted_checkpoint(src, dst, embedding_dim=32)
    assert dst.is_file()
    assert report.loaded
    loaded = MultiModalSleepFM.from_checkpoint(str(dst))
    assert loaded.channels["ecg"] == 1


def test_strict_raises_on_mismatch(tmp_path):
    ckpt = _fake_official_ckpt({"bas": 5, "ecg": 1, "respiratory": 3}, embedding_dim=32)
    path = tmp_path / "official.pt"
    torch.save(ckpt, path)
    try:
        build_model_from_official(
            path,
            channels={"bas": 10, "ecg": 2, "respiratory": 7},
            embedding_dim=32,
            strict=True,
        )
        raised = False
    except RuntimeError:
        raised = True
    assert raised
