"""Channel meta checks and CinC label-coverage gates."""

from pathlib import Path

import pytest
import torch

from sleepfm.data.label_coverage import (
    apply_metric_gate,
    compute_label_coverage,
    coverage_from_entries,
    gate_claimed_metrics,
)
from sleepfm.data.synthetic import write_synthetic_dataset
from sleepfm.models.channel_meta import (
    OFFICIAL_CINC_CHANNELS,
    PAPER_CHANNELS,
    assert_channels_compatible,
    check_channel_meta,
    format_channel_triplet,
)
from sleepfm.models.official_adapter import apply_official_weights, build_model_from_official
from sleepfm.models.encoders import EffNet
from sleepfm.models.sleepfm import MultiModalSleepFM


def test_format_channel_triplet():
    assert format_channel_triplet(PAPER_CHANNELS) == "10/2/7"
    assert format_channel_triplet(OFFICIAL_CINC_CHANNELS) == "5/1/3"


def test_check_channel_meta_mismatch_fail_fast(tmp_path):
    data = tmp_path / "data"
    write_synthetic_dataset(
        data,
        PAPER_CHANNELS,
        64,
        {"pretrain": 4, "valid": 2, "train": 4, "test": 2},
        seed=0,
    )
    report = check_channel_meta(
        OFFICIAL_CINC_CHANNELS,
        data_channels=PAPER_CHANNELS,
        allow_mismatch=False,
        context="unit",
    )
    assert not report.ok
    assert any("CHANNEL MISMATCH" in m for m in report.messages)

    with pytest.raises(RuntimeError, match="CHANNEL MISMATCH"):
        assert_channels_compatible(
            OFFICIAL_CINC_CHANNELS,
            data_dir=data,
            allow_mismatch=False,
        )
    ok = assert_channels_compatible(
        OFFICIAL_CINC_CHANNELS,
        data_dir=data,
        allow_mismatch=True,
    )
    assert ok.ok


def test_label_coverage_gate_degenerate():
    entries = [{"stage_id": 0, "apnea": 0} for _ in range(20)]
    cov = coverage_from_entries(entries, dataset="cinc2018")
    assert not cov.has_aasm_staging
    assert not cov.has_respiratory_events
    gate = gate_claimed_metrics(cov)
    assert not gate.claim_staging
    assert not gate.claim_apnea
    metrics = apply_metric_gate(
        {"staging": {"macro_auroc": 0.9}, "apnea": {"auroc": 0.8}},
        gate,
    )
    assert metrics["staging"]["skipped"]
    assert metrics["apnea"]["skipped"]


def test_label_coverage_on_synthetic(tiny_data_dir):
    cov = compute_label_coverage(tiny_data_dir)
    assert cov.n_epochs > 0
    # Synthetic data has varied stages + apnea
    assert cov.has_aasm_staging
    gate = gate_claimed_metrics(cov)
    assert gate.claim_staging


def test_official_vs_paper_fail_fast_without_allow(tmp_path):
    channels = {"bas": 5, "ecg": 1, "respiratory": 3}
    blobs = {}
    mapping = {
        "bas": "sleep_stages_state_dict",
        "ecg": "ekg_state_dict",
        "respiratory": "respiratory_state_dict",
    }
    for our, key in mapping.items():
        enc = EffNet(in_channel=channels[our], embedding_dim=32)
        blobs[key] = enc.state_dict()
    path = tmp_path / "official.pt"
    torch.save({**blobs, "temperature": 0.2}, path)
    with pytest.raises(RuntimeError, match="channel"):
        build_model_from_official(
            path,
            channels=PAPER_CHANNELS,
            embedding_dim=32,
            allow_channel_mismatch=False,
        )
    # Soft path still reports mismatches
    model = MultiModalSleepFM(channels=PAPER_CHANNELS, embedding_dim=32)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    report = apply_official_weights(model, ckpt, strict=False)
    assert report.skipped_shape
    assert any("CHANNEL MISMATCH" in m or "CHANNEL" in m for m in report.messages)


def test_paper_suite_help_has_temporal_flags():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_paper_suite.py"), "--help"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "--train-temporal" in r.stdout
    assert "--temporal-checkpoint" in r.stdout
    assert "--allow-channel-mismatch" in r.stdout
