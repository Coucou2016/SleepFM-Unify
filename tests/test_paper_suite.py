"""Tests for check_data_ready, paper suite dry-run, SeqStagingBaseline."""

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _run_check(*args: str) -> subprocess.CompletedProcess:
    # Force UTF-8 so Windows GBK consoles do not drop stdout on ASCII dashes.
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_data_ready.py"), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_check_data_ready_empty(tmp_path):
    empty = tmp_path / "raw"
    empty.mkdir()
    r = _run_check("--data-dir", str(empty))
    assert r.returncode != 0
    assert "no EDF" in (r.stdout or "").lower() or "ERROR" in (r.stdout or "")


def test_check_data_ready_exported(tiny_data_dir):
    r = _run_check("--data-dir", str(tiny_data_dir))
    assert r.returncode == 0
    assert "index.json" in (r.stdout or "")


def test_check_data_ready_edf_guess(tmp_path):
    raw = tmp_path / "cinc2018"
    raw.mkdir()
    (raw / "tr01.edf").write_bytes(b"\x00" * 16)
    r = _run_check("--data-dir", str(raw), "--dataset", "cinc2018")
    assert r.returncode == 0
    assert "export_edf" in (r.stdout or "")


def test_seq_staging_baseline_shapes():
    from sleepfm.models.encoders import SeqStagingBaseline

    m = SeqStagingBaseline(in_channel=19, num_classes=5, hidden=32)
    x = torch.randn(2, 19, 256)
    assert m(x).shape == (2, 5)
    seq = torch.randn(2, 8, 19, 256)
    pad = torch.zeros(2, 8, dtype=torch.bool)
    pad[:, -2:] = True
    out = m(seq, padding_mask=pad)
    assert out.shape == (2, 8, 5)


def test_paper_suite_demo(tmp_path):
    """Dry-run suite helpers on synthetic data (avoids long dual-pretrain in CI)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_paper_suite", ROOT / "scripts" / "run_paper_suite.py"
    )
    suite = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(suite)

    from sleepfm.data.synthetic import write_synthetic_dataset
    from sleepfm.utils.config import load_config

    cfg = load_config(ROOT / "configs" / "default.yaml")
    data = tmp_path / "data"
    out = tmp_path / "suite"
    out.mkdir()
    demo = cfg["demo"]
    write_synthetic_dataset(
        data,
        cfg["channels"],
        int(demo["sample_rate"] * demo["clip_seconds"]),
        {
            "pretrain": 32,
            "valid": 8,
            "train": 24,
            "test": 16,
        },
        seed=0,
        num_participants=12,
        epochs_per_participant=2,
        apnea_rate=0.25,
    )
    v = suite._run_validate(str(data), strict=True)
    assert v["ok"]
    # One short LOO pretrain only
    ckpt = suite._pretrain(
        cfg,
        str(data),
        out / "pretrain_loo",
        demo=True,
        unify=False,
        config_path=str(ROOT / "configs" / "default.yaml"),
    )
    assert Path(ckpt).is_file()
    down = suite._downstream(cfg, str(data), ckpt, batch_size=8)
    assert "staging" in down
    ret = suite._retrieval(ckpt, str(data), "pretrain", max_gallery=16)
    assert ret["n"] <= 16
    ab = suite._ablation(cfg, ckpt, str(data), batch_size=8)
    assert ab
    fs = suite._fewshot(cfg, ckpt, str(data), [1, 2], batch_size=8)
    assert fs
    night = suite._night(ckpt, str(data), batch_size=8)
    assert "temporal_head" in night
    summary = {
        "validate": v,
        "downstream": down,
        "retrieval": ret,
        "modality_ablation": ab,
        "fewshot": fs,
        "night": night,
        "checkpoint": ckpt,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    assert (out / "summary.json").is_file()


def test_paper_suite_cli_help():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_paper_suite.py"), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "--demo" in r.stdout
    assert "--train-temporal" in r.stdout
    assert "--temporal-checkpoint" in r.stdout
    assert "--allow-channel-mismatch" in r.stdout
