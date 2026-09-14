"""Dataset split and participant isolation tests."""

import json
import tempfile
from pathlib import Path

import pytest

from sleepfm.data.splits import (
    PAPER_SPLIT_PAIRS,
    assert_disjoint_splits,
    assert_paper_isolation,
    downstream_isolation_ok,
    split_overlap,
)
from sleepfm.data.synthetic import write_synthetic_dataset
from sleepfm.data.validate import validate_dataset


def test_participant_disjoint(tiny_data_dir):
    assert_disjoint_splits(
        tiny_data_dir,
        [("pretrain", "train"), ("pretrain", "test"), ("pretrain", "valid")],
        by="participant_id",
    )


def test_path_disjoint(tiny_data_dir):
    assert_disjoint_splits(
        tiny_data_dir,
        list(PAPER_SPLIT_PAIRS),
        by="path",
    )


def test_downstream_isolation(tiny_data_dir):
    checks = downstream_isolation_ok(tiny_data_dir)
    assert all(checks.values()), checks
    assert_paper_isolation(tiny_data_dir, strict=True)


def test_validate_dataset(tiny_data_dir):
    ok, msgs = validate_dataset(tiny_data_dir)
    assert ok, msgs


def test_train_test_participant_leak_fails(tmp_path):
    """Strict/paper mode must raise when train and test share a participant."""
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    data_dir = tmp_path / "leak"
    write_synthetic_dataset(
        data_dir,
        channels,
        clip_length=32,
        splits={"pretrain": 8, "valid": 4, "train": 8, "test": 4},
        seed=0,
        num_participants=8,
        epochs_per_participant=2,
    )
    index_path = data_dir / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    # Inject train participant into test.
    leak_pid = payload["splits"]["train"][0]["participant_id"]
    payload["splits"]["test"][0]["participant_id"] = leak_pid
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    has, overlap = split_overlap(data_dir, "train", "test", by="participant_id")
    assert has and leak_pid in overlap
    with pytest.raises(RuntimeError, match="isolation|leak"):
        assert_paper_isolation(data_dir, strict=True)


def test_strict_missing_participant_id_fail_closed(tmp_path):
    """Strict mode must raise when participant_id fields are absent (not silent True)."""
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    data_dir = tmp_path / "no_pid"
    write_synthetic_dataset(
        data_dir,
        channels,
        clip_length=32,
        splits={"pretrain": 8, "valid": 4, "train": 8, "test": 4},
        seed=0,
        num_participants=8,
        epochs_per_participant=2,
    )
    index_path = data_dir / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    for split in payload["splits"].values():
        for entry in split:
            entry.pop("participant_id", None)
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="participant_id|fail-closed|isolation"):
        assert_paper_isolation(data_dir, strict=True)


def test_pretrain_train_no_overlap_integration():
    """Regression: downstream LR must not see pretrain epoch files."""
    channels = {"bas": 4, "ecg": 2, "respiratory": 3}
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        write_synthetic_dataset(
            data_dir,
            channels,
            clip_length=64,
            splits={"pretrain": 20, "valid": 8, "train": 16, "test": 8},
            seed=1,
            num_participants=10,
            epochs_per_participant=2,
        )
        has, overlap = split_overlap(data_dir, "pretrain", "train", by="path")
        assert not has, overlap
        has_p, overlap_p = split_overlap(data_dir, "pretrain", "train", by="participant_id")
        assert not has_p, overlap_p
        assert_paper_isolation(data_dir, strict=True)
