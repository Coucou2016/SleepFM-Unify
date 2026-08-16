"""Dataset split and participant isolation tests."""

import pytest

from sleepfm.data.splits import assert_disjoint_splits, downstream_isolation_ok, split_overlap
from sleepfm.data.synthetic import write_synthetic_dataset
from sleepfm.data.validate import validate_dataset


def test_participant_disjoint(tiny_data_dir):
    assert_disjoint_splits(
        tiny_data_dir,
        [("pretrain", "train"), ("pretrain", "test")],
        by="participant_id",
    )


def test_path_disjoint(tiny_data_dir):
    assert_disjoint_splits(
        tiny_data_dir,
        [("pretrain", "train"), ("pretrain", "test"), ("train", "test")],
        by="path",
    )


def test_downstream_isolation(tiny_data_dir):
    checks = downstream_isolation_ok(tiny_data_dir)
    assert all(checks.values()), checks


def test_validate_dataset(tiny_data_dir):
    ok, msgs = validate_dataset(tiny_data_dir)
    assert ok, msgs


def test_pretrain_train_no_overlap_integration():
    """Regression: downstream LR must not see pretrain epoch files."""
    import tempfile
    from pathlib import Path

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
