"""Shared fixtures for SleepFM unit tests."""

from pathlib import Path

import pytest

from sleepfm.data.synthetic import write_synthetic_dataset


@pytest.fixture
def demo_channels():
    return {"bas": 10, "ecg": 2, "respiratory": 7}


@pytest.fixture
def tiny_data_dir(tmp_path, demo_channels):
    """Participant-level synthetic dataset in a temp directory."""
    data_dir = tmp_path / "data"
    write_synthetic_dataset(
        data_dir,
        demo_channels,
        clip_length=320,
        splits={"pretrain": 16, "valid": 8, "train": 12, "test": 16},
        seed=42,
        num_participants=12,
        epochs_per_participant=2,
        apnea_rate=0.25,
    )
    return data_dir
