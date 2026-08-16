"""Participant isolation remains required; exporter splits must not leak."""

from sleepfm.data.splits import assert_disjoint_splits, assign_participant_splits
from sleepfm.eval.experiments import MODALITY_COMBOS, unique_participants
from sleepfm.data.dataset import SleepEpochDataset


def test_seven_modality_combos():
    assert len(MODALITY_COMBOS) == 7
    assert ("bas", "ecg", "respiratory") in MODALITY_COMBOS
    assert ("ecg", "respiratory") in MODALITY_COMBOS


def test_fewshot_participant_filter(tiny_data_dir):
    full = SleepEpochDataset(tiny_data_dir, split="train", return_labels=True)
    pids = unique_participants(full.entries)
    assert pids
    subset = SleepEpochDataset(
        tiny_data_dir, split="train", return_labels=True, participant_ids=pids[:1]
    )
    assert len(subset) <= len(full)
    assert all(e["participant_id"] == pids[0] for e in subset.entries)


def test_assign_splits_no_leakage_vs_dataset(tiny_data_dir):
    assert_disjoint_splits(
        tiny_data_dir,
        [("pretrain", "train"), ("pretrain", "test"), ("train", "test")],
        by="participant_id",
    )
    mapping = assign_participant_splits(["A", "B", "C", "D"], seed=0)
    assert len(set(mapping.values())) == 4
