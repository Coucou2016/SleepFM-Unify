"""Participant isolation remains required; exporter splits must not leak."""

from sleepfm.data.splits import assert_disjoint_splits, assign_participant_splits
from sleepfm.eval.experiments import (
    MODALITY_COMBOS,
    _metric_ci95,
    summarize_fewshot_runs,
    unique_participants,
)
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


def test_fewshot_ci95_summary():
    s = _metric_ci95([0.4, 0.5, 0.6, 0.55, 0.45])
    assert s["n"] == 5
    assert s["ci95_low"] < s["mean"] < s["ci95_high"]
    assert "±" in s["mean_pm_ci95"]
    agg = summarize_fewshot_runs(
        [
            {
                "staging": {"macro_auroc": 0.5, "macro_auprc": 0.4},
                "apnea": {"auroc": 0.6, "auprc": 0.55},
            },
            {
                "staging": {"macro_auroc": 0.55, "macro_auprc": 0.42},
                "apnea": {"auroc": 0.58, "auprc": 0.5},
            },
        ]
    )
    assert agg["n_repeats"] == 2
    assert agg["staging_macro_auroc"]["n"] == 2


def test_assign_splits_no_leakage_vs_dataset(tiny_data_dir):
    assert_disjoint_splits(
        tiny_data_dir,
        [("pretrain", "train"), ("pretrain", "test"), ("train", "test")],
        by="participant_id",
    )
    mapping = assign_participant_splits(["A", "B", "C", "D"], seed=0)
    assert len(set(mapping.values())) == 4
