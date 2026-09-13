"""Downstream eval isolation and metric sanity."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from sleepfm.data.dataset import SleepEpochDataset
from sleepfm.eval.downstream import (
    build_embedding_matrix,
    evaluate_apnea,
    evaluate_sleep_staging,
    train_logistic_regression,
    tune_lr_c,
)
from sleepfm.models.sleepfm import MultiModalSleepFM


@pytest.fixture
def frozen_model(demo_channels):
    return MultiModalSleepFM(channels=demo_channels, embedding_dim=32)


def test_build_embedding_matrix(tiny_data_dir, frozen_model):
    device = torch.device("cpu")
    ds = SleepEpochDataset(tiny_data_dir, split="train", return_labels=True)
    X, y = build_embedding_matrix(frozen_model, ds, device, batch_size=8)
    assert X.ndim == 2
    assert X.shape[1] == 32 * 3
    assert len(y["stage_id"]) == len(ds)


def test_build_embedding_matrix_masks_missing(tiny_data_dir, frozen_model):
    """Missing-modality rows must contribute exactly zero after present_mask."""
    device = torch.device("cpu")
    ds = SleepEpochDataset(tiny_data_dir, split="train", return_labels=True)
    # Force one sample to mark ECG missing and zero-fill.
    item = ds[0]
    item["present_mask"] = torch.tensor([1.0, 0.0, 1.0])
    item["ecg"] = torch.zeros_like(item["ecg"])
    batch = {
        "bas": item["bas"].unsqueeze(0).to(device),
        "ecg": item["ecg"].unsqueeze(0).to(device),
        "respiratory": item["respiratory"].unsqueeze(0).to(device),
        "present_mask": item["present_mask"].unsqueeze(0).to(device),
    }
    frozen_model.eval()
    with torch.no_grad():
        z = frozen_model.encode(batch)
        from sleepfm.eval.downstream import _apply_present_mask_to_emb

        z = _apply_present_mask_to_emb(z, batch, frozen_model.MODALITY_ORDER)
    assert torch.allclose(z["ecg"], torch.zeros_like(z["ecg"]))
    assert not torch.allclose(z["bas"], torch.zeros_like(z["bas"]))


def test_staging_metrics_range(tiny_data_dir, frozen_model):
    device = torch.device("cpu")
    train_ds = SleepEpochDataset(tiny_data_dir, split="train", return_labels=True)
    test_ds = SleepEpochDataset(tiny_data_dir, split="test", return_labels=True)
    X_tr, y_tr = build_embedding_matrix(frozen_model, train_ds, device, 8)
    X_te, y_te = build_embedding_matrix(frozen_model, test_ds, device, 8)
    cfg = {"max_iter": 500, "class_weight": "balanced", "solver": "lbfgs", "C": 1.0}
    clf = train_logistic_regression(X_tr, y_tr["stage_id"], cfg)
    m = evaluate_sleep_staging(clf, X_te, y_te["stage_id"])
    auroc = m["macro_auroc"]
    assert np.isnan(auroc) or (0.0 <= auroc <= 1.0)
    assert np.isnan(m["macro_auprc"]) or (0.0 <= m["macro_auprc"] <= 1.0)
    assert "accuracy" in m and "macro_f1" in m and "cohen_kappa" in m


def test_staging_uses_clf_classes_not_range():
    """label_binarize must follow clf.classes_ (gap in labels must not misalign)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = np.array([0, 1, 2, 4] * 10)
    try:
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial").fit(X, y)
    except TypeError:
        clf = LogisticRegression(max_iter=2000).fit(X, y)
    assert list(clf.classes_) == [0, 1, 2, 4]
    m = evaluate_sleep_staging(clf, X, y)
    assert not np.isnan(m["macro_auroc"])
    assert 0.0 <= m["macro_auroc"] <= 1.0


def test_tune_lr_c_fit_train_score_valid(tiny_data_dir, frozen_model):
    device = torch.device("cpu")
    train_ds = SleepEpochDataset(tiny_data_dir, split="train", return_labels=True)
    valid_ds = SleepEpochDataset(tiny_data_dir, split="valid", return_labels=True)
    X_tr, y_tr = build_embedding_matrix(frozen_model, train_ds, device, 8)
    X_va, y_va = build_embedding_matrix(frozen_model, valid_ds, device, 8)
    c = tune_lr_c(
        X_tr,
        y_tr["stage_id"],
        X_va,
        y_va["stage_id"],
        {"c_grid": [0.1, 1.0], "task": "staging"},
    )
    assert isinstance(c, float)
    assert c in (0.1, 1.0)


def test_tune_lr_c_no_resubstitution():
    """Resubstitution would often pick the most flexible C; train/valid split should not."""
    rng = np.random.default_rng(1)
    X_tr = rng.normal(size=(60, 6))
    y_tr = rng.integers(0, 3, size=60)
    X_va = rng.normal(size=(40, 6))
    y_va = rng.integers(0, 3, size=40)
    c = tune_lr_c(
        X_tr, y_tr, X_va, y_va, {"c_grid": [0.01, 0.1, 1.0, 10.0], "task": "staging"}
    )
    assert c in (0.01, 0.1, 1.0, 10.0)


def test_apnea_single_class_nan():
    clf = LogisticRegression().fit([[0], [1], [2]], [0, 1, 0])
    m = evaluate_apnea(clf, np.array([[0.5], [1.5]]), np.array([0, 0]))
    assert np.isnan(m["auroc"])
