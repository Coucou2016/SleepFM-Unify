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


def test_tune_lr_c_returns_float(tiny_data_dir, frozen_model):
    device = torch.device("cpu")
    valid_ds = SleepEpochDataset(tiny_data_dir, split="valid", return_labels=True)
    X, y = build_embedding_matrix(frozen_model, valid_ds, device, 8)
    c = tune_lr_c(X, y["stage_id"], {"c_grid": [0.1, 1.0], "task": "staging"})
    assert isinstance(c, float)


def test_apnea_single_class_nan():
    clf = LogisticRegression().fit([[0], [1], [2]], [0, 1, 0])
    m = evaluate_apnea(clf, np.array([[0.5], [1.5]]), np.array([0, 0]))
    assert np.isnan(m["auroc"])
