"""Downstream evaluation with logistic regression on frozen embeddings (paper Sec 3.2)."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal
from sleepfm.models.sleepfm import MultiModalSleepFM


def _collate_with_labels(batch):
    signals = collate_multimodal(batch)
    labels = {
        "stage_id": torch.tensor([b["labels"]["stage_id"] for b in batch]),
        "apnea": torch.tensor([b["labels"]["apnea"] for b in batch]),
    }
    return signals, labels


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_embedding_matrix(
    model: MultiModalSleepFM,
    dataset: SleepEpochDataset,
    device: torch.device,
    batch_size: int = 32,
    keep_modalities: Optional[Sequence[str]] = None,
    space: str = "downstream",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_with_labels,
    )
    labels_stage, labels_apnea = [], []
    emb_chunks = []
    keep = set(keep_modalities) if keep_modalities is not None else None

    with torch.no_grad():
        for batch, lab in loader:
            batch = _batch_to_device(batch, device)
            z = model.encode(batch, space=space)
            parts = []
            for m in model.MODALITY_ORDER:
                if m not in z:
                    continue
                vec = z[m]
                if keep is not None and m not in keep:
                    vec = torch.zeros_like(vec)
                parts.append(vec)
            combined = torch.cat(parts, dim=-1)
            emb_chunks.append(combined.cpu().numpy())
            labels_stage.extend(lab["stage_id"].tolist())
            labels_apnea.extend(lab["apnea"].tolist())

    X = np.concatenate(emb_chunks, axis=0)
    y = {
        "stage_id": np.array(labels_stage, dtype=np.int64),
        "apnea": np.array(labels_apnea, dtype=np.int64),
    }
    return X, y


def tune_lr_c(
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    downstream_cfg: dict,
    c_grid: Optional[list] = None,
) -> float:
    """Select L2 C by macro-AUROC on validation embeddings (staging) or AUROC (binary)."""
    if c_grid is None:
        c_grid = downstream_cfg.get("c_grid", [0.01, 0.1, 1.0, 10.0, 100.0])
    best_c = downstream_cfg.get("C", 1.0)
    best_score = -1.0
    task = downstream_cfg.get("task", "staging")
    for c in c_grid:
        cfg = {**downstream_cfg, "C": c}
        clf = train_logistic_regression(X_valid, y_valid, cfg)
        if task == "apnea":
            score = evaluate_apnea(clf, X_valid, y_valid).get("auroc", float("nan"))
        else:
            score = evaluate_sleep_staging(clf, X_valid, y_valid)["macro_auroc"]
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_c = c
    return float(best_c)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    downstream_cfg: dict,
) -> LogisticRegression:
    cfg = {
        k: v
        for k, v in downstream_cfg.items()
        if k
        not in (
            "train_split",
            "test_split",
            "valid_split",
            "tune_c_on_valid",
            "c_grid",
            "task",
        )
    }
    kwargs = dict(
        max_iter=cfg.get("max_iter", 10000),
        class_weight=cfg.get("class_weight", "balanced"),
        solver=cfg.get("solver", "lbfgs"),
    )
    penalty = cfg.get("penalty", "l2")
    if penalty == "l2":
        kwargs["C"] = cfg.get("C", 1.0)
    elif penalty == "l1":
        kwargs["penalty"] = "l1"
        kwargs["C"] = cfg.get("C", 1.0)
    elif penalty is not None:
        kwargs["penalty"] = penalty
        kwargs["C"] = cfg.get("C", 1.0)
    # sklearn >= 1.5 removed multi_class; multinomial is default for multiclass LBFGS
    try:
        if len(np.unique(y_train)) > 2:
            clf = LogisticRegression(multi_class="multinomial", **kwargs)
        else:
            clf = LogisticRegression(**kwargs)
    except TypeError:
        clf = LogisticRegression(**kwargs)
    clf.fit(X_train, y_train)
    return clf


def evaluate_sleep_staging(
    clf: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    proba = clf.predict_proba(X_test)
    n_classes = proba.shape[1]
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    macro_auroc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    macro_auprc = average_precision_score(y_bin, proba, average="macro")
    return {"macro_auroc": float(macro_auroc), "macro_auprc": float(macro_auprc)}


def evaluate_apnea(
    clf: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    if len(np.unique(y_test)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "note": "single class in test set"}
    proba = clf.predict_proba(X_test)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_test, proba)),
        "auprc": float(average_precision_score(y_test, proba)),
    }


def can_train_binary(y: np.ndarray) -> bool:
    return len(np.unique(y)) >= 2
