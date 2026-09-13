"""Downstream evaluation with logistic regression on frozen embeddings (paper Sec 3.2)."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
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
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) if torch.is_tensor(vv) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _apply_present_mask_to_emb(
    z: Dict[str, torch.Tensor],
    batch: dict,
    modality_order: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Zero out embeddings for modalities marked absent in ``present_mask``."""
    mask = batch.get("present_mask")
    if mask is None:
        return z
    out = {}
    for i, m in enumerate(modality_order):
        if m not in z:
            continue
        if i < mask.size(-1):
            out[m] = z[m] * mask[:, i : i + 1].to(dtype=z[m].dtype)
        else:
            out[m] = z[m]
    return out


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
            z = _apply_present_mask_to_emb(z, batch, model.MODALITY_ORDER)
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
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    downstream_cfg: dict,
    c_grid: Optional[list] = None,
) -> float:
    """Select L2 C by scoring on validation after fitting on train (no resubstitution)."""
    if c_grid is None:
        c_grid = downstream_cfg.get("c_grid", [0.01, 0.1, 1.0, 10.0, 100.0])
    best_c = float(downstream_cfg.get("C", 1.0))
    best_score = -1.0
    task = downstream_cfg.get("task", "staging")
    for c in c_grid:
        cfg = {**downstream_cfg, "C": c}
        clf = train_logistic_regression(X_train, y_train, cfg)
        if task == "apnea":
            score = evaluate_apnea(clf, X_valid, y_valid).get("auroc", float("nan"))
        else:
            score = evaluate_sleep_staging(clf, X_valid, y_valid).get(
                "macro_auroc", float("nan")
            )
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_c = float(c)
    return best_c


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
    """Macro AUROC/AUPRC aligned to ``clf.classes_``; also Accuracy / Macro F1 / κ."""
    proba = clf.predict_proba(X_test)
    classes = np.asarray(clf.classes_)
    pred = clf.predict(X_test)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_test, pred)),
    }

    # Align one-hot columns to the classifier's class order (not 0..n-1).
    y_bin = label_binarize(y_test, classes=classes)
    if y_bin.ndim == 1 or y_bin.shape[1] == 1:
        # Binary case: label_binarize may return a single column.
        if len(classes) == 2:
            y_bin = np.hstack([1 - y_bin.reshape(-1, 1), y_bin.reshape(-1, 1)])
        else:
            y_bin = y_bin.reshape(-1, 1)

    # Drop columns for classes that never appear in y_test (roc_auc undefined).
    if y_bin.shape[1] == proba.shape[1] and y_bin.shape[1] >= 2:
        present = y_bin.sum(axis=0) > 0
        if int(present.sum()) >= 2:
            try:
                out["macro_auroc"] = float(
                    roc_auc_score(
                        y_bin[:, present],
                        proba[:, present],
                        average="macro",
                        multi_class="ovr",
                    )
                )
            except ValueError:
                out["macro_auroc"] = float("nan")
            try:
                out["macro_auprc"] = float(
                    average_precision_score(y_bin[:, present], proba[:, present], average="macro")
                )
            except ValueError:
                out["macro_auprc"] = float("nan")
        else:
            out["macro_auroc"] = float("nan")
            out["macro_auprc"] = float("nan")
            out["note"] = "fewer than 2 classes present in test for AUROC"
    else:
        out["macro_auroc"] = float("nan")
        out["macro_auprc"] = float("nan")
        out["note"] = "class/proba shape mismatch or single-class classifier"
    return out


def evaluate_apnea(
    clf: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    if len(np.unique(y_test)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "note": "single class in test set"}
    # Use positive class column from clf.classes_ when available.
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 in classes:
        pos_idx = classes.index(1)
    else:
        pos_idx = min(1, len(classes) - 1)
    proba = clf.predict_proba(X_test)[:, pos_idx]
    return {
        "auroc": float(roc_auc_score(y_test, proba)),
        "auprc": float(average_precision_score(y_test, proba)),
    }


def can_train_binary(y: np.ndarray) -> bool:
    return len(np.unique(y)) >= 2
