"""Night-level labels and linear probes.

Night ``ahi`` / ``ahi_bin`` use **apnea-epoch rate** placeholders (not clinical AHI).
Sleep-efficiency is stage-based (non-Wake fraction), also a coarse placeholder.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, r2_score, roc_auc_score
from torch.utils.data import DataLoader

from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal
from sleepfm.data.night_dataset import group_entries_by_night, night_summary_from_entries
from sleepfm.eval.downstream import train_logistic_regression
from sleepfm.models.sleepfm import MultiModalSleepFM
from sleepfm.models.temporal import NightTemporalEncoder, contextualize_sequence


def epoch_seconds_from_meta(meta: dict) -> float:
    if meta.get("clip_seconds"):
        return float(meta["clip_seconds"])
    clip = float(meta.get("clip_length") or 7680)
    fs = float(meta.get("sample_rate") or 256)
    return clip / fs


def _collate_with_labels(batch):
    signals = collate_multimodal(batch)
    labels = {
        "stage_id": torch.tensor([b["labels"]["stage_id"] for b in batch]),
        "apnea": torch.tensor([b["labels"]["apnea"] for b in batch]),
    }
    return signals, labels


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def encode_epoch_features(
    model: MultiModalSleepFM,
    dataset: SleepEpochDataset,
    device: torch.device,
    batch_size: int = 32,
    reduce: str = "concat",
    space: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Per-epoch embeddings. ``concat`` is the downstream default; ``mean`` matches temporal training."""
    model.eval()
    if space is None:
        if reduce == "mean":
            space = "shared" if getattr(model, "unify", False) else "downstream"
        else:
            space = "downstream"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_with_labels,
    )
    labels_stage, labels_apnea = [], []
    emb_chunks = []
    with torch.no_grad():
        for batch, lab in loader:
            batch = _batch_to_device(batch, device)
            z = model.encode(batch, space=space)
            parts = [z[m] for m in model.MODALITY_ORDER if m in z]
            if not parts:
                continue
            if reduce == "mean":
                combined = torch.stack(parts, dim=0).mean(dim=0)
            else:
                combined = torch.cat(parts, dim=-1)
            emb_chunks.append(combined.cpu().numpy())
            labels_stage.extend(lab["stage_id"].tolist())
            labels_apnea.extend(lab["apnea"].tolist())
    if not emb_chunks:
        return np.zeros((0, 0), dtype=np.float32), {
            "stage_id": np.zeros((0,), dtype=np.int64),
            "apnea": np.zeros((0,), dtype=np.int64),
        }
    X = np.concatenate(emb_chunks, axis=0)
    y = {
        "stage_id": np.array(labels_stage, dtype=np.int64),
        "apnea": np.array(labels_apnea, dtype=np.int64),
    }
    return X, y


def _night_items(
    entries: Sequence[dict],
    epoch_seconds: float,
) -> List[dict]:
    groups = group_entries_by_night(list(entries))
    path_to_row = {e["path"]: i for i, e in enumerate(entries)}
    items = []
    for key, recs in groups.items():
        rows = [path_to_row[e["path"]] for e in recs if e["path"] in path_to_row]
        if not rows:
            continue
        items.append(
            {
                "key": f"{key[0]}::{key[1]}",
                "rows": rows,
                "summary": night_summary_from_entries(recs, epoch_seconds=epoch_seconds),
            }
        )
    return items


def night_eval_pack(
    model: MultiModalSleepFM,
    data_dir: str,
    split: str,
    device: torch.device,
    batch_size: int = 32,
    space: str = "downstream",
    temporal_encoder: Optional[NightTemporalEncoder] = None,
) -> dict:
    """Night vectors + epoch features. Uses the temporal head when provided."""
    ds = SleepEpochDataset(data_dir, split=split, return_labels=True)
    epoch_seconds = epoch_seconds_from_meta(ds.meta)
    used_temporal = temporal_encoder is not None
    if used_temporal:
        X_ep, y_ep = encode_epoch_features(
            model, ds, device, batch_size=batch_size, reduce="mean"
        )
        d_model = int(getattr(temporal_encoder, "d_model", X_ep.shape[1] if X_ep.size else 0))
        if X_ep.size and int(X_ep.shape[1]) != d_model:
            raise ValueError(
                f"Temporal head d_model={d_model} but epoch features have dim {X_ep.shape[1]}. "
                "Shared-mean embeddings must match the trained temporal encoder."
            )
        items = _night_items(ds.entries, epoch_seconds)
        ctx_epoch = np.array(X_ep, copy=True)
        night_vecs: List[np.ndarray] = []
        temporal_encoder.eval()
        with torch.no_grad():
            for item in items:
                rows = item["rows"]
                seq = torch.from_numpy(np.asarray(X_ep[rows], dtype=np.float32)).to(device)
                ctx = contextualize_sequence(temporal_encoder, seq)
                ctx_np = ctx.detach().cpu().numpy()
                for i, r in enumerate(rows):
                    ctx_epoch[r] = ctx_np[i]
                night_vecs.append(ctx_np.mean(axis=0))
        dim = int(X_ep.shape[1]) if X_ep.ndim == 2 and X_ep.size else d_model
        X_nights = np.stack(night_vecs, axis=0) if night_vecs else np.zeros((0, dim))
        epoch_features = ctx_epoch
    else:
        X_ep, y_ep = encode_epoch_features(
            model, ds, device, batch_size=batch_size, reduce="concat", space=space
        )
        items = _night_items(ds.entries, epoch_seconds)
        night_vecs = [X_ep[it["rows"]].mean(axis=0) for it in items]
        dim = int(X_ep.shape[1]) if X_ep.ndim == 2 and X_ep.size else 0
        X_nights = np.stack(night_vecs, axis=0) if night_vecs else np.zeros((0, dim))
        epoch_features = X_ep
    return {
        "X": X_nights,
        "summaries": [it["summary"] for it in items],
        "keys": [it["key"] for it in items],
        "X_epochs": epoch_features,
        "y_stage": y_ep["stage_id"],
        "used_temporal": used_temporal,
    }


def night_embedding_table(
    model: MultiModalSleepFM,
    data_dir: str,
    split: str,
    device: torch.device,
    batch_size: int = 32,
    space: str = "downstream",
    temporal_encoder: Optional[NightTemporalEncoder] = None,
) -> Tuple[np.ndarray, List[dict], List[str]]:
    pack = night_eval_pack(
        model,
        data_dir,
        split,
        device,
        batch_size=batch_size,
        space=space,
        temporal_encoder=temporal_encoder,
    )
    return pack["X"], pack["summaries"], pack["keys"]


def epoch_sequence_kappa(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Linear probe on epoch sequences; Cohen's κ when staging labels exist."""
    if X_train is None or len(X_train) < 2 or len(np.unique(y_train)) < 2:
        return {"note": "need >=2 train epochs and >=2 stage classes"}
    if X_test is None or len(X_test) == 0:
        return {"note": "empty test epoch sequence"}
    try:
        clf = train_logistic_regression(
            X_train,
            y_train,
            {"max_iter": 2000, "class_weight": "balanced", "solver": "lbfgs"},
        )
        pred = clf.predict(X_test)
        out = {
            "cohen_kappa": float(cohen_kappa_score(y_test, pred)),
            "accuracy": float(accuracy_score(y_test, pred)),
        }
        n_cls = int(max(len(np.unique(y_train)), len(np.unique(y_test))))
        if n_cls >= 3:
            out["cohen_kappa_linear"] = float(
                cohen_kappa_score(y_test, pred, weights="linear")
            )
        return out
    except Exception as exc:
        return {"error": str(exc)}


def probe_night_tasks(
    X_train: np.ndarray,
    summaries_train: Sequence[dict],
    X_test: np.ndarray,
    summaries_test: Sequence[dict],
) -> Dict[str, dict]:
    y_bin_tr = np.array([s["ahi_bin"] for s in summaries_train], dtype=np.int64)
    y_bin_te = np.array([s["ahi_bin"] for s in summaries_test], dtype=np.int64)
    y_se_tr = np.array([s["sleep_efficiency"] for s in summaries_train], dtype=np.float64)
    y_se_te = np.array([s["sleep_efficiency"] for s in summaries_test], dtype=np.float64)
    out: Dict[str, dict] = {
        "n_train_nights": int(len(summaries_train)),
        "n_test_nights": int(len(summaries_test)),
        "ahi_note": (
            "ahi_bin uses apnea-positive epochs/hour cut-points as a placeholder; "
            "not clinical AASM AHI"
        ),
    }
    if len(np.unique(y_bin_tr)) >= 2 and len(X_train) >= 2:
        try:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X_train, y_bin_tr)
            pred = clf.predict(X_test)
            out["ahi_bin"] = {
                "accuracy": float(accuracy_score(y_bin_te, pred)),
                "n_classes_train": int(len(np.unique(y_bin_tr))),
                "label": "apnea_epoch_rate_bin_placeholder",
            }
            if len(np.unique(y_bin_te)) >= 2 and hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_test)
                if proba.shape[1] == 2:
                    out["ahi_bin"]["auroc"] = float(roc_auc_score(y_bin_te, proba[:, 1]))
        except Exception as exc:
            out["ahi_bin"] = {"error": str(exc)}
    else:
        out["ahi_bin"] = {
            "note": "need >=2 apnea_epoch_rate bins in train nights (placeholder labels)"
        }

    if len(X_train) >= 2:
        reg = LinearRegression()
        reg.fit(X_train, y_se_tr)
        pred_se = reg.predict(X_test)
        out["sleep_efficiency"] = {
            "r2": float(r2_score(y_se_te, pred_se)) if len(y_se_te) > 1 else float("nan"),
            "mae": float(np.mean(np.abs(pred_se - y_se_te))),
        }
    else:
        out["sleep_efficiency"] = {"note": "not enough nights"}
    return out
