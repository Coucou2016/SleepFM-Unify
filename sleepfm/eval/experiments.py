"""Paper-plan experiment helpers: modality ablation, few-shot, transfer."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats

from sleepfm.data.dataset import SleepEpochDataset
from sleepfm.eval.downstream import (
    build_embedding_matrix,
    can_train_binary,
    evaluate_apnea,
    evaluate_sleep_staging,
    train_logistic_regression,
    tune_lr_c,
)
from sleepfm.models.sleepfm import MultiModalSleepFM

# Paper-mode default: ≥10 participant-level repeats with mean±95% CI.
PAPER_FEWSHOT_REPEATS = 10
DEMO_FEWSHOT_REPEATS = 2

MODALITY_COMBOS: Tuple[Tuple[str, ...], ...] = (
    ("bas",),
    ("ecg",),
    ("respiratory",),
    ("bas", "ecg"),
    ("bas", "respiratory"),
    ("ecg", "respiratory"),
    ("bas", "ecg", "respiratory"),
)


def combo_name(mods: Sequence[str]) -> str:
    return "+".join(mods)


def unique_participants(entries: Iterable[dict]) -> List[str]:
    seen = []
    for entry in entries:
        pid = entry.get("participant_id")
        if pid is None:
            continue
        pid = str(pid)
        if pid not in seen:
            seen.append(pid)
    return seen


def sample_participants(participants: Sequence[str], k: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    pids = list(participants)
    rng.shuffle(pids)
    return pids[: min(k, len(pids))]


def probe_split(
    model: MultiModalSleepFM,
    data_dir: str,
    device: torch.device,
    downstream_cfg: dict,
    train_split: str = "train",
    test_split: str = "test",
    valid_split: Optional[str] = "valid",
    batch_size: int = 32,
    keep_modalities: Optional[Sequence[str]] = None,
    train_participant_ids: Optional[Sequence[str]] = None,
    space: str = "downstream",
) -> Dict[str, dict]:
    train_ds = SleepEpochDataset(
        data_dir,
        split=train_split,
        return_labels=True,
        participant_ids=train_participant_ids,
    )
    test_ds = SleepEpochDataset(data_dir, split=test_split, return_labels=True)
    if len(train_ds) == 0 or len(test_ds) == 0:
        return {"error": "empty train or test split"}

    X_tr, y_tr = build_embedding_matrix(
        model, train_ds, device, batch_size, keep_modalities=keep_modalities, space=space
    )
    X_te, y_te = build_embedding_matrix(
        model, test_ds, device, batch_size, keep_modalities=keep_modalities, space=space
    )
    lr_cfg = dict(downstream_cfg)
    if downstream_cfg.get("tune_c_on_valid") and valid_split:
        try:
            valid_ds = SleepEpochDataset(data_dir, split=valid_split, return_labels=True)
            if len(valid_ds) > 0:
                X_va, y_va = build_embedding_matrix(
                    model,
                    valid_ds,
                    device,
                    batch_size,
                    keep_modalities=keep_modalities,
                    space=space,
                )
                lr_cfg["C"] = tune_lr_c(
                    X_tr,
                    y_tr["stage_id"],
                    X_va,
                    y_va["stage_id"],
                    {**downstream_cfg, "task": "staging"},
                )
        except KeyError:
            pass

    out: Dict[str, dict] = {}
    clf = train_logistic_regression(X_tr, y_tr["stage_id"], lr_cfg)
    out["staging"] = evaluate_sleep_staging(clf, X_te, y_te["stage_id"])
    if can_train_binary(y_tr["apnea"]) and can_train_binary(y_te["apnea"]):
        out["apnea"] = evaluate_apnea(
            train_logistic_regression(X_tr, y_tr["apnea"], lr_cfg),
            X_te,
            y_te["apnea"],
        )
    else:
        out["apnea"] = {"note": "skipped (need both classes in train and test)"}
    out["n_train"] = int(X_tr.shape[0])
    out["n_test"] = int(X_te.shape[0])
    out["n_train_participants"] = len(unique_participants(train_ds.entries))
    return out


def modality_ablation_table(
    model: MultiModalSleepFM,
    data_dir: str,
    device: torch.device,
    downstream_cfg: dict,
    batch_size: int = 32,
) -> Dict[str, dict]:
    table = {}
    for combo in MODALITY_COMBOS:
        table[combo_name(combo)] = probe_split(
            model,
            data_dir,
            device,
            downstream_cfg,
            keep_modalities=combo,
            batch_size=batch_size,
        )
    return table


def _metric_ci95(values: Sequence[float], confidence: float = 0.95) -> Dict[str, Any]:
    """Participant-level repeat summary: mean ± 95% CI (Student-t when n≥2)."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "n": 0,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "mean_pm_ci95": "n/a",
        }
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "mean": mean,
            "std": 0.0,
            "n": 1,
            "ci95_low": mean,
            "ci95_high": mean,
            "mean_pm_ci95": f"{mean:.4f} (n=1)",
        }
    std = float(np.std(arr, ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return {
            "mean": mean,
            "std": 0.0,
            "n": n,
            "ci95_low": mean,
            "ci95_high": mean,
            "mean_pm_ci95": f"{mean:.4f}±0.0000",
        }
    sem = std / np.sqrt(n)
    low, high = stats.t.interval(confidence, df=n - 1, loc=mean, scale=sem)
    half = float((high - low) / 2.0)
    return {
        "mean": mean,
        "std": std,
        "n": n,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "mean_pm_ci95": f"{mean:.4f}±{half:.4f}",
    }


def summarize_fewshot_runs(runs: Sequence[dict]) -> Dict[str, Any]:
    """Aggregate staging/apnea metrics across few-shot repeats."""
    staging_auroc = []
    staging_auprc = []
    apnea_auroc = []
    apnea_auprc = []
    for run in runs:
        st = run.get("staging") or {}
        if isinstance(st, dict) and not st.get("skipped"):
            staging_auroc.append(st.get("macro_auroc"))
            staging_auprc.append(st.get("macro_auprc"))
        ap = run.get("apnea") or {}
        if isinstance(ap, dict) and not ap.get("skipped") and "note" not in ap:
            apnea_auroc.append(ap.get("auroc"))
            apnea_auprc.append(ap.get("auprc"))
    return {
        "staging_macro_auroc": _metric_ci95(staging_auroc),
        "staging_macro_auprc": _metric_ci95(staging_auprc),
        "apnea_auroc": _metric_ci95(apnea_auroc),
        "apnea_auprc": _metric_ci95(apnea_auprc),
        "n_repeats": len(runs),
    }


def fewshot_curve(
    model: MultiModalSleepFM,
    data_dir: str,
    device: torch.device,
    downstream_cfg: dict,
    ks: Sequence[int],
    seed: int = 42,
    batch_size: int = 32,
    n_repeats: int = PAPER_FEWSHOT_REPEATS,
    space: str = "downstream",
    summarize: bool = True,
) -> Dict[str, Any]:
    """k-shot linear probe by participant.

    Paper mode defaults to ``PAPER_FEWSHOT_REPEATS`` (≥10) and reports mean±95% CI
    at the participant-sampling level. Demo / CI may pass a smaller ``n_repeats``.
    """
    train_ds = SleepEpochDataset(data_dir, split=downstream_cfg.get("train_split", "train"))
    pool = unique_participants(train_ds.entries)
    curve: Dict[str, Any] = {
        "protocol": {
            "n_repeats": int(n_repeats),
            "space": space,
            "participant_level": True,
            "ci": "95% Student-t on repeat means",
        }
    }
    for k in ks:
        runs = []
        for r in range(n_repeats):
            chosen = sample_participants(pool, int(k), seed + 1000 * r + int(k))
            metrics = probe_split(
                model,
                data_dir,
                device,
                downstream_cfg,
                train_participant_ids=chosen,
                batch_size=batch_size,
                space=space,
            )
            metrics["k"] = int(k)
            metrics["participants"] = chosen
            metrics["repeat"] = int(r)
            runs.append(metrics)
        entry: Dict[str, Any] = {"runs": runs}
        if summarize:
            entry["summary"] = summarize_fewshot_runs(runs)
        curve[str(k)] = entry
    return curve
