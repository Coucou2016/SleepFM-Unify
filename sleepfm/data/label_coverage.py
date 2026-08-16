"""Label-coverage gate for CinC / incomplete AASM + respiratory annotations.

CinC 2018 is primarily an arousal challenge: without sidecar stage/respiratory
XML/CSV/JSON, exporters often write degenerate ``stage_id`` (all Wake) and
``apnea=0``. Claiming staging / SDB metrics in that case is misleading.

Use :func:`compute_label_coverage` after export and :func:`gate_claimed_metrics`
in evaluate / paper-suite paths.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

PathLike = Union[str, Path]

# Minimum unique AASM stage ids (Wake/N1/N2/N3/REM → 0..4) to claim staging.
MIN_STAGING_CLASSES = 2
# Prefer warning when fewer than this many classes appear (full AASM = 5).
FULL_AASM_CLASSES = 5
# Minimum positive apnea epoch rate to claim SDB/apnea metrics.
MIN_APNEA_POSITIVE_RATE = 0.01


@dataclass
class LabelCoverage:
    """Summary of stage / apnea labels across splits."""

    n_epochs: int = 0
    stage_counts: Dict[str, int] = field(default_factory=dict)
    stage_n_unique: int = 0
    apnea_n_positive: int = 0
    apnea_positive_rate: float = 0.0
    has_aasm_staging: bool = False
    has_full_aasm_staging: bool = False
    has_respiratory_events: bool = False
    dataset: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    by_split: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricGate:
    """Which downstream / night metrics may be claimed."""

    claim_staging: bool
    claim_apnea: bool
    claim_night_ahi: bool
    claim_night_staging_kappa: bool
    messages: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _iter_entries(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Iterable[Mapping[str, Any]]:
    for entries in splits.values():
        for e in entries:
            yield e


def _split_stats(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    stages = [int(e.get("stage_id", 0)) for e in entries]
    apneas = [int(e.get("apnea", 0)) for e in entries]
    counts = Counter(stages)
    n = len(entries)
    pos = sum(1 for a in apneas if a)
    return {
        "n_epochs": n,
        "stage_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "stage_n_unique": len(counts),
        "apnea_n_positive": pos,
        "apnea_positive_rate": float(pos / n) if n else 0.0,
    }


def compute_label_coverage(
    data_dir: PathLike,
    *,
    splits: Optional[Sequence[str]] = None,
) -> LabelCoverage:
    """Scan ``index.json`` (or use ``meta.label_coverage`` if already present)."""
    path = Path(data_dir) / "index.json"
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    meta = payload.get("meta") or {}
    all_splits = payload.get("splits") or {}
    use_splits = list(splits) if splits is not None else list(all_splits.keys())
    filtered = {k: all_splits.get(k, []) for k in use_splits}

    by_split = {k: _split_stats(v) for k, v in filtered.items()}
    entries = list(_iter_entries(filtered))
    overall = _split_stats(entries)

    dataset = meta.get("dataset")
    cov = LabelCoverage(
        n_epochs=overall["n_epochs"],
        stage_counts=overall["stage_counts"],
        stage_n_unique=overall["stage_n_unique"],
        apnea_n_positive=overall["apnea_n_positive"],
        apnea_positive_rate=overall["apnea_positive_rate"],
        dataset=str(dataset) if dataset is not None else None,
        by_split=by_split,
    )
    cov.has_aasm_staging = cov.stage_n_unique >= MIN_STAGING_CLASSES
    cov.has_full_aasm_staging = cov.stage_n_unique >= FULL_AASM_CLASSES
    cov.has_respiratory_events = cov.apnea_positive_rate >= MIN_APNEA_POSITIVE_RATE

    if not cov.has_aasm_staging:
        cov.notes.append(
            "Staging labels look degenerate (fewer than "
            f"{MIN_STAGING_CLASSES} unique stage_id). "
            "CinC without AASM sidecar often defaults to Wake-only — "
            "do not claim sleep-staging metrics."
        )
    elif not cov.has_full_aasm_staging:
        cov.notes.append(
            f"Only {cov.stage_n_unique}/{FULL_AASM_CLASSES} AASM stage classes present; "
            "staging metrics are partial (not full Wake/N1/N2/N3/REM coverage)."
        )
    if not cov.has_respiratory_events:
        cov.notes.append(
            "Respiratory/apnea positives below "
            f"{MIN_APNEA_POSITIVE_RATE:.0%} of epochs — "
            "do not claim SDB/AHI metrics (common for raw CinC arousal-only)."
        )
    ds = (cov.dataset or "").lower()
    if "cinc" in ds and (not cov.has_aasm_staging or not cov.has_respiratory_events):
        cov.notes.append(
            "Dataset tagged as CinC: expect arousal-focused labels unless "
            "XML/CSV/JSON stage+respiratory annotations were provided at export."
        )
    return cov


def gate_claimed_metrics(
    coverage: LabelCoverage,
    *,
    require_full_aasm: bool = False,
) -> MetricGate:
    """Decide which metrics may be reported given label coverage."""
    claim_staging = coverage.has_full_aasm_staging if require_full_aasm else coverage.has_aasm_staging
    claim_apnea = coverage.has_respiratory_events
    messages = list(coverage.notes)
    skipped: List[str] = []
    if not claim_staging:
        skipped.append("staging")
        messages.append("GATE: skip/claim-blocked sleep staging metrics.")
    if not claim_apnea:
        skipped.append("apnea")
        skipped.append("night_ahi")
        messages.append("GATE: skip/claim-blocked apnea/SDB and night AHI metrics.")
    claim_kappa = claim_staging
    if not claim_kappa:
        skipped.append("night_staging_kappa")
    return MetricGate(
        claim_staging=claim_staging,
        claim_apnea=claim_apnea,
        claim_night_ahi=claim_apnea,
        claim_night_staging_kappa=claim_kappa,
        messages=messages,
        skipped=skipped,
    )


def apply_metric_gate(
    metrics: Dict[str, Any],
    gate: MetricGate,
    *,
    staging_keys: Sequence[str] = ("staging", "staging_epoch_kappa"),
    apnea_keys: Sequence[str] = ("apnea",),
    night_ahi_keys: Sequence[str] = ("ahi_bin", "ahi", "night_ahi"),
) -> Dict[str, Any]:
    """Replace blocked metric blobs with an explicit skipped marker."""
    out = dict(metrics)
    out["label_gate"] = gate.to_dict()

    def _block(key: str, reason: str) -> None:
        if key not in out:
            return
        out[key] = {
            "skipped": True,
            "reason": reason,
            "prior_value_removed": True,
        }

    if not gate.claim_staging:
        for k in staging_keys:
            _block(k, "insufficient AASM staging label coverage")
    if not gate.claim_apnea:
        for k in apnea_keys:
            _block(k, "insufficient respiratory/apnea label coverage")
    if not gate.claim_night_ahi:
        for k in night_ahi_keys:
            _block(k, "insufficient respiratory/apnea label coverage")
    if not gate.claim_night_staging_kappa:
        _block("staging_epoch_kappa", "insufficient AASM staging label coverage")
    return out


def coverage_from_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    dataset: Optional[str] = None,
) -> LabelCoverage:
    """Compute coverage from an in-memory epoch list (export path)."""
    stats = _split_stats(entries)
    cov = LabelCoverage(
        n_epochs=stats["n_epochs"],
        stage_counts=stats["stage_counts"],
        stage_n_unique=stats["stage_n_unique"],
        apnea_n_positive=stats["apnea_n_positive"],
        apnea_positive_rate=stats["apnea_positive_rate"],
        dataset=dataset,
    )
    cov.has_aasm_staging = cov.stage_n_unique >= MIN_STAGING_CLASSES
    cov.has_full_aasm_staging = cov.stage_n_unique >= FULL_AASM_CLASSES
    cov.has_respiratory_events = cov.apnea_positive_rate >= MIN_APNEA_POSITIVE_RATE
    if not cov.has_aasm_staging:
        cov.notes.append(
            "Exported staging looks degenerate — add AASM annotations before claiming staging metrics."
        )
    if not cov.has_respiratory_events:
        cov.notes.append(
            "Exported apnea rate near zero — add respiratory events before claiming SDB metrics."
        )
    return cov
