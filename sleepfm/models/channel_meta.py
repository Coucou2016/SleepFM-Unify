"""Channel-count meta checks (paper 10/2/7 vs official CinC demo 5/1/3).

Silent mismatches previously skipped ``stage1`` tensors when loading official
weights into a paper montage model. Callers should use
:func:`assert_channels_compatible` / :func:`check_channel_meta` and pass
``allow_mismatch=True`` only when the override is intentional.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

# Paper clinic montage (docs/DATA_SCHEMA.md, configs/default.yaml).
PAPER_CHANNELS: Dict[str, int] = {"bas": 10, "ecg": 2, "respiratory": 7}

# Official rthapa84 CinC demo checkpoint (Sleep_Stages / EKG / Respiratory).
OFFICIAL_CINC_CHANNELS: Dict[str, int] = {"bas": 5, "ecg": 1, "respiratory": 3}

MODALITY_ORDER = ("bas", "ecg", "respiratory")

PathLike = Union[str, Path]


def format_channel_triplet(channels: Mapping[str, int]) -> str:
    """Human-readable bas/ecg/respiratory counts, e.g. ``10/2/7``."""
    return "/".join(str(int(channels.get(m, "?"))) for m in MODALITY_ORDER)


def channels_equal(a: Mapping[str, int], b: Mapping[str, int]) -> bool:
    return all(int(a.get(m, -1)) == int(b.get(m, -1)) for m in MODALITY_ORDER)


def describe_montage(channels: Mapping[str, int]) -> str:
    trip = format_channel_triplet(channels)
    if channels_equal(channels, PAPER_CHANNELS):
        return f"{trip} (paper clinic)"
    if channels_equal(channels, OFFICIAL_CINC_CHANNELS):
        return f"{trip} (official CinC demo)"
    return f"{trip} (custom)"


@dataclass
class ChannelCheckReport:
    """Result of comparing model / checkpoint / data channel counts."""

    ok: bool
    model_channels: Dict[str, int] = field(default_factory=dict)
    data_channels: Optional[Dict[str, int]] = None
    expected_channels: Optional[Dict[str, int]] = None
    messages: List[str] = field(default_factory=list)
    mismatches: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join(self.messages + self.mismatches)


def load_data_channels(data_dir: PathLike) -> Dict[str, int]:
    """Read ``meta.channels`` from an exported ``index.json``."""
    path = Path(data_dir) / "index.json"
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    meta = payload.get("meta") or {}
    ch = meta.get("channels")
    if not isinstance(ch, Mapping):
        raise KeyError(f"index.json meta.channels missing in {path}")
    return {m: int(ch[m]) for m in MODALITY_ORDER if m in ch}


def check_channel_meta(
    model_channels: Mapping[str, int],
    *,
    data_channels: Optional[Mapping[str, int]] = None,
    expected_channels: Optional[Mapping[str, int]] = None,
    allow_mismatch: bool = False,
    context: str = "",
) -> ChannelCheckReport:
    """Compare model channels to data and/or an expected montage.

    Returns ``ok=False`` on mismatch unless ``allow_mismatch=True`` (still
    records loud WARNING messages).
    """
    model = {m: int(model_channels[m]) for m in MODALITY_ORDER if m in model_channels}
    report = ChannelCheckReport(ok=True, model_channels=dict(model))
    prefix = f"{context}: " if context else ""

    report.messages.append(
        f"{prefix}model channels={describe_montage(model)}"
    )

    targets: List[tuple[str, Mapping[str, int]]] = []
    if data_channels is not None:
        data = {m: int(data_channels[m]) for m in MODALITY_ORDER if m in data_channels}
        report.data_channels = dict(data)
        targets.append(("data meta", data))
    if expected_channels is not None:
        exp = {m: int(expected_channels[m]) for m in MODALITY_ORDER if m in expected_channels}
        report.expected_channels = dict(exp)
        targets.append(("expected", exp))

    for label, other in targets:
        if channels_equal(model, other):
            report.messages.append(
                f"{prefix}OK vs {label} {describe_montage(other)}"
            )
            continue
        msg = (
            f"{prefix}CHANNEL MISMATCH vs {label}: "
            f"model {describe_montage(model)} != {label} {describe_montage(other)}. "
            f"Official CinC demo weights are typically "
            f"{format_channel_triplet(OFFICIAL_CINC_CHANNELS)}; "
            f"paper/export schema is {format_channel_triplet(PAPER_CHANNELS)}. "
            f"Rebuild/convert with matching --channels, or pass "
            f"--allow-channel-mismatch only if intentional."
        )
        report.mismatches.append(msg)
        report.ok = False

    if report.mismatches and allow_mismatch:
        report.ok = True
        for m in report.mismatches:
            report.messages.append("WARNING (allowed): " + m)
        report.mismatches = []
    elif report.mismatches:
        for m in report.mismatches:
            report.messages.append("ERROR: " + m)

    return report


def assert_channels_compatible(
    model_channels: Mapping[str, int],
    *,
    data_dir: Optional[PathLike] = None,
    data_channels: Optional[Mapping[str, int]] = None,
    expected_channels: Optional[Mapping[str, int]] = None,
    allow_mismatch: bool = False,
    context: str = "",
) -> ChannelCheckReport:
    """Fail-fast channel check for evaluate / suite paths."""
    if data_channels is None and data_dir is not None:
        data_channels = load_data_channels(data_dir)
    report = check_channel_meta(
        model_channels,
        data_channels=data_channels,
        expected_channels=expected_channels,
        allow_mismatch=allow_mismatch,
        context=context,
    )
    if not report.ok:
        raise RuntimeError(report.summary())
    for line in report.messages:
        if line.startswith("WARNING"):
            warnings.warn(line, UserWarning, stacklevel=2)
    return report


def warn_official_vs_paper(
    inferred: Mapping[str, int],
    requested: Optional[Mapping[str, int]] = None,
) -> List[str]:
    """Loud messages when official CinC 5/1/3 meets paper 10/2/7."""
    messages: List[str] = []
    inf = {m: int(inferred[m]) for m in MODALITY_ORDER if m in inferred}
    if not inf:
        return messages
    if channels_equal(inf, OFFICIAL_CINC_CHANNELS):
        messages.append(
            "WARNING: checkpoint looks like official CinC demo "
            f"{format_channel_triplet(OFFICIAL_CINC_CHANNELS)}, not paper "
            f"{format_channel_triplet(PAPER_CHANNELS)}. "
            "Do not evaluate on 10/2/7 exports without remapping or a matching convert."
        )
    if requested is not None and not channels_equal(inf, requested):
        messages.append(
            "WARNING: inferred checkpoint channels "
            f"{describe_montage(inf)} != requested "
            f"{describe_montage(requested)}. First-layer (stage1) weights will be "
            "skipped unless you rebuild the model with inferred channels."
        )
    return messages


def channels_from_checkpoint_payload(ckpt: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    """Extract channels from a native or converted checkpoint dict."""
    ch = ckpt.get("channels")
    if isinstance(ch, Mapping) and all(m in ch for m in MODALITY_ORDER):
        return {m: int(ch[m]) for m in MODALITY_ORDER}
    inferred = ckpt.get("inferred_channels")
    if isinstance(inferred, Mapping) and inferred:
        return {m: int(inferred[m]) for m in MODALITY_ORDER if m in inferred}
    return None
