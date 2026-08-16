"""Load official rthapa84/sleepfm-codebase checkpoints into MultiModalSleepFM.

Official ``best.pt`` stores three separate EffNet state dicts plus temperature::

    respiratory_state_dict / sleep_stages_state_dict / ekg_state_dict
    temperature

Our package nests the same EffNet under ``encoders.{bas,ecg,respiratory}``.
Aliases ``resp_state_dict`` / ``sleep_state_dict`` are accepted for older notes.

The bundled CinC demo checkpoint uses **5 / 1 / 3** input channels (Sleep_Stages /
EKG / Respiratory in the official config), not the paper clinic montage
``10 / 2 / 7``. Shape mismatches on ``stage1`` (and any other tensors) are
reported and skipped unless ``strict=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from sleepfm.models.channel_meta import (
    OFFICIAL_CINC_CHANNELS,
    PAPER_CHANNELS,
    describe_montage,
    format_channel_triplet,
    warn_official_vs_paper,
)

# Official modality blob key → our encoder name. First match wins.
OFFICIAL_MODALITY_KEYS: Dict[str, Tuple[str, ...]] = {
    "bas": ("sleep_stages_state_dict", "sleep_state_dict", "bas_state_dict"),
    "ecg": ("ekg_state_dict", "ecg_state_dict"),
    "respiratory": ("respiratory_state_dict", "resp_state_dict"),
}

# Human-readable notes for docs / CLI.
OFFICIAL_CHANNEL_HINT = {
    "bas": "official Sleep_Stages (often 5 leads on CinC demo)",
    "ecg": "official EKG (often 1 lead on CinC demo)",
    "respiratory": "official Respiratory (often 3 leads on CinC demo)",
}

PathLike = Union[str, Path]


@dataclass
class AdapterReport:
    """What mapped, what was skipped, and inferred channel counts."""

    source_keys: Dict[str, str] = field(default_factory=dict)
    loaded: List[str] = field(default_factory=list)
    skipped_shape: List[str] = field(default_factory=list)
    skipped_missing_in_model: List[str] = field(default_factory=list)
    unused_in_checkpoint: List[str] = field(default_factory=list)
    inferred_channels: Dict[str, int] = field(default_factory=dict)
    temperature: Optional[float] = None
    messages: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.loaded) and not any(
            m.startswith("ERROR:") for m in self.messages
        )

    def summary(self) -> str:
        lines = list(self.messages)
        lines.append(f"loaded={len(self.loaded)} skipped_shape={len(self.skipped_shape)}")
        if self.inferred_channels:
            lines.append(f"inferred_channels={self.inferred_channels}")
        if self.temperature is not None:
            lines.append(f"temperature={self.temperature}")
        return "\n".join(lines)


def is_official_checkpoint(obj: Any) -> bool:
    """True if ``obj`` looks like an official SleepFM checkpoint dict."""
    if not isinstance(obj, Mapping):
        return False
    if "model_state_dict" in obj and any(
        k.startswith("encoders.") for k in obj.get("model_state_dict", {})
    ):
        return False
    for aliases in OFFICIAL_MODALITY_KEYS.values():
        if any(k in obj for k in aliases):
            return True
    return False


def strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove ``module.`` from DataParallel keys."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module.") :]] = v
        else:
            out[k] = v
    return out


def _pick_modality_blob(ckpt: Mapping[str, Any], our_name: str) -> Optional[Tuple[str, Dict[str, torch.Tensor]]]:
    for key in OFFICIAL_MODALITY_KEYS[our_name]:
        if key in ckpt and isinstance(ckpt[key], Mapping):
            return key, strip_module_prefix(ckpt[key])
    return None


def infer_in_channels(state: Mapping[str, torch.Tensor]) -> Optional[int]:
    w = state.get("stage1.weight")
    if w is None or not torch.is_tensor(w) or w.ndim < 2:
        return None
    return int(w.shape[1])


def convert_official_to_ours(
    ckpt: Mapping[str, Any],
    *,
    target_state: Optional[Mapping[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], AdapterReport]:
    """Build a nested ``encoders.*`` state dict from an official checkpoint.

    When ``target_state`` is provided, only tensors whose shapes match the
    target are included (others go to ``skipped_shape``).
    """
    report = AdapterReport()
    if not is_official_checkpoint(ckpt):
        report.messages.append(
            "ERROR: not an official SleepFM checkpoint "
            "(expected respiratory_state_dict / sleep_stages_state_dict / ekg_state_dict)."
        )
        return {}, report

    converted: Dict[str, torch.Tensor] = {}
    for our_name in ("bas", "ecg", "respiratory"):
        picked = _pick_modality_blob(ckpt, our_name)
        if picked is None:
            report.messages.append(
                f"WARNING: no official blob for modality '{our_name}' "
                f"(tried {OFFICIAL_MODALITY_KEYS[our_name]})."
            )
            continue
        src_key, blob = picked
        report.source_keys[our_name] = src_key
        ch = infer_in_channels(blob)
        if ch is not None:
            report.inferred_channels[our_name] = ch
            report.messages.append(
                f"MAP {src_key} → encoders.{our_name}.* "
                f"({OFFICIAL_CHANNEL_HINT[our_name]}; in_channels={ch})"
            )
        else:
            report.messages.append(f"MAP {src_key} → encoders.{our_name}.*")

        for param_name, tensor in blob.items():
            dst = f"encoders.{our_name}.{param_name}"
            if target_state is not None and dst not in target_state:
                report.skipped_missing_in_model.append(dst)
                continue
            if target_state is not None:
                want = target_state[dst]
                if tuple(want.shape) != tuple(tensor.shape):
                    report.skipped_shape.append(
                        f"{dst}: ckpt{tuple(tensor.shape)} vs model{tuple(want.shape)}"
                    )
                    continue
            converted[dst] = tensor
            report.loaded.append(dst)

    if "temperature" in ckpt:
        try:
            report.temperature = float(ckpt["temperature"])
        except (TypeError, ValueError):
            report.messages.append("WARNING: could not parse temperature")

    if target_state is not None:
        for k in target_state:
            if k.startswith("encoders.") and k not in converted:
                if not any(k == s.split(":")[0] for s in report.skipped_shape):
                    if k not in report.skipped_missing_in_model:
                        report.unused_in_checkpoint.append(k)

    if report.inferred_channels:
        report.messages.extend(warn_official_vs_paper(report.inferred_channels))

    if report.skipped_shape:
        trip_inf = (
            format_channel_triplet(report.inferred_channels)
            if report.inferred_channels
            else "unknown"
        )
        report.messages.append(
            f"ERROR: CHANNEL MISMATCH — skipped {len(report.skipped_shape)} tensors "
            f"(checkpoint ~{trip_inf} vs model/target; paper is "
            f"{format_channel_triplet(PAPER_CHANNELS)}, official CinC demo is "
            f"{format_channel_triplet(OFFICIAL_CINC_CHANNELS)}). "
            "Rebuild with inferred channels, convert with matching --channels, "
            "or pass allow_channel_mismatch / --allow-channel-mismatch only if intentional:"
        )
        for s in report.skipped_shape[:20]:
            report.messages.append(f"  {s}")
        if len(report.skipped_shape) > 20:
            report.messages.append(f"  ... and {len(report.skipped_shape) - 20} more")

    if not report.loaded:
        report.messages.append(
            "ERROR: no tensors mapped. Channel counts likely incompatible; "
            "construct the model with official CinC channels "
            f"{OFFICIAL_CINC_CHANNELS} or matching inferred_channels."
        )
    return converted, report


def load_official_checkpoint(
    path: PathLike,
    map_location: str = "cpu",
) -> Tuple[dict, AdapterReport]:
    """Load a ``.pt`` file and convert to our nested state dict (no model yet)."""
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not is_official_checkpoint(ckpt):
        report = AdapterReport()
        report.messages.append(
            f"ERROR: {path} is not an official SleepFM checkpoint "
            "(no respiratory/sleep_stages/ekg state_dict keys)."
        )
        return {}, report
    return convert_official_to_ours(ckpt)


def apply_official_weights(
    model: nn.Module,
    ckpt: Mapping[str, Any],
    *,
    strict: bool = False,
    allow_channel_mismatch: bool = False,
    require_channel_match: bool = False,
) -> AdapterReport:
    """Load mapped official weights into an existing MultiModalSleepFM.

    Shape mismatches (typical official CinC ``5/1/3`` vs paper ``10/2/7``) are
    always reported as ERROR-level CHANNEL MISMATCH messages. Loading still
    proceeds with matching tensors unless ``strict`` / ``require_channel_match``
    is set (fail-fast) — pass ``allow_channel_mismatch=True`` to acknowledge
    intentional partial loads.
    """
    target = model.state_dict()
    converted, report = convert_official_to_ours(ckpt, target_state=target)
    model_ch = getattr(model, "channels", None)
    if isinstance(model_ch, Mapping) and report.inferred_channels:
        report.messages.extend(
            warn_official_vs_paper(report.inferred_channels, requested=model_ch)
        )
        report.messages.append(
            f"INFO: model montage={describe_montage(model_ch)}; "
            f"checkpoint inferred={describe_montage(report.inferred_channels)}"
        )

    stage1_skipped = [s for s in report.skipped_shape if "stage1" in s]
    fail_fast = (strict or require_channel_match) and not allow_channel_mismatch
    if stage1_skipped and fail_fast:
        raise RuntimeError(
            "channel check failed (official vs model montage):\n" + report.summary()
        )

    if not converted:
        if strict:
            raise RuntimeError(report.summary())
        return report

    missing_unexpected = model.load_state_dict(converted, strict=False)
    # load_state_dict returns MissingKeys / UnexpectedKeys namedtuples
    missing = list(getattr(missing_unexpected, "missing_keys", []) or [])
    unexpected = list(getattr(missing_unexpected, "unexpected_keys", []) or [])
    if unexpected:
        report.messages.append(f"WARNING: unexpected keys after load: {unexpected[:5]}")
    if missing:
        report.unused_in_checkpoint.extend(missing)
        report.messages.append(
            f"INFO: model keys left at init ({len(missing)}), including first-layer "
            "or Unify heads if present."
        )

    temp = report.temperature
    if temp is not None and hasattr(model, "temperature"):
        with torch.no_grad():
            model.temperature.fill_(temp)

    if strict and report.skipped_shape and not allow_channel_mismatch:
        raise RuntimeError(
            "strict=True but shape mismatches remain:\n" + report.summary()
        )
    if strict and not report.ok and not allow_channel_mismatch:
        raise RuntimeError(report.summary())
    return report


def build_model_from_official(
    path: PathLike,
    *,
    channels: Optional[Dict[str, int]] = None,
    embedding_dim: int = 512,
    device: str = "cpu",
    strict: bool = False,
    unify: bool = False,
    allow_channel_mismatch: bool = False,
):
    """Construct MultiModalSleepFM and load official weights.

    If ``channels`` is omitted, uses inferred stage1 widths from the checkpoint
    (CinC demo → typically ``bas=5, ecg=1, respiratory=3``).
    Explicit paper ``10/2/7`` against a ``5/1/3`` checkpoint fails unless
    ``allow_channel_mismatch=True`` (or ``strict`` path with matching rebuild).
    """
    from sleepfm.models.sleepfm import MultiModalSleepFM

    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not is_official_checkpoint(ckpt):
        raise ValueError(
            f"{path} is not an official SleepFM checkpoint. "
            "Expected keys like respiratory_state_dict / sleep_stages_state_dict / ekg_state_dict."
        )

    _, probe = convert_official_to_ours(ckpt)
    inferred = dict(probe.inferred_channels)
    ch = channels or dict(inferred)
    if not ch or set(ch) < {"bas", "ecg", "respiratory"}:
        defaults = dict(OFFICIAL_CINC_CHANNELS)
        merged = dict(defaults)
        merged.update(ch)
        ch = merged
        probe.messages.append(
            f"INFO: using channels={ch} (fill defaults for missing inferences)."
        )
    probe.messages.extend(warn_official_vs_paper(inferred or ch, requested=channels))

    # Fail-fast when caller forces a montage that cannot map stage1.
    require_match = channels is not None
    model = MultiModalSleepFM(
        channels=ch,
        embedding_dim=embedding_dim,
        unify=unify,
    )
    report = apply_official_weights(
        model,
        ckpt,
        strict=strict,
        allow_channel_mismatch=allow_channel_mismatch,
        require_channel_match=require_match,
    )
    report.messages = probe.messages + report.messages
    report.inferred_channels = probe.inferred_channels or report.inferred_channels
    report.temperature = report.temperature if report.temperature is not None else probe.temperature
    model.to(device)
    return model, report


def save_converted_checkpoint(
    official_path: PathLike,
    output_path: PathLike,
    *,
    channels: Optional[Dict[str, int]] = None,
    embedding_dim: int = 512,
    device: str = "cpu",
    strict: bool = False,
    allow_channel_mismatch: bool = False,
) -> AdapterReport:
    """Convert official → our ``best.pt`` layout (``model_state_dict`` + meta)."""
    model, report = build_model_from_official(
        official_path,
        channels=channels,
        embedding_dim=embedding_dim,
        device=device,
        strict=strict,
        unify=False,
        allow_channel_mismatch=allow_channel_mismatch,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "channels": dict(model.channels),
        "embedding_dim": embedding_dim,
        "unify": False,
        "temperature": float(model.temperature.detach().cpu().item()),
        "source": "official_sleepfm_adapter",
        "adapter_notes": report.messages,
        "inferred_channels": report.inferred_channels,
    }
    torch.save(payload, out)
    report.messages.append(f"Wrote native checkpoint: {out}")
    return report
