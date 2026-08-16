"""Convert / load official SleepFM checkpoints into our MultiModalSleepFM layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleepfm.models.official_adapter import (  # noqa: E402
    is_official_checkpoint,
    save_converted_checkpoint,
)
import torch  # noqa: E402


def _parse_channels(s: str | None) -> dict | None:
    if not s:
        return None
    # format: bas=5,ecg=1,respiratory=3
    out = {}
    for part in s.split(","):
        k, v = part.strip().split("=")
        out[k.strip()] = int(v.strip())
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Adapt official rthapa84 SleepFM checkpoint → MultiModalSleepFM"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to official best.pt (or zip-extracted checkpoint)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write native checkpoint (model_state_dict). Default: alongside input as *.native.pt",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Override channels, e.g. bas=5,ecg=1,respiratory=3 (default: infer from ckpt)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any encoder tensor shape mismatches",
    )
    parser.add_argument(
        "--allow-channel-mismatch",
        action="store_true",
        help="Allow converting into a non-matching montage (e.g. paper 10/2/7 vs CinC 5/1/3); "
        "stage1 weights are skipped — not recommended",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path to write adapter report JSON",
    )
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.is_file():
        print(f"ERROR: checkpoint not found: {path}")
        sys.exit(1)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not is_official_checkpoint(ckpt):
        print(
            "ERROR: not an official SleepFM checkpoint. Expected keys like "
            "respiratory_state_dict / sleep_stages_state_dict / ekg_state_dict.\n"
            "Native SleepFM checkpoints already load via MultiModalSleepFM.from_checkpoint."
        )
        sys.exit(1)

    channels = _parse_channels(args.channels)
    out = Path(args.output) if args.output else path.with_suffix(".native.pt")
    try:
        report = save_converted_checkpoint(
            path,
            out,
            channels=channels,
            strict=args.strict,
            allow_channel_mismatch=args.allow_channel_mismatch,
        )
    except RuntimeError as exc:
        print(str(exc))
        print(
            "\nHint: omit --channels to infer CinC demo 5/1/3, or pass matching "
            "--channels bas=5,ecg=1,respiratory=3. Use --allow-channel-mismatch "
            "only for intentional partial loads into paper 10/2/7."
        )
        sys.exit(1)

    print(report.summary())
    print()
    print("What maps:")
    print("  sleep_stages_state_dict (aliases: sleep_state_dict) → encoders.bas.*")
    print("  ekg_state_dict (aliases: ecg_state_dict)           → encoders.ecg.*")
    print("  respiratory_state_dict (aliases: resp_state_dict)  → encoders.respiratory.*")
    print("  temperature                                         → model.temperature")
    print("What cannot map:")
    print("  Unify proj_shared/proj_private (official has none)")
    print("  Temporal head (official has none)")
    print("  stage1 when channel counts differ (paper 10/2/7 vs CinC demo 5/1/3)")
    print(f"Native checkpoint: {out}")
    print(f"Load: MultiModalSleepFM.from_checkpoint(r'{out}')")

    if args.report_json:
        payload = {
            "source_keys": report.source_keys,
            "loaded": len(report.loaded),
            "skipped_shape": report.skipped_shape,
            "inferred_channels": report.inferred_channels,
            "temperature": report.temperature,
            "messages": report.messages,
            "output": str(out),
        }
        Path(args.report_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not report.loaded:
        sys.exit(1)


if __name__ == "__main__":
    main()
