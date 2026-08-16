"""Export EDF / MAT / NPZ PSG folders to SleepFM index.json + .npy epochs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleepfm.data.channel_map import load_channel_table
from sleepfm.data.edf_export import export_dataset, export_recordings, make_fixture_recording
from sleepfm.data.validate import validate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Export PSG recordings to SleepFM DATA_SCHEMA (index.json + .npy)"
    )
    parser.add_argument("--dataset", type=str, default="cinc2018", help="cinc2018 | shhs | mesa")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--channel-config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Write a tiny synthetic PSG fixture (no PhysioNet/NSRR download)",
    )
    parser.add_argument("--validate", action="store_true", help="Run validate_dataset after export")
    args = parser.parse_args()

    table = load_channel_table(args.channel_config or args.dataset)
    print(f"Dataset={args.dataset}  table={table.dataset}  fs={table.target_fs}  clip={table.clip_seconds}s")
    print("Documented missing channels:", table.missing_slots)
    if table.access:
        print("--- access ---")
        print(table.access)
        print("--------------")

    output_dir = args.output_dir or str(ROOT / "data" / args.dataset)

    if args.fixture:
        recs = [
            make_fixture_recording("rec_a", "S001", fs=table.target_fs, duration_sec=table.clip_seconds * 3, seed=0),
            make_fixture_recording("rec_b", "S002", fs=table.target_fs, duration_sec=table.clip_seconds * 3, seed=1),
            make_fixture_recording("rec_c", "S003", fs=table.target_fs, duration_sec=table.clip_seconds * 3, seed=2),
            make_fixture_recording("rec_d", "S004", fs=table.target_fs, duration_sec=table.clip_seconds * 3, seed=3),
        ]
        summary = export_recordings(
            recs,
            output_dir,
            table,
            seed=args.seed,
            dry_run=args.dry_run,
            dataset_name=f"{args.dataset}_fixture",
        )
    else:
        if not args.input_dir:
            parser.error("--input-dir is required unless --fixture is set")
        summary = export_dataset(
            args.input_dir,
            output_dir,
            dataset=args.dataset,
            channel_config=args.channel_config,
            seed=args.seed,
            dry_run=args.dry_run,
            max_recordings=args.max_recordings,
        )

    print(json.dumps({k: v for k, v in summary.items() if k not in ("access", "notes")}, indent=2))
    if summary.get("notes"):
        print("notes:", summary["notes"])
    if summary.get("hint"):
        print(summary["hint"])

    if args.validate and not args.dry_run:
        ok, messages = validate_dataset(output_dir, strict_participants=True)
        for msg in messages:
            print(msg)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
