"""NSRR convenience wrapper around scripts/export_edf.py (SHHS / MESA)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_edf import main as export_main


def main():
    parser = argparse.ArgumentParser(description="Export NSRR SHHS/MESA EDFs to SleepFM schema")
    parser.add_argument("--dataset", type=str, default="shhs", choices=["shhs", "mesa"])
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--channel-config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args, extra = parser.parse_known_args()
    argv = ["--dataset", args.dataset]
    if args.input_dir:
        argv += ["--input-dir", args.input_dir]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if args.channel_config:
        argv += ["--channel-config", args.channel_config]
    argv += ["--seed", str(args.seed)]
    if args.dry_run:
        argv.append("--dry-run")
    if args.max_recordings is not None:
        argv += ["--max-recordings", str(args.max_recordings)]
    if args.fixture:
        argv.append("--fixture")
    if args.validate:
        argv.append("--validate")
    sys.argv = [sys.argv[0]] + argv + extra
    export_main()


if __name__ == "__main__":
    main()
