"""CLI: validate SleepFM data_dir (index.json + .npy files + split isolation)."""

import argparse
import sys

from sleepfm.data.validate import validate_dataset


def main():
    parser = argparse.ArgumentParser(description="Validate SleepFM dataset on disk")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument(
        "--strict-participants",
        action="store_true",
        help="Require participant_level_splits in meta",
    )
    args = parser.parse_args()
    ok, messages = validate_dataset(args.data_dir, strict_participants=args.strict_participants)
    for msg in messages:
        print(msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
