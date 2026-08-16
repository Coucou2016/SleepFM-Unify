"""Generate synthetic multi-modal sleep dataset for demo runs."""

import argparse
from pathlib import Path

from sleepfm.data.synthetic import write_synthetic_dataset
from sleepfm.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SleepFM dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--demo", action="store_true", help="Use fast demo dimensions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    channels = cfg["channels"]
    if args.demo:
        demo = cfg["demo"]
        sample_rate = demo["sample_rate"]
        clip_seconds = demo["clip_seconds"]
        splits = {
            "pretrain": demo.get("num_pretrain", demo.get("num_train", 64)),
            "valid": demo["num_val"],
            "train": demo.get("num_train", 48),
            "test": demo["num_test"],
        }
    else:
        sample_rate = cfg["sample_rate"]
        clip_seconds = cfg["clip_seconds"]
        splits = {"pretrain": 512, "valid": 64, "train": 256, "test": 128}

    clip_length = int(sample_rate * clip_seconds)
    out = args.output or cfg["data_dir"]
    kwargs = {"seed": cfg["seed"]}
    if args.demo:
        demo = cfg["demo"]
        kwargs["apnea_rate"] = demo.get("apnea_rate", 0.15)
        kwargs["num_participants"] = demo.get("num_participants")
        kwargs["epochs_per_participant"] = demo.get("epochs_per_participant")
    path = write_synthetic_dataset(out, channels, clip_length, splits, **kwargs)
    print(f"Wrote synthetic dataset to {path}")


if __name__ == "__main__":
    main()
