"""Source checkpoint → target data_dir linear probe (cross-dataset transfer)."""

import argparse
import json

import torch

from sleepfm.eval.experiments import probe_split
from sleepfm.models.sleepfm import MultiModalSleepFM
from sleepfm.utils.config import load_config
from sleepfm.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="SleepFM transfer: frozen source encoder, target LR")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Source-domain pretrained checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Target-domain SleepFM data_dir")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(args.checkpoint, device=str(device))
    model.to(device)
    metrics = probe_split(
        model,
        args.data_dir,
        device,
        cfg["downstream"],
        batch_size=args.batch_size,
    )
    print(json.dumps({"checkpoint": args.checkpoint, "data_dir": args.data_dir, **metrics}, indent=2))


if __name__ == "__main__":
    main()
