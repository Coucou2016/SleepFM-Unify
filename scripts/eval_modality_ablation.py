"""7 missing-modality combinations (paper-plan Table 5)."""

import argparse
import json

import torch

from sleepfm.eval.experiments import modality_ablation_table
from sleepfm.models.sleepfm import MultiModalSleepFM
from sleepfm.utils.config import load_config
from sleepfm.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="SleepFM modality-ablation linear probe")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    data_dir = args.data_dir or cfg["data_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(args.checkpoint, device=str(device))
    model.to(device)
    table = modality_ablation_table(
        model, data_dir, device, cfg["downstream"], batch_size=args.batch_size
    )
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
