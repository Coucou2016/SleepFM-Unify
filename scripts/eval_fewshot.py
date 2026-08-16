"""k-shot linear probe by participant (paper-plan Figure 2 analogue)."""

import argparse
import json

import torch

from sleepfm.eval.experiments import fewshot_curve
from sleepfm.models.sleepfm import MultiModalSleepFM
from sleepfm.utils.config import load_config
from sleepfm.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="SleepFM few-shot (k participants)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ks", type=str, default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    data_dir = args.data_dir or cfg["data_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(args.checkpoint, device=str(device))
    model.to(device)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    curve = fewshot_curve(
        model,
        data_dir,
        device,
        cfg["downstream"],
        ks=ks,
        seed=cfg["seed"],
        batch_size=args.batch_size,
        n_repeats=args.repeats,
    )
    print(json.dumps(curve, indent=2))


if __name__ == "__main__":
    main()
