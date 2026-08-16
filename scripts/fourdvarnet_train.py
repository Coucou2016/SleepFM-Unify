"""Train 4DVarNet on synthetic or user-provided OSSE data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sleepfm.utils.config import load_config
from sleepfm.utils.seed import set_seed
from fourdvarnet.training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fourdvarnet.yaml")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.demo:
        d = cfg.get("demo", {})
        cfg["osse"] = {**cfg.get("osse", {}), **{k: d[k] for k in ("n_lat", "n_lon", "n_days") if k in d}}
        cfg["epochs"] = d.get("epochs", 2)
        cfg["batch_size"] = d.get("batch_size", 2)
        cfg["n_iter"] = d.get("n_iter", 4)
        if "loss_weights" in d:
            cfg["loss_weights"] = d["loss_weights"]
    set_seed(cfg.get("seed", 42))
    ckpt = run_training(cfg)
    print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
