"""Generate synthetic OSSE benchmark for 4DVarNet training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sleepfm.utils.config import load_config
from sleepfm.utils.seed import set_seed
from fourdvarnet.data.synthetic_osse import SyntheticOSSEConfig, save_osse_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 4DVarNet synthetic OSSE data")
    parser.add_argument("--config", type=str, default="configs/fourdvarnet.yaml")
    parser.add_argument("--demo", action="store_true", help="Small fast benchmark")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    osse = cfg.get("osse", {})
    if args.demo:
        demo = cfg.get("demo", {})
        osse = {**osse, **{k: demo[k] for k in ("n_lat", "n_lon", "n_days") if k in demo}}

    data_cfg = SyntheticOSSEConfig(
        n_lat=int(osse.get("n_lat", 48)),
        n_lon=int(osse.get("n_lon", 48)),
        n_days=int(osse.get("n_days", 90)),
        n_tracks=int(osse.get("n_tracks", 4)),
        seed=int(osse.get("seed", 42)),
    )
    window = int(cfg.get("window", osse.get("window", 7)))
    out = save_osse_npz(data_cfg, cfg["data_dir"], window=window)
    print(f"Saved OSSE splits to {out}")


if __name__ == "__main__":
    main()
