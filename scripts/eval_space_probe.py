"""Downstream probes on shared-only / private-only / concat embedding spaces.

Useful ablation when Unify is enabled: contrastive terms train shared, while
downstream default concatenates shared||private. Compare the three spaces from
the same checkpoint without retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleepfm.data.splits import assert_paper_isolation
from sleepfm.eval.experiments import probe_split
from sleepfm.models.sleepfm import MultiModalSleepFM
from sleepfm.utils.config import load_config
from sleepfm.utils.seed import set_seed

SPACES = ("concat", "shared", "private", "downstream")


def main():
    parser = argparse.ArgumentParser(
        description="Shared / private / concat downstream probes"
    )
    parser.add_argument("--config", type=str, default="configs/unify.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--spaces",
        type=str,
        default="concat,shared,private",
        help="Comma-separated: concat|shared|private|downstream",
    )
    parser.add_argument(
        "--skip-isolation",
        action="store_true",
        help="Skip assert_paper_isolation (debug only)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    data_dir = args.data_dir or cfg["data_dir"]
    if not args.skip_isolation:
        assert_paper_isolation(data_dir, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(args.checkpoint, device=str(device))
    model.to(device)

    spaces = [s.strip() for s in args.spaces.split(",") if s.strip()]
    for s in spaces:
        if s not in SPACES:
            raise SystemExit(f"Unknown space {s!r}; choose from {SPACES}")

    # Map UI name "concat" to encode space "downstream" when model.downstream_space is concat
    encode_map = {
        "concat": "downstream",
        "downstream": "downstream",
        "shared": "shared",
        "private": "private",
    }

    out = {"checkpoint": args.checkpoint, "data_dir": data_dir, "spaces": {}}
    for name in spaces:
        space = encode_map[name]
        metrics = probe_split(
            model,
            data_dir,
            device,
            cfg["downstream"],
            batch_size=args.batch_size,
            space=space,
        )
        metrics["requested_space"] = name
        metrics["encode_space"] = space
        out["spaces"][name] = metrics
        print(f"[{name}] staging={metrics.get('staging')} apnea={metrics.get('apnea')}")

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
