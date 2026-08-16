#!/usr/bin/env python
"""Protocol completeness checklist for SleepFM-Unify paper readiness.

Reports local data inventory, credential flags (presence only), and next
commands. Does not download data or invent metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dir_status(rel: str) -> dict:
    p = ROOT / rel
    out = {"path": rel, "exists": p.is_dir()}
    if not p.is_dir():
        return out
    index = p / "index.json"
    out["index_json"] = index.is_file()
    npy = list(p.glob("*.npy"))
    out["npy_count"] = len(npy)
    return out


def _cred_flags() -> dict:
    keys = [
        "PHYSIONET_USER",
        "PHYSIONET_PASSWORD",
        "PHYSIONET_USERNAME",
        "NSRR_TOKEN",
        "WFDB_USER",
    ]
    flags = {k: bool(os.environ.get(k)) for k in keys}
    netrc = Path.home() / ".netrc"
    flags["netrc_exists"] = netrc.is_file()
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    inventory = {
        "synthetic": _dir_status("data/synthetic"),
        "cinc2018_fixture": _dir_status("data/cinc2018_fixture"),
        "raw_cinc2018": _dir_status("data/raw/cinc2018"),
        "raw_shhs": _dir_status("data/raw/shhs"),
        "raw_mesa": _dir_status("data/raw/mesa"),
        "exported_cinc2018": _dir_status("data/cinc2018"),
        "exported_shhs": _dir_status("data/shhs"),
        "exported_mesa": _dir_status("data/mesa"),
    }
    creds = _cred_flags()
    real_ready = any(
        inventory[k].get("exists") and inventory[k].get("npy_count", 0) > 0
        for k in ("exported_cinc2018", "exported_shhs", "exported_mesa")
    )
    report = {
        "inventory": inventory,
        "credentials_present": creds,
        "real_psg_exported": real_ready,
        "paper_metrics_status": "ready_to_fill" if real_ready else "pending_dai_buchong",
        "docs": {
            "data_access": "docs/DATA_ACCESS.md",
            "unify": "docs/UNIFY.md",
        },
        "next_commands": [
            "python scripts/generate_synthetic_data.py --demo",
            "python scripts/run_paper_suite.py --demo",
            "See docs/DATA_ACCESS.md for CinC/SHHS download after credentials exist",
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("SleepFM-Unify protocol checklist")
        print(f"  paper_metrics_status: {report['paper_metrics_status']} (use docs Chinese: 待补充 when pending)")
        print(f"  real_psg_exported: {real_ready}")
        print("  inventory:")
        for name, st in inventory.items():
            print(f"    - {name}: exists={st.get('exists')} npy={st.get('npy_count', 0)} index={st.get('index_json', False)}")
        print("  credentials (presence only):")
        for k, v in creds.items():
            print(f"    - {k}: {'yes' if v else 'no'}")
        print("  next:")
        for c in report["next_commands"]:
            print(f"    - {c}")
    return 0 if inventory["synthetic"].get("exists") else 1


if __name__ == "__main__":
    raise SystemExit(main())
