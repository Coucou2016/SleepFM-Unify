"""Inspect a directory for CinC / SHHS / MESA raw PSG layout readiness.

Exit codes and flags distinguish **raw download readiness** from **exported
pretrain readiness** (``index.json`` schema). Exit 0 means the requested stage
passed; default stage is ``raw`` for backward-compatible CLI use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]

DATASET_HINTS = {
    "cinc2018": {
        "edf": ("*.edf", "*.EDF"),
        "wfdb": ("*.mat", "*.hea"),
        "notes": "PhysioNet CinC 2018 training/ (EDF or WFDB .mat+.hea)",
        "export": (
            "python scripts/export_edf.py --dataset cinc2018 "
            "--input-dir {raw} --output-dir data/cinc2018 --validate"
        ),
    },
    "shhs": {
        "edf": ("*.edf", "*.EDF"),
        "xml": ("*-nsrr.xml", "*nsrr*.xml"),
        "notes": "NSRR SHHS EDFs + annotations-events-nsrr XML",
        "export": (
            "python scripts/export_nsrr.py --dataset shhs "
            "--input-dir {raw} --output-dir data/shhs --validate"
        ),
    },
    "mesa": {
        "edf": ("*.edf", "*.EDF"),
        "xml": ("*-nsrr.xml", "*nsrr*.xml"),
        "notes": "NSRR MESA EDFs + annotations-events-nsrr XML",
        "export": (
            "python scripts/export_nsrr.py --dataset mesa "
            "--input-dir {raw} --output-dir data/mesa --validate"
        ),
    },
}


def _count_globs(root: Path, patterns: Tuple[str, ...], limit: int = 5000) -> int:
    n = 0
    for pat in patterns:
        for _ in root.rglob(pat):
            n += 1
            if n >= limit:
                return n
    return n


def inspect_dir(path: Path, dataset: str | None = None) -> Dict:
    """Inspect path and return readiness flags.

    Keys
    ----
    raw_ready:
        Raw PSG files present (EDF or WFDB mat+hea). Enough to *start* export.
    exported_ready / pretrain_ready:
        ``index.json`` present — schema-ready for validate / pretrain (not a
        guarantee that CinC/SHHS labels are clinically complete).
    ready:
        Alias of ``raw_ready`` when no ``index.json``, else ``pretrain_ready``.
        Kept for older callers; prefer the explicit flags.
    """
    path = path.resolve()
    result = {
        "path": str(path),
        "exists": path.is_dir(),
        "dataset_guess": dataset,
        "counts": {},
        "raw_ready": False,
        "exported_ready": False,
        "pretrain_ready": False,
        "ready": False,
        "messages": [],
        "next_commands": [],
    }
    if not path.is_dir():
        result["messages"].append(f"ERROR: not a directory: {path}")
        return result

    edf_n = _count_globs(path, ("*.edf", "*.EDF"))
    mat_n = _count_globs(path, ("*.mat",))
    hea_n = _count_globs(path, ("*.hea",))
    xml_n = _count_globs(path, ("*-nsrr.xml", "*nsrr*.xml", "*.xml"))
    index_json = (path / "index.json").is_file()
    result["counts"] = {
        "edf": edf_n,
        "mat": mat_n,
        "hea": hea_n,
        "xml": xml_n,
        "index_json": int(index_json),
    }

    if index_json:
        result["messages"].append(
            "Found index.json — exported SleepFM dataset (pretrain_ready=True). "
            "This is NOT the same as raw_ready; run validate before claiming metrics."
        )
        result["exported_ready"] = True
        result["pretrain_ready"] = True
        result["ready"] = True
        result["next_commands"].append(
            f"python scripts/validate_data.py --data-dir {path} --strict-participants"
        )
        result["next_commands"].append(
            f"python scripts/pretrain.py --config configs/default.yaml --data-dir {path} "
            f"--output-dir outputs/{path.name}_loo"
        )
        return result

    # Guess dataset
    guess = dataset
    if guess is None:
        if mat_n or hea_n:
            guess = "cinc2018"
        elif xml_n and edf_n:
            name = path.name.lower()
            if "mesa" in name:
                guess = "mesa"
            elif "shhs" in name:
                guess = "shhs"
            else:
                guess = "shhs"
        elif edf_n:
            name = path.name.lower()
            if "mesa" in name:
                guess = "mesa"
            elif "shhs" in name:
                guess = "shhs"
            else:
                guess = "cinc2018"
    result["dataset_guess"] = guess

    empty = edf_n == 0 and mat_n == 0 and hea_n == 0
    if empty:
        result["messages"].append(
            "ERROR: no EDF / WFDB (.mat/.hea) files found (raw_ready=False). "
            "Download CinC 2018 (PhysioNet) or SHHS/MESA (NSRR DUA), unpack here, re-run."
        )
        result["messages"].append(
            "Access: https://physionet.org/content/challenge-2018/1.0.0/ | "
            "https://sleepdata.org/datasets/shhs | https://sleepdata.org/datasets/mesa"
        )
        result["next_commands"].append(
            "python scripts/export_edf.py --dataset cinc2018 --fixture "
            "--output-dir data/cinc2018_fixture --validate"
        )
        return result

    hint = DATASET_HINTS.get(guess or "cinc2018", DATASET_HINTS["cinc2018"])
    result["messages"].append(f"Detected signals under {path} ({hint['notes']}).")
    result["messages"].append(
        f"Counts: edf={edf_n} mat={mat_n} hea={hea_n} xml={xml_n}"
    )
    if guess in ("shhs", "mesa") and xml_n == 0:
        result["messages"].append(
            "WARNING: NSRR XML annotations not found (*-nsrr.xml). "
            "Staging/apnea labels may be missing until XML is present."
        )
    if guess == "cinc2018" and mat_n and not hea_n:
        result["messages"].append("WARNING: .mat without .hea — WFDB headers usually required.")

    export_cmd = hint["export"].format(raw=path)
    result["next_commands"].append(
        f"python scripts/export_edf.py --dataset cinc2018 --input-dir {path} --dry-run"
        if guess == "cinc2018"
        else f"python scripts/export_nsrr.py --dataset {guess} --input-dir {path} --dry-run"
    )
    result["next_commands"].append(export_cmd)
    result["next_commands"].append(
        f"python scripts/check_data_ready.py --data-dir data/{guess} --stage pretrain"
    )
    result["raw_ready"] = edf_n > 0 or (mat_n > 0 and hea_n > 0)
    result["ready"] = result["raw_ready"]
    result["messages"].append(
        f"raw_ready={result['raw_ready']} (files present for export); "
        "pretrain_ready=False until export writes index.json."
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check SleepFM data readiness. "
            "Exit 0 iff the selected --stage passes: "
            "raw = raw_ready; pretrain/exported = pretrain_ready (index.json)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Raw download dir or exported dataset dir",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cinc2018", "shhs", "mesa"],
        default=None,
        help="Force dataset type (else guessed from files / folder name)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["raw", "pretrain", "exported"],
        default="raw",
        help=(
            "Which readiness gate drives exit code. "
            "raw (default): raw PSG present. "
            "pretrain/exported: index.json present for validate/pretrain."
        ),
    )
    args = parser.parse_args()
    result = inspect_dir(Path(args.data_dir), dataset=args.dataset)
    for msg in result["messages"]:
        print(msg)
    if result["counts"]:
        print("counts:", result["counts"])
    if result["dataset_guess"]:
        print("dataset_guess:", result["dataset_guess"])
    print(
        "flags:",
        {
            "raw_ready": result["raw_ready"],
            "exported_ready": result["exported_ready"],
            "pretrain_ready": result["pretrain_ready"],
            "ready_alias": result["ready"],
            "stage": args.stage,
        },
    )
    if result["next_commands"]:
        print("\nNext commands:")
        for cmd in result["next_commands"]:
            print(f"  {cmd}")

    if args.stage == "raw":
        ok = bool(result["raw_ready"] or result["pretrain_ready"])
    else:
        ok = bool(result["pretrain_ready"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
