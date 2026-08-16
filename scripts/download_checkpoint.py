"""Download official SleepFM demo checkpoint (optional; requires network)."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

# Official repo bundles a small checkpoint under sleepfm/checkpoint/
DEFAULT_URL = (
    "https://github.com/rthapa84/sleepfm-codebase/archive/refs/heads/main.zip"
)
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Download SleepFM official checkpoint")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "outputs" / "official_checkpoint"),
        help="Directory to store extracted .pt files",
    )
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Also write a native MultiModalSleepFM checkpoint via official_adapter",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="With --convert, fail on any shape mismatch",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "sleepfm-codebase.zip"

    print(f"Downloading {args.url} ...")
    try:
        urlretrieve(args.url, zip_path)
    except URLError as exc:
        print(
            "Download failed (network blocked or URL changed). "
            "Manually clone https://github.com/rthapa84/sleepfm-codebase "
            f"and copy sleepfm/checkpoint/best.pt into {out_dir}\n"
            "Then: python scripts/load_official_checkpoint.py "
            f"--checkpoint {out_dir}/best.pt --convert"
        )
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Extracting to {out_dir} ...")
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if "checkpoint" in name and name.endswith(".pt"):
                zf.extract(name, out_dir)
                extracted.append(out_dir / name)
                print(f"  extracted {name}")

    if not extracted:
        print("No .pt files found under checkpoint/ in the archive.")
        sys.exit(1)

    # Prefer a flat copy for convenience
    primary = extracted[0]
    flat = out_dir / "best.pt"
    if primary.resolve() != flat.resolve():
        flat.write_bytes(primary.read_bytes())
        print(f"  copied → {flat}")

    print(
        "Official keys: respiratory_state_dict / sleep_stages_state_dict / ekg_state_dict.\n"
        "Convert with:\n"
        f"  python scripts/load_official_checkpoint.py --checkpoint {flat}\n"
        "Or MultiModalSleepFM.from_checkpoint(path) (auto-detects official layout).\n"
        "Note: CinC demo uses ~5/1/3 channels, not paper 10/2/7 — adapter reports mismatches."
    )

    if args.convert:
        from sleepfm.models.official_adapter import save_converted_checkpoint

        native = out_dir / "best.native.pt"
        report = save_converted_checkpoint(flat, native, strict=args.strict)
        print(report.summary())
        if not report.loaded:
            sys.exit(1)


if __name__ == "__main__":
    main()
