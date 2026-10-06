"""Build a FLAT (no sub-directory) mirror of this repository for GitHub.

Why: the owner wants a single branch where *every* code / documentation / paper /
report / measured-result file sits directly at the repository root, with no
folders at all.  That makes it trivial for external readers (chatgpt.com,
another agent, a browser, or a plain `git clone`) to enumerate and read the full
content without guessing paths, and it makes cross-review easy.

Naming convention (deterministic, collision-free):

    <relative/path/to/file>  ->  <relative>__<path>__<to>__<file>

Root-level files (README.md, LICENSE, ...) keep their original name so that the
repository landing page still renders the README.

Large binaries (``*.pt``, ``*.npy``, raw EDF, zips, ...) are intentionally
excluded: they exceed what a text-reading reviewer / model can consume and would
bloat the mirror.  Everything that matters for reviewing the science - source,
tests, configs, docs, paper, report, figures, and the measured JSON results - is
included.

Usage:
    python scripts/build_flat_upload.py                 # -> E:\\sleepfm_flat_upload
    python scripts/build_flat_upload.py --out D:\\flat  # custom destination
    python scripts/build_flat_upload.py --check         # verify an existing mirror
"""
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- what to walk (relative to repo root) -----------------------------------
INCLUDE_DIRS = [
    "sleepfm",
    "scripts",
    "tests",
    "configs",
    "docs",
    "fourdvarnet",
    "outputs",
    "data",
]
INCLUDE_ROOT_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "requirements.txt",
    "THIRD_PARTY_NOTICES.md",
    ".gitignore",
]

# --- what to skip -----------------------------------------------------------
SKIP_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".tmp_pytest",
    "sleepfm.egg-info",
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".ruff_cache",
    "official_checkpoint",
}
SKIP_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".parquet",
    ".feather",
    ".pkl",
    ".pickle",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".edf",
    ".EDF",
    ".mat",  # WFDB annotations / raw arrays live next to the raw PSG
    ".so",
}
# Heavy raw-data trees are referenced, not mirrored; their small index/metadata
# files are still copied so a reviewer can see the exact dataset composition.
SKIP_PATH_PREFIXES = (
    "data/cinc2018_full24/",
    "data/cinc2018/",
    "data/raw/",
)

# Small index/metadata files we DO want even inside otherwise-skipped trees.
KEEP_ALWAYS_SUFFIXES = {".json", ".yaml", ".yml", ".md", ".csv", ".txt"}
KEEP_ALWAYS_NAMES = {"index.json", "cpu_protocol.json", "meta.json"}

GUIDE_NAME = "START_HERE_FLAT_LAYOUT.md"


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001 - best effort provenance
        return ""


def flat_name(rel: Path) -> str:
    """Root files keep their name; nested files join with '__'."""
    if rel.parent == Path("."):
        return rel.name
    return "__".join(rel.parts)


def should_skip(rel: Path) -> bool:
    posix = rel.as_posix()
    if any(part in SKIP_DIR_NAMES for part in rel.parts):
        return True
    if rel.name.startswith(".env"):
        return True
    if rel.suffix in SKIP_SUFFIXES:
        return False if rel.name in KEEP_ALWAYS_NAMES else True
    if any(posix.startswith(p) for p in SKIP_PATH_PREFIXES):
        # keep only tiny metadata / prose from skipped trees
        if rel.name in KEEP_ALWAYS_NAMES or rel.suffix in KEEP_ALWAYS_SUFFIXES:
            if rel.stat().st_size <= 2 * 1024 * 1024:
                return False
        return True
    return False


def collect() -> list[Path]:
    files: list[Path] = []
    for name in INCLUDE_ROOT_FILES:
        p = ROOT / name
        if p.is_file():
            files.append(Path(name))
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)
            if should_skip(rel):
                continue
            files.append(rel)
    # de-dup, keep deterministic order
    seen: set[str] = set()
    out: list[Path] = []
    for rel in files:
        key = rel.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(rel)
    return sorted(out, key=lambda r: r.as_posix())


def build_guide(manifest: list[dict], out_dir: Path, commit: str, branch: str) -> str:
    total_mb = sum(m["bytes"] for m in manifest) / (1024 * 1024)
    rows = "\n".join(
        f"| `{m['flat']}` | `{m['source']}` | {m['bytes'] / 1024:.1f} KB |"
        for m in manifest
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""# START HERE - this repository is INTENTIONALLY flat (no folders)

**This layout is deliberate, not an accident.** Every source file, test, config,
document, figure, the paper, the research report, the measured result files and
the audit trail are placed **directly in the repository root**, with **no
sub-directories at all**.

Generated: {generated}
Source commit (of the original nested tree): `{commit or "unknown"}`
Mirror branch: `{branch}`
Files in this mirror: **{len(manifest)}** (~{total_mb:.1f} MB of text/JSON/PDF)

---

## 1. Why the flat layout

1. **Full-content reading by external reviewers.** Web-based reviewers and
   model-based reviewers (chatgpt.com, another coding agent, a browser) can
   enumerate a flat root with a single listing call instead of recursively
   guessing folder names. Nothing hides behind a directory they did not think
   to open.
2. **Reproducible cross-review.** Two independent agents reading the *same*
   flat listing will see the *same* file set, which makes disagreement
   debuggable ("you read `docs__paper__paper.md`, I read `paper.md`" cannot
   happen).
3. **Zero path ambiguity.** Deeply nested trees cause reviewers to cite paths
   that do not exist in their view. A one-level namespace removes that class of
   error.
4. **Easy archival.** A flat set of files can be zipped, copied into any
   chat/issue, or diffed against another repository without restructuring.

Trade-off we accept: the flat branch is **not** the canonical Python package
layout, so `pip install -e .` / `pytest` are **not** intended to run from this
branch. Use `main` for running code; use this branch for **reading and
reviewing**. Both branches hold the same file contents.

## 2. Naming convention (how to map a flat name back to a real path)

| Flat name form | Original path |
|---|---|
| `README.md`, `LICENSE`, `pyproject.toml` | root-level files keep their name |
| `sleepfm__models__sleepfm.py` | `sleepfm/models/sleepfm.py` |
| `docs__paper__paper.md` | `docs/paper/paper.md` |
| `docs__results__cinc2018_full24_cpu__measured_compact.json` | `docs/results/cinc2018_full24_cpu/measured_compact.json` |

Rule: replace every `/` with `__` (double underscore). Root files are unchanged.

## 3. Suggested reading order for a reviewer

1. `README.md` - project overview, scope, honest status.
2. `START_HERE_FLAT_LAYOUT.md` - this file (layout + inventory).
3. `docs__paper__paper.md` - the manuscript draft (methods + framing).
4. `docs__reports__report.md` - the parallel research report (long-form
   explanations of every figure and table).
5. `docs__UNIFY.md` - the exact method definition, losses and evaluation
   protocol.
6. `docs__DEVELOPMENT_STATUS.md` / `docs__EXPERIMENT_CHECKLIST.md` - what is
   verified, what is still blank, and why.
7. `docs__results__cinc2018_cpu8__measured_compact.json` and
   `docs__results__cinc2018_full24_cpu__measured_compact.json` - the **measured**
   numbers that back the tables (PhysioNet/CinC 2018, CPU protocol, honest
   scale).
8. `sleepfm__models__sleepfm.py` - the model: shared/private heads, mixed
   contrastive loss, missing-modality objective, channel-aware pooling.
9. `sleepfm__eval__downstream.py`, `sleepfm__eval__retrieval.py`,
   `sleepfm__eval__night.py` - evaluation paths (train/valid/test separation,
   co-presence-filtered retrieval, masked aggregation).
10. `tests__*.py` - the regression tests that pin the behaviour above.
11. `docs__reports__round-0*.md` and `docs__reports__section-19-final*.md` -
    the audit/iteration trail.

## 4. What is deliberately NOT here (and where it lives)

| Excluded | Reason | Where / how to get it |
|---|---|---|
| `*.pt` / `*.pth` checkpoints (~80-170 MB each) | far beyond what a text reviewer can read; Git-hostile | regenerate with `scripts__pretrain.py`; download official with `scripts__download_checkpoint.py` |
| Processed `*.npy` epochs (`data/cinc2018*`, ~16 GB) | size only; content is derived | re-export from raw with `scripts__export_edf.py` |
| Raw PSG / EDF (`data/raw/`, ~3.2 GB) | licensed third-party data, not redistributable | PhysioNet (CinC 2018) / NSRR (SHHS, MESA) under their terms - see `docs__DATA_ACCESS.md` |
| `.zip` source snapshots | redundant with the flat mirror | - |
| `.env`, keys, cookies, tokens | secrets policy | never committed |

Everything excluded is **derived or licensed**, so the mirror still contains
100% of what is needed to audit the science.

## 5. Honesty statement about the numbers

* The only **measured real-data** results in this mirror are the PhysioNet/CinC
  2018 CPU-protocol runs under `docs__results__cinc2018_cpu8__*` and
  `docs__results__cinc2018_full24_cpu__*`.
* SHHS / MESA results are **not present** because access requires an NSRR data
  use agreement. They are marked as pending, never invented.
* Synthetic-data numbers are demo/CI artefacts only and are labelled as such.

## 6. How to cross-review this mirror

1. Read `START_HERE_FLAT_LAYOUT.md` (this file) first - it is the map.
2. Cross-check each claim in `docs__paper__paper.md` against the measured JSON
   in `docs__results__*`.
3. Cross-check the code paths cited in the paper against the corresponding flat
   `sleepfm__*.py` files.
4. Cross-check the stated method against `docs__UNIFY.md` **and** the actual
   implementation; discrepancies are what this review is for.
5. Report findings by flat filename (e.g. `sleepfm__eval__retrieval.py`) so the
   reference is unambiguous.

## 7. Complete inventory ({len(manifest)} files)

| Flat file (this branch) | Original path (main branch) | Size |
|---|---|---|
{rows}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=r"E:\sleepfm_flat_upload")
    ap.add_argument("--branch", default="flat")
    ap.add_argument("--check", action="store_true", help="verify an existing mirror")
    args = ap.parse_args()

    out_dir = Path(args.out)
    commit = git("rev-parse", "HEAD")

    files = collect()
    manifest: list[dict] = []
    for rel in files:
        src = ROOT / rel
        manifest.append(
            {
                "source": rel.as_posix(),
                "flat": flat_name(rel),
                "bytes": src.stat().st_size,
            }
        )

    if args.check:
        missing = [m["flat"] for m in manifest if not (out_dir / m["flat"]).is_file()]
        extras = sorted(
            p.name
            for p in out_dir.iterdir()
            if p.is_file() and p.name not in {m["flat"] for m in manifest} and p.name != GUIDE_NAME
        )
        changed = [
            m["flat"]
            for m in manifest
            if (out_dir / m["flat"]).is_file()
            and not filecmp.cmp(ROOT / m["source"], out_dir / m["flat"], shallow=False)
        ]
        print(f"checked {len(manifest)} files -> missing={len(missing)} extras={len(extras)} changed={len(changed)}")
        for label, items in (("missing", missing), ("extras", extras), ("changed", changed)):
            for i in items[:20]:
                print(f"  {label}: {i}")
        return 0 if not (missing or changed) else 1

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names: set[str] = set()
    for m in manifest:
        flat = m["flat"]
        if flat in names:
            print(f"FATAL: flat name collision on {flat}", file=sys.stderr)
            return 2
        names.add(flat)
        shutil.copy2(ROOT / m["source"], out_dir / flat)

    guide = build_guide(manifest, out_dir, commit, args.branch)
    (out_dir / GUIDE_NAME).write_text(guide, encoding="utf-8")

    (out_dir / "_flat_manifest.json").write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(),
                "source_commit": commit,
                "branch": args.branch,
                "count": len(manifest),
                "files": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total = sum(m["bytes"] for m in manifest)
    print(f"flat mirror -> {out_dir}")
    print(f"files={len(manifest)} total={total / (1024 * 1024):.2f} MB (+{GUIDE_NAME}, _flat_manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
