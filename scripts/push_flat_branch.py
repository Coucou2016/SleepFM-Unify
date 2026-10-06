"""Publish a FLAT (folder-free) tree of this repository to GitHub.

Motivation
----------
The repository owner wants one place where **every** code file, test, config,
document, figure, paper, report and measured-result JSON sits directly at the
repository root - no sub-directories at all - so that an external reader
(chatgpt.com, another agent, a plain browser) can enumerate and read the whole
project without guessing paths, and so that cross-review is unambiguous.

`scripts/build_flat_upload.py` produces that flat file set on disk.  This script
publishes it:

1. uploads each flat file as a Git blob (Git Data API),
2. creates a single tree containing those blobs at the root,
3. creates an **orphan** commit (no parent) with that tree,
4. points `refs/heads/<branch>` (default `flat`) at it,
5. optionally sets that branch as the repository default branch.

An orphan commit is used deliberately: the flat layout shares no paths with the
nested `main` tree, so it must not inherit `main`'s history.  `main` is left
completely untouched and remains the branch for running code (`pip install -e .`,
`pytest`).

Why the API instead of `git push`: on this host `github.com:443` is often
unreachable while `api.github.com` works, so blob/tree/commit upload via `gh api`
is the only reliable transport.

Usage
-----
    python scripts/push_flat_branch.py --flat-dir E:\\sleepfm_flat_upload
    python scripts/push_flat_branch.py --flat-dir E:\\sleepfm_flat_upload --branch flat
    python scripts/push_flat_branch.py --flat-dir E:\\sleepfm_flat_upload --no-default

Safety
------
* Never touches `main` or any other existing branch.
* Only `refs/heads/<branch>` is force-updated, and only for the branch this
  script owns (the mirror branch).
* Secrets are never read from the repo; only the already-generated flat files
  are uploaded.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

REPO = "Coucou2016/SleepFM-Unify"
ROOT = Path(__file__).resolve().parents[1]


def gh_api(method: str, path: str, payload: dict | None = None) -> dict:
    cmd = ["gh", "api", "-X", method, path]
    if payload is None:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True)
    else:
        proc = subprocess.run(
            cmd + ["--input", "-"],
            input=json.dumps(payload).encode("utf-8"),
            cwd=ROOT,
            capture_output=True,
        )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
        sys.stderr.write(proc.stdout.decode("utf-8", errors="replace"))
        raise SystemExit(f"gh api failed: {method} {path} rc={proc.returncode}")
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    return json.loads(text) if text else {}


def create_blob(data: bytes) -> str:
    created = gh_api(
        "POST",
        f"repos/{REPO}/git/blobs",
        {"content": base64.b64encode(data).decode("ascii"), "encoding": "base64"},
    )
    return created["sha"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flat-dir", default=r"E:\sleepfm_flat_upload")
    ap.add_argument("--branch", default="flat")
    ap.add_argument("--message", default=None)
    ap.add_argument("--no-default", action="store_true", help="do not set default branch")
    ap.add_argument(
        "--tree-sha",
        default=None,
        help="reuse an already-uploaded tree (skips blob upload; use after a transient failure)",
    )
    ap.add_argument(
        "--add",
        nargs="*",
        default=None,
        help="upload only these flat files on top of --base-tree (incremental update)",
    )
    ap.add_argument(
        "--base-tree",
        default=None,
        help="existing tree sha to extend when using --add",
    )
    args = ap.parse_args()

    flat = Path(args.flat_dir)
    if not flat.is_dir():
        raise SystemExit(f"flat dir not found: {flat} (run scripts/build_flat_upload.py first)")

    files = sorted(p for p in flat.iterdir() if p.is_file())
    if not files:
        raise SystemExit("flat dir is empty")

    nested = [p for p in flat.iterdir() if p.is_dir()]
    if nested:
        raise SystemExit(f"flat dir must contain NO sub-directories, found {len(nested)}")

    print(f"uploading {len(files)} flat files to {REPO} (branch={args.branch})")

    if args.tree_sha:
        print(f"reusing existing tree {args.tree_sha}")
        tree = {"sha": args.tree_sha}
        tree_items = files
    elif args.add:
        tree_items = []
        total = 0
        for name in args.add:
            p = flat / name
            if not p.is_file():
                raise SystemExit(f"--add file not found in flat dir: {name}")
            data = p.read_bytes()
            total += len(data)
            sha = create_blob(data)
            tree_items.append(
                {"path": name, "mode": "100644", "type": "blob", "sha": sha}
            )
            print(f"  + {name} ({len(data)} bytes)")
        payload = {"tree": tree_items}
        if args.base_tree:
            payload["base_tree"] = args.base_tree
            print(f"extending tree {args.base_tree}")
        tree = gh_api("POST", f"repos/{REPO}/git/trees", payload)
        print(f"tree {tree['sha']}")
    else:
        tree_items = []
        total = 0
        for i, p in enumerate(files, 1):
            data = p.read_bytes()
            total += len(data)
            sha = create_blob(data)
            tree_items.append(
                {"path": p.name, "mode": "100644", "type": "blob", "sha": sha}
            )
            if i % 25 == 0 or i == len(files):
                print(f"  blobs {i}/{len(files)} ({total / 1024 / 1024:.2f} MB)")

        tree = gh_api("POST", f"repos/{REPO}/git/trees", {"tree": tree_items})
        print(f"tree {tree['sha']}")

    message = args.message or (
        "Flat repository layout: all code, docs, paper, report, figures and "
        "measured results at the root, no folders.\n\n"
        "This layout is intentional - see START_HERE_FLAT_LAYOUT.md. It exists so "
        "external readers (chatgpt.com, other agents, browsers) can enumerate and "
        "read every file without path guessing, and so cross-review is "
        "unambiguous.\n\n"
        "The nested, runnable layout is preserved unchanged on `main`.\n\n"
        f"Files: {len(tree_items)}"
    )
    commit = gh_api(
        "POST",
        f"repos/{REPO}/git/commits",
        {"message": message, "tree": tree["sha"], "parents": []},
    )
    print(f"orphan commit {commit['sha']}")

    ref_path = f"repos/{REPO}/git/refs/heads/{args.branch}"
    existing = None
    try:
        existing = gh_api("GET", f"repos/{REPO}/git/ref/heads/{args.branch}")
    except SystemExit:
        existing = None

    if existing is None:
        gh_api(
            "POST",
            f"repos/{REPO}/git/refs",
            {"ref": f"refs/heads/{args.branch}", "sha": commit["sha"]},
        )
        print(f"created refs/heads/{args.branch} -> {commit['sha']}")
    else:
        gh_api(
            "PATCH",
            ref_path,
            {"sha": commit["sha"], "force": True},
        )
        print(f"updated refs/heads/{args.branch} -> {commit['sha']} (mirror branch, force)")

    if not args.no_default:
        gh_api("PATCH", f"repos/{REPO}", {"default_branch": args.branch})
        print(f"default branch set to {args.branch}")

    print(f"https://github.com/{REPO}/tree/{args.branch}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
