"""Sync the current git HEAD tree to a GitHub branch via the Git Data API.

Why this exists: on this host `github.com:443` (git smart-HTTP) is frequently
unreachable while `api.github.com` works.  `scripts/push_flat_branch.py` handles
the folder-free review branch; this script handles the normal **nested** branch
(`main`) so the two layouts cannot drift apart.

Behaviour
---------
* Uploads every file tracked at `HEAD` (via `git ls-files`) as a blob.
* Builds a **complete** tree (no base_tree) so deletions are handled correctly.
* Creates a commit whose parent is the current remote tip, so `main` history is
  preserved and the push is a normal fast-forward.
* Fast-forward only by default; pass `--force` to overwrite an out-of-band tip
  (e.g. after an earlier API upload that git does not know about).

Usage
-----
    python scripts/push_git_tree_via_api.py --branch main
    python scripts/push_git_tree_via_api.py --branch main --dry-run
    python scripts/push_git_tree_via_api.py --branch main --force
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


def run(cmd: list[str], *, binary: bool = False):
    out = subprocess.check_output(cmd, cwd=ROOT)
    return out if binary else out.decode("utf-8", errors="replace").strip()


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
        raise SystemExit(f"gh api failed: {method} {path} rc={proc.returncode}")
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    return json.loads(text) if text else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    status = run(["git", "status", "--porcelain"])
    if status.strip():
        raise SystemExit("Working tree not clean; commit first")

    paths = [p for p in run(["git", "ls-files"]).splitlines() if p.strip()]
    if not paths:
        raise SystemExit("No tracked files")

    head = run(["git", "rev-parse", "HEAD"])
    remote = gh_api("GET", f"repos/{REPO}/git/ref/heads/{args.branch}")
    parent_sha = remote["object"]["sha"]

    if parent_sha == head:
        print(f"Already in sync: {head}")
        return 0

    print(f"HEAD={head[:12]} remote {args.branch}={parent_sha[:12]} files={len(paths)}")
    if args.dry_run:
        return 0

    tree_items = []
    total = 0
    for i, path in enumerate(paths, 1):
        blob = subprocess.check_output(["git", "cat-file", "-p", f"HEAD:{path}"], cwd=ROOT)
        total += len(blob)
        created = gh_api(
            "POST",
            f"repos/{REPO}/git/blobs",
            {"content": base64.b64encode(blob).decode("ascii"), "encoding": "base64"},
        )
        tree_items.append(
            {"path": path, "mode": "100644", "type": "blob", "sha": created["sha"]}
        )
        if i % 25 == 0 or i == len(paths):
            print(f"  blobs {i}/{len(paths)} ({total / 1024 / 1024:.2f} MB)")

    tree = gh_api("POST", f"repos/{REPO}/git/trees", {"tree": tree_items})
    print(f"tree {tree['sha']}")

    message = args.message or run(["git", "log", "-1", "--format=%B"])
    commit = gh_api(
        "POST",
        f"repos/{REPO}/git/commits",
        {
            "message": message,
            "tree": tree["sha"],
            "parents": [parent_sha],
            "author": {
                "name": run(["git", "log", "-1", "--format=%an"]),
                "email": run(["git", "log", "-1", "--format=%ae"]),
                "date": run(["git", "log", "-1", "--format=%aI"]),
            },
            "committer": {
                "name": run(["git", "log", "-1", "--format=%cn"]),
                "email": run(["git", "log", "-1", "--format=%ce"]),
                "date": run(["git", "log", "-1", "--format=%cI"]),
            },
        },
    )
    print(f"commit {commit['sha']}")

    gh_api(
        "PATCH",
        f"repos/{REPO}/git/refs/heads/{args.branch}",
        {"sha": commit["sha"], "force": bool(args.force)},
    )
    print(f"refs/heads/{args.branch} -> {commit['sha']}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
