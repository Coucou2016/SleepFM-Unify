"""Upload commits ahead of GitHub main via Git Data API (api.github.com).

Use when github.com:443 is blocked but gh api works. Fast-forward only.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

REPO = "Coucou2016/SleepFM-Unify"
ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, binary: bool = False) -> bytes | str:
    out = subprocess.check_output(cmd, cwd=ROOT)
    return out if binary else out.decode("utf-8", errors="replace").strip()


def gh_api(method: str, path: str, payload: dict | None = None):
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


def main() -> None:
    remote = gh_api("GET", f"repos/{REPO}/git/ref/heads/main")
    parent_sha = remote["object"]["sha"]
    parent = gh_api("GET", f"repos/{REPO}/git/commits/{parent_sha}")
    base_tree = parent["tree"]["sha"]
    local_sha = run(["git", "rev-parse", "HEAD"])
    if parent_sha == local_sha:
        print(f"Already up to date: {local_sha}")
        return

    # Diff working tree / HEAD against remote tree content via local files at HEAD
    # Use git diff against the remote tree by fetching tree listing is hard without object;
    # instead diff name-status of files changed since the last known matching tree.
    # Prefer: list files differing between HEAD and what we can compare via `git ls-files`
    # + blob SHAs vs remote tree recursively — simplified: upload all files changed
    # relative to origin/main if present, else use git status + commit range.

    status = run(["git", "status", "--porcelain"])
    if status.strip():
        raise SystemExit("Working tree not clean; commit first")

    # Build file list: all paths that differ from remote tree using GitHub compare API
    # Fallback: git diff --name-status against empty if no local origin match
    try:
        local_base = run(["git", "merge-base", "HEAD", "origin/main"])
    except subprocess.CalledProcessError:
        local_base = None

    # Always use GitHub compare of parent.. cannot; upload HEAD tree wholesale is huge.
    # Strategy: create blobs for every path changed in `git log --name-only parent..HEAD`
    # But local parent may differ in SHA from GitHub. Use content from HEAD for paths
    # that differ from remote by walking `git diff-tree -r --name-status` needs two trees.
    # Recreate: download remote tree file list is expensive.
    # Practical approach used previously: diff vs local origin/main if same tree as remote.

    remote_tree = base_tree
    local_tree = run(["git", "rev-parse", "HEAD^{tree}"])
    if remote_tree == local_tree:
        print("Trees identical; only commit metadata differs — skipping")
        return

    # Get changed paths by comparing trees via git (requires remote tree object).
    # If missing, create a temporary remote-tracking commit with known tree via commit-tree.
    have_remote_tree = True
    try:
        run(["git", "cat-file", "-t", remote_tree])
    except subprocess.CalledProcessError:
        have_remote_tree = False

    if not have_remote_tree:
        # We know blobs for remote tree exist on GitHub; for local diff use
        # `gh api` compare commits if local has the parent commit.
        try:
            run(["git", "cat-file", "-t", parent_sha])
            have_parent = True
        except subprocess.CalledProcessError:
            have_parent = False
        if not have_parent:
            # Diff all files changed in the tip commit only vs its first parent locally
            lines = run(["git", "diff", "--name-status", "HEAD~1", "HEAD"]).splitlines()
        else:
            # Can't tree-diff without remote tree; use name-status from merge-base heuristic
            lines = run(["git", "diff", "--name-status", parent_sha, "HEAD"]).splitlines()
    else:
        lines = run(["git", "diff", "--name-status", remote_tree, "HEAD"]).splitlines()

    tree_items: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        status_code = parts[0]
        if status_code.startswith("R") or status_code.startswith("C"):
            old, path = parts[1], parts[2]
            tree_items.append({"path": old, "mode": "100644", "type": "blob", "sha": None})
        elif status_code.startswith("D"):
            tree_items.append({"path": parts[1], "mode": "100644", "type": "blob", "sha": None})
            continue
        else:
            path = parts[1]
        mode = run(["git", "ls-tree", "HEAD", "--", path]).split()[0]
        blob = run(["git", "cat-file", "-p", f"HEAD:{path}"], binary=True)
        assert isinstance(blob, bytes)
        created = gh_api(
            "POST",
            f"repos/{REPO}/git/blobs",
            {"content": base64.b64encode(blob).decode("ascii"), "encoding": "base64"},
        )
        tree_items.append({"path": path, "mode": mode, "type": "blob", "sha": created["sha"]})
        print(f"blob {status_code} {path} -> {created['sha'][:12]} ({len(blob)} bytes)")

    if not tree_items:
        raise SystemExit("No file changes to upload")

    new_tree = gh_api(
        "POST",
        f"repos/{REPO}/git/trees",
        {"base_tree": base_tree, "tree": tree_items},
    )
    msg = run(["git", "log", "-1", "--format=%B"])
    new_commit = gh_api(
        "POST",
        f"repos/{REPO}/git/commits",
        {
            "message": msg,
            "tree": new_tree["sha"],
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
    updated = gh_api(
        "PATCH",
        f"repos/{REPO}/git/refs/heads/main",
        {"sha": new_commit["sha"], "force": False},
    )
    print(f"updated refs/heads/main -> {updated['object']['sha']}")
    print(f"local HEAD was {local_sha}")
    print("OK")


if __name__ == "__main__":
    main()
