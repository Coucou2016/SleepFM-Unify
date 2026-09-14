"""Download an open-access CinC 2018 training subset (no PhysioNet login required).

PhysioNet Challenge 2018 training is ODCA-licensed open data:
https://physionet.org/content/challenge-2018/1.0.0/

Prefers unsigned S3 (`s3://physionet-open/...`) via boto3 when available; falls
back to HTTPS. Full training is ~135 GB — use ``--max-subjects`` for CPU
protocol runs.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_HTTP = "https://physionet.org/files/challenge-2018/1.0.0/training/"
S3_BUCKET = "physionet-open"
S3_PREFIX = "challenge-2018/1.0.0/training"
FILES = (".hea", ".mat", ".arousal", "-arousal.mat")


def list_subjects_http() -> list[str]:
    html = urllib.request.urlopen(BASE_HTTP, timeout=180).read().decode("utf-8", "replace")
    found = re.findall(r'href="(tr\d+-\d+)/"', html)
    out: list[str] = []
    seen = set()
    for s in found:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def list_subjects_s3(s3) -> list[str]:
    out: list[str] = []
    seen = set()
    token = None
    while True:
        kwargs = {
            "Bucket": S3_BUCKET,
            "Prefix": S3_PREFIX + "/",
            "Delimiter": "/",
            "MaxKeys": 1000,
        }
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for p in resp.get("CommonPrefixes") or []:
            pref = p["Prefix"].rstrip("/").split("/")[-1]
            if pref.startswith("tr") and pref not in seen:
                seen.add(pref)
                out.append(pref)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


def _s3_client():
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        return boto3.client("s3", config=Config(signature_version=UNSIGNED, retries={"max_attempts": 8}))
    except Exception:
        return None


def download_file_http(url: str, dest: Path, retries: int = 4) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest.stat().st_size
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SleepFM-Unify/1.0"})
            with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(dest)
            return dest.stat().st_size
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed {url}: {last_err}")


def download_file_s3(s3, key: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest.stat().st_size
    tmp = dest.with_suffix(dest.suffix + ".part")
    s3.download_file(S3_BUCKET, key, str(tmp))
    tmp.replace(dest)
    return dest.stat().st_size


def download_subject(subject: str, out_root: Path, s3=None) -> dict:
    sub_dir = out_root / "training" / subject
    sizes = {}
    for suffix in FILES:
        name = f"{subject}{suffix}"
        dest = sub_dir / name
        try:
            if s3 is not None:
                key = f"{S3_PREFIX}/{subject}/{name}"
                sizes[name] = download_file_s3(s3, key, dest)
            else:
                url = f"{BASE_HTTP}{subject}/{name}"
                sizes[name] = download_file_http(url, dest)
        except Exception as exc:
            if suffix == "-arousal.mat":
                sizes[name] = 0
                print(f"  skip optional {name}: {exc}", flush=True)
            else:
                raise
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "data" / "raw" / "cinc2018"),
    )
    parser.add_argument("--max-subjects", type=int, default=24)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--http-only", action="store_true")
    args = parser.parse_args()

    s3 = None if args.http_only else _s3_client()
    if s3 is not None:
        print("Using unsigned S3 physionet-open", flush=True)
        subjects = list_subjects_s3(s3)
    else:
        print("Using HTTPS PhysioNet mirror", flush=True)
        subjects = list_subjects_http()
    print(f"Found {len(subjects)} training subjects", flush=True)
    if args.list_only:
        for s in subjects[: args.max_subjects]:
            print(s)
        return 0

    chosen = subjects[args.offset : args.offset + args.max_subjects]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    total = 0
    workers = max(1, int(args.workers))

    def _one(item):
        i, subj = item
        print(f"[{i}/{len(chosen)}] start {subj}", flush=True)
        sizes = download_subject(subj, out_root, s3=s3)
        mb = sum(sizes.values()) / 1e6
        print(f"[{i}/{len(chosen)}] done {subj} ({mb:.1f} MB)", flush=True)
        return sum(sizes.values())

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, (i, s)) for i, s in enumerate(chosen, 1)]
        for fut in as_completed(futs):
            total += fut.result()

    print(f"Done. {len(chosen)} subjects, {total/1e9:.2f} GB → {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
