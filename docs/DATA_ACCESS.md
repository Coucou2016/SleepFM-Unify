# Real PSG data access (CinC / SHHS / MESA)

This repository **does not ship** PhysioNet or NSRR recordings. Measured CinC
CPU-8 metrics live in `docs/results/cinc2018_cpu8/` (JSON only). Large raw/export
arrays stay local under `data/` (gitignored).

## Disk status (this machine, 2026-09-15)

| Path | Role | Present? |
|------|------|----------|
| `data/synthetic/` | Demo / CI | Yes |
| `data/cinc2018_fixture/` | Schema fixture (no DUA) | Yes |
| `data/raw/cinc2018/` | Real CinC 2018 (open training subset) | **Yes** (24 subjects via S3) |
| `data/cinc2018/` | Exported epochs + CPU stride index | **Yes** (8-subject measured protocol) |
| `data/cinc2018_full24/` | Full 24-subject export + CPU stride index | **Yes** (21532 full / 2100 CPU epochs) |
| `data/raw/shhs/` | Real SHHS | **No** |
| `data/raw/mesa/` | Real MESA | **No** |
| Env `PHYSIONET_USER` / `PHYSIONET_PASSWORD` | Optional (CinC training is open) | Unset |
| `NSRR_TOKEN` | NSRR downloads | **Unset — SHHS/MESA blocked until you export NSRR_TOKEN** |

Re-check anytime:

```powershell
python scripts/protocol_checklist.py
python scripts/check_data_ready.py --path data/raw/cinc2018 --dataset cinc2018 --stage raw
```

## CinC 2018 (PhysioNet challenge-2018) — open training

Training files are **open-access** (ODC-By). No PhysioNet login required for the
training set used here.

```powershell
# Subset (recommended). Uses unsigned S3 physionet-open when boto3 is installed.
python scripts/download_cinc2018_subset.py --max-subjects 24 --workers 4
python scripts/export_edf.py --dataset cinc2018 --input-dir data/raw/cinc2018 --output-dir data/cinc2018 --validate
python scripts/pretrain.py --config configs/cinc_cpu.yaml --data-dir data/cinc2018
python scripts/pretrain.py --config configs/unify_cinc_cpu.yaml --data-dir data/cinc2018
python scripts/run_paper_suite.py --config configs/cinc_cpu.yaml --unify-config configs/unify_cinc_cpu.yaml --data-dir data/cinc2018 --space-probe --fewshot-repeats 10 --skip-pretrain --checkpoint outputs/pretrain_cinc_cpu/best.pt --unify-checkpoint outputs/unify_cinc_cpu/best.pt

# Full 24-subject export + CPU stride protocol (measured: docs/results/cinc2018_full24_cpu/)
python scripts/export_edf.py --dataset cinc2018 --input-dir data/raw/cinc2018 --output-dir data/cinc2018_full24 --validate
# then build stride index (see data/cinc2018_full24/cpu_protocol.json) and:
python scripts/pretrain.py --config configs/cinc_full24_cpu.yaml --data-dir data/cinc2018_full24
python scripts/pretrain.py --config configs/unify_cinc_full24_cpu.yaml --data-dir data/cinc2018_full24 --unify
python scripts/run_paper_suite.py --config configs/cinc_full24_cpu.yaml --unify-config configs/unify_cinc_full24_cpu.yaml --data-dir data/cinc2018_full24 --space-probe --fewshot-repeats 3 --max-gallery 500 --skip-pretrain --checkpoint outputs/pretrain_cinc_full24_cpu/best.pt --unify-checkpoint outputs/unify_cinc_full24_cpu/best.pt
```

Full ~135 GB training tree:

```text
aws s3 sync --no-sign-request s3://physionet-open/challenge-2018/1.0.0/training/ DESTINATION
```

**Honesty:** Label coverage uses WFDB `.arousal` sleep stages + respiratory events
when present. Gates in `sleepfm/data/label_coverage.py` still apply. CPU-8 tables
are scale-limited measured numbers, not ICML clinic SOTA.

## SHHS / MESA (NSRR) — blocked without user DUA

1. Request access at https://sleepdata.org/ (DUA).
2. Obtain an NSRR token and set `NSRR_TOKEN` (never commit).
3. Download EDFs + `*-nsrr.xml` into `data/raw/shhs/` or `data/raw/mesa/`.
4. Export:

```powershell
python scripts/export_nsrr.py --dataset shhs --input-dir data/raw/shhs --output-dir data/shhs --validate
python scripts/export_nsrr.py --dataset mesa --input-dir data/raw/mesa --output-dir data/mesa --validate
```

Until those credentials exist, **do not invent SHHS/MESA metrics**.

## What Cursor will **not** do

- Invent CinC / SHHS / MESA metrics.
- Commit secrets or huge raw PSG blobs.
- Force-push or deploy.
