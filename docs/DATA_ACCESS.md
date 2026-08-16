# Real PSG data access (CinC / SHHS / MESA)

This repository **does not ship** PhysioNet or NSRR recordings. Paper tables that need
clinical metrics remain **待补充** until you complete the steps below.

## Disk status (this machine, 2026-08-16)

| Path | Role | Present? |
|------|------|----------|
| `data/synthetic/` | Demo / CI | Yes |
| `data/cinc2018_fixture/` | Schema fixture (no DUA) | Yes |
| `data/raw/cinc2018/` | Real CinC 2018 | **No** |
| `data/raw/shhs/` | Real SHHS | **No** |
| `data/raw/mesa/` | Real MESA | **No** |
| Env `PHYSIONET_USER` / `PHYSIONET_PASSWORD` | PhysioNet wget/wfdb | **Unset** |
| `~/.netrc` PhysioNet entry | Same | **Not found** |
| `NSRR_TOKEN` | NSRR downloads | **Unset** |

## CinC 2018 (PhysioNet challenge-2018)

1. Create a free account at https://physionet.org/ and complete any required training / EULA for **You Snooze You Win — The PhysioNet Computing in Cardiology Challenge 2018** (`challenge-2018`).
2. Set credentials (PowerShell example; do **not** commit these):

```powershell
$env:PHYSIONET_USER = "your_physionet_username"
$env:PHYSIONET_PASSWORD = "your_physionet_password"
# optional legacy:
# New-Item $env:USERPROFILE\.netrc -ItemType File -Force
# Add-Content $env:USERPROFILE\.netrc "machine physionet.org login USER password PASS"
```

3. Download training set into `data/raw/cinc2018/` (official layout under `training/`). Prefer PhysioNet’s documented `wget`/`wfdb` flow for `https://physionet.org/files/challenge-2018/1.0.0/`.
4. Gate + export:

```powershell
python scripts/check_data_ready.py --path data/raw/cinc2018 --dataset cinc2018 --stage raw
python scripts/export_edf.py --dataset cinc2018 --input-dir data/raw/cinc2018 --output-dir data/cinc2018 --validate
python scripts/check_data_ready.py --path data/cinc2018 --dataset cinc2018 --stage pretrain
python scripts/run_paper_suite.py --data-dir data/cinc2018 --output-dir outputs/paper_suite_cinc
```

5. **Honesty:** CinC 2018 is arousal-centric; staging / SDB label coverage is gated by
   `sleepfm/data/label_coverage.py`. Do not claim SleepFM ICML staging AUROC numbers
   until local coverage passes.

## SHHS / MESA (NSRR)

1. Request access at https://sleepdata.org/ (DUA).
2. Obtain an NSRR token and set `NSRR_TOKEN` (never commit).
3. Download EDFs + `*-nsrr.xml` into `data/raw/shhs/` or `data/raw/mesa/`.
4. Export:

```powershell
python scripts/export_nsrr.py --dataset shhs --input-dir data/raw/shhs --output-dir data/shhs --validate
python scripts/export_nsrr.py --dataset mesa --input-dir data/raw/mesa --output-dir data/mesa --validate
```

## What Cursor will **not** do

- Invent CinC / SHHS / MESA metrics.
- Download challenge data without credentials.
- Commit `/data/` or `/outputs/` (see `.gitignore`).
