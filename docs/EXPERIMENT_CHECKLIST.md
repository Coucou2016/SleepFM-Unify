# Experiment checklist (before formal real-data tables)

Operational gates for SleepFM-Unify. Manuscript narrative lives in
`docs/paper/paper.md`; engineering status in `docs/DEVELOPMENT_STATUS.md`.

## P0 correctness (must be green before any formal retrain narrative)

| Gate | Status |
|------|--------|
| VICReg / private variance on **raw** private (not L2 unit vectors) | **Done** (code + CinC retrain) |
| Optional \(\lambda_{\mathrm{cov}}\) (`private_cov`) off-diagonal term | **Done** (Unify config `0.01`) |
| Retrieval co-presence: pair A↔B only where both present; directed A→B / B→A; baseline \(k/N_{\mathrm{pair}}\) | **Done** |
| `unify.enabled: false` LOO baseline path unchanged | **Done** |
| No fabricated CinC / SHHS / MESA metrics | **Done** (CinC measured JSON only; SHHS/MESA empty) |

**Policy:** Formal real-data tables use checkpoints trained **after** the raw-private
VICReg + retrieval co-presence fixes. CinC CPU-8 results: `docs/results/cinc2018_cpu8/`.

## Data readiness

| Step | CinC 2018 | SHHS / MESA |
|------|-----------|-------------|
| Credentials / DUA | Open training (ODC-By); no login | **Blocked** — need NSRR DUA + `NSRR_TOKEN` |
| Raw on disk | Yes (`data/raw/cinc2018/`, 24 subjects pulled) | No |
| Export + validate | Yes (8-subject measured protocol; 24-subject export expanding) | No |
| Label coverage gate | Pass (full AASM + respiratory from `.arousal`) | — |
| Strict participant isolation | Pass | — |

```powershell
python scripts/download_cinc2018_subset.py --max-subjects 24
python scripts/check_data_ready.py --path data/raw/cinc2018 --dataset cinc2018 --stage raw
python scripts/export_edf.py --dataset cinc2018 --input-dir data/raw/cinc2018 --output-dir data/cinc2018 --validate
python scripts/run_paper_suite.py --config configs/cinc_cpu.yaml --unify-config configs/unify_cinc_cpu.yaml --data-dir data/cinc2018 --space-probe --fewshot-repeats 10
```

## Train / eval matrix (filled from measured CinC CPU-8)

| Experiment | Config / script | Status |
|------------|-----------------|--------|
| LOO baseline | `configs/cinc_cpu.yaml` | **Done** — staging AUROC 0.454; apnea 0.662 |
| Unify full | `configs/unify_cinc_cpu.yaml` (`channel_aware_pool: true`) | **Done** — staging 0.495; apnea 0.436 |
| Retrieval | `scripts/eval_retrieval.py` / paper suite | **Done** — Unify R@10 macro 0.0198≈chance |
| Downstream / space probe | concat / shared / private | **Done** |
| Few-shot | ≥10 repeats | **Done** (degenerate: 1 train participant) |
| Modality ablation | 7 subsets | **Done** |
| Night | continuous apnea-positive epoch rate | κ measured; rate **N/A** (&lt;2 nights/split) |
| SeqStagingBaseline + EffNet supervised | `scripts/train_supervised.py` | **Done** (0.518 / 0.393 macro AUROC) |
| FOCAL / CIMSleepNet / OSF | external | Citation only |

## Honesty reminders

- CinC CPU-8 ≠ clinic-scale SleepFM; report sample size with every table.
- Synthetic AUROC≈0.5 = smoke only.
- Night severity = apnea-positive epoch rate, not AASM AHI.
- Official clinical / NC weights: see `THIRD_PARTY_NOTICES.md` (not redistributed).
- Channel meta 5/1/3 vs schema 10/2/7 fails closed unless `--allow-channel-mismatch`.
