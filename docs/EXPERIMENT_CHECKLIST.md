# Experiment checklist (before formal real-data tables)

Operational gates for SleepFM-Unify. Manuscript narrative lives in
`docs/paper/paper.md`; engineering status in `docs/DEVELOPMENT_STATUS.md`.

## P0 correctness (must be green before any formal retrain narrative)

| Gate | Status |
|------|--------|
| VICReg / private variance on **raw** private (not L2 unit vectors) | Required |
| Optional \(\lambda_{\mathrm{cov}}\) (`private_cov`) off-diagonal term | Optional config |
| Retrieval co-presence: pair A↔B only where both present; directed A→B / B→A; baseline \(k/N_{\mathrm{pair}}\) | Required |
| `unify.enabled: false` LOO baseline path unchanged | Required |
| No fabricated CinC / SHHS / MESA metrics | Required |

**Policy:** Do not publish formal real-data final tables from checkpoints trained
*before* the raw-private VICReg + retrieval co-presence fixes. Retrain Unify
(and re-evaluate retrieval) after those fixes. Synthetic demos / unit tests are fine.

## Data readiness

1. PhysioNet / NSRR account + DUA as needed (`docs/DATA_ACCESS.md`).
2. `check_data_ready.py --stage raw` → export → `--stage pretrain`.
3. `validate_data.py --strict-participants` / `assert_paper_isolation`
   (missing `participant_id` → fail-closed; missing `night_id` → **N/A**, not silent PASS).
4. Label coverage gate for CinC (no staging/SDB claims on arousal-only).

## Train / eval matrix (fill only after P0 retrain)

| Experiment | Config | Notes |
|------------|--------|-------|
| LOO baseline | `configs/default.yaml` | Paper-aligned; do not disable |
| Unify full | `configs/unify.yaml` | After P0 retrain |
| Unify + temporal | `configs/unify_temporal.yaml` | Optional night head |
| Retrieval | `scripts/eval_retrieval.py` | Co-presence metrics |
| Downstream / space probe | concat / shared / private | Same checkpoint |
| Few-shot | ≥10 repeats, mean±95% CI | Paper mode |
| Night | continuous `apnea_positive_epoch_rate` | Not clinical AHI; no 5/15/30 bins |

## Honesty reminders

- Synthetic AUROC≈0.5 = smoke only.
- Night severity = apnea-positive epoch rate, not AASM AHI.
- Official clinical / NC weights: see `THIRD_PARTY_NOTICES.md` (not redistributed).
- Channel meta 5/1/3 vs schema 10/2/7 fails closed unless `--allow-channel-mismatch`.
