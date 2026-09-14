# Development status (meta)

Engineering / paper-protocol notes that should **not** pollute `docs/paper/paper.md`.
See also `docs/EXPERIMENT_CHECKLIST.md` for the pre-retrain gate list.

## Honest constraints

- **No invented CinC / SHHS / MESA metrics.** Empty cells mean data or schedule is
  still missing — not that numbers were omitted from a finished run.
- **Retrain after P0 correctness fixes** is done for the CinC CPU-8 protocol
  (`docs/results/cinc2018_cpu8/`). Checkpoints are post raw-private VICReg +
  retrieval co-presence.
- Synthetic AUROC≈0.5 and few-epoch loss curves remain **smoke tests only** unless
  explicitly labeled as the CinC measured run.

## Closed vs open (code)

| Item | Status |
|------|--------|
| Raw vs normalized private VICReg (+ optional \(\lambda_{\mathrm{cov}}\)) | Closed (code + tests) |
| Retrieval co-presence + directed pairs | Closed (code + tests) |
| Sample-wise dropout excludes full set (K≥2) | Closed |
| `masked_modality_mean` for temporal + night | Closed |
| Strict fail-closed missing `participant_id` | Closed |
| Missing `night_id` → explicit **N/A** (not silent PASS) | Closed |
| Malformed `channel_mask` → ValueError | Closed |
| Night continuous `apnea_positive_epoch_rate` (no 5/15/30 bins) | Closed |
| PRIVATE diagnostics (eff. rank / per-dim std) | Closed (optional logging) |
| `pyproject` extras `paper` / `psg`; fourdvarnet excluded from install | Closed |
| `ChannelAwareMaskedPool` wired (`channel_aware_pool` config; Unify default on) | Closed |
| Real CinC open-subset download + export + LOO/Unify retrain + paper suite | Closed (CPU-8 measured; see `docs/results/cinc2018_cpu8/`) |
| Real SHHS/MESA DUA download + formal retrain | **Open** (user NSRR action) |
| GPU / full-night / multi-seed clinic-scale tables | **Open** (hardware / schedule) |

## CinC measured path (this machine)

1. `python scripts/download_cinc2018_subset.py --max-subjects 24` (open S3; no login).
2. `python scripts/export_edf.py --dataset cinc2018 --input-dir data/raw/cinc2018 --output-dir data/cinc2018 --validate`
3. CPU stride index (keeps real `.npy`; see `data/cinc2018/cpu_protocol.json`).
4. `pretrain.py` LOO + Unify → `run_paper_suite.py` + supervised baselines.
5. Metrics archived under `docs/results/cinc2018_cpu8/` (committable JSON).

## Next steps for SHHS / MESA

1. Obtain NSRR DUA + download EDFs / `*-nsrr.xml`.
2. `scripts/check_data_ready.py --stage raw` → `export_nsrr.py` → `--stage pretrain`.
3. `validate_data.py --strict-participants` / `assert_paper_isolation`.
4. Pretrain LOO baseline + Unify; evaluate retrieval with co-presence metrics.
5. Only then fill SHHS/MESA paper cells (still no fabricated numbers).
