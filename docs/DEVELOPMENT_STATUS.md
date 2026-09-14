# Development status (meta)

Engineering / paper-protocol notes that should **not** pollute `docs/paper/paper.md`.

## Honest constraints

- **No invented CinC / SHHS / MESA metrics.** Tables in the paper draft that say **待补充** mean data or formal retrain is still missing — not that numbers were omitted from a finished run.
- **Retrain after P0 correctness fixes.** Checkpoints trained before:
  1. VICReg on **raw** private (not L2-normalized unit vectors), and
  2. Retrieval **co-presence** filtering (pair A↔B only where both modalities present),
  must be discarded for any formal real-data narrative. Synthetic demos / unit tests are fine.
- Synthetic AUROC≈0.5 and few-epoch loss curves are **smoke tests only**.

## Closed vs open (code)

| Item | Status |
|------|--------|
| Raw vs normalized private VICReg | Closed (code + tests) |
| Retrieval co-presence + directed pairs | Closed (code + tests) |
| Sample-wise dropout excludes full set (K≥2) | Closed |
| `masked_modality_mean` for temporal + night | Closed |
| Strict fail-closed missing `participant_id` | Closed |
| Malformed `channel_mask` → ValueError | Closed |
| Night continuous `apnea_positive_epoch_rate` (no 5/15/30 bins) | Closed |
| `ChannelAwareMaskedPool` stub | Stub only (not default forward) |
| Default install excludes `fourdvarnet` | Closed (`pyproject.toml`) |
| Real SHHS/MESA DUA download + formal retrain | **Open** (user action) |

## Next steps for SHHS / MESA

1. Obtain NSRR DUA + download EDFs / `*-nsrr.xml`.
2. `scripts/check_data_ready.py --stage raw` → `export_nsrr.py` → `--stage pretrain`.
3. `validate_data.py --strict-participants` / `assert_paper_isolation`.
4. Pretrain LOO baseline + Unify **after** these P0 fixes; evaluate retrieval with co-presence metrics.
5. Only then fill paper tables (still no fabricated numbers).
