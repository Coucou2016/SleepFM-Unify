# SleepFM-Unify P1 close-out (2026-08-15, continued)

Follows `docs/reports/2026-08-15-unify-continue.md` after dual-agent P0 fixes.

## Closed this turn (P1)

| Item | Change |
|------|--------|
| Paper suite temporal | `run_paper_suite.py`: `--train-temporal` (unify_temporal pretrain) and `--temporal-checkpoint`; night eval prefers temporal ckpt. **`--demo` stays fast** (temporal opt-in only). |
| Channel meta 5/1/3 vs 10/2/7 | `sleepfm/models/channel_meta.py`; adapter ERROR messages; evaluate / suite **fail-fast** unless `--allow-channel-mismatch`. |
| CinC label gate | `sleepfm/data/label_coverage.py`; export writes `meta.label_coverage` / `label_gate`; evaluate + suite skip claiming staging/SDB when labels degenerate. |
| Docs | `docs/UNIFY.md` + this report. |

## Not done (still blocked / deferred)

- ChatGPT browser upload / senior pass
- Real CinC / SHHS downloads
- P2 gallery RNG, `check_data_ready` naming, clinical AHI wording
- `fourdvarnet/` (explicitly skipped)
- Git commit (not requested)

## Files touched

- `sleepfm/models/channel_meta.py` (new)
- `sleepfm/data/label_coverage.py` (new)
- `sleepfm/models/official_adapter.py`
- `sleepfm/models/sleepfm.py` (`from_checkpoint` warnings)
- `sleepfm/data/edf_export.py` (label meta)
- `scripts/run_paper_suite.py`
- `scripts/evaluate_downstream.py`
- `scripts/evaluate_night.py`
- `scripts/load_official_checkpoint.py`
- `tests/test_channel_label_gates.py` (new)
- `tests/test_paper_suite.py`
- `docs/UNIFY.md`
- `docs/reports/2026-08-15-unify-p1-close.md` (this file)

## Test results

| Gate | Result |
|------|--------|
| `python scripts/run_tests.py` | **64 passed**, 145 warnings, ~75s, exit 0 |
| `python scripts/smoke_test.py` | **SMOKE TEST PASSED**, exit 0 |

`TMP`/`TEMP` = `E:\Projects\20260522-SleepFM\.tmp_pytest`. Host pyarrow AV traceback under pytest still present (known).

## Override flags

| Flag | Where | Meaning |
|------|-------|---------|
| `--allow-channel-mismatch` | suite, evaluate_*, load_official | Acknowledge intentional montage mismatch |
| `--force-metrics` | evaluate_downstream / evaluate_night | Claim metrics despite label gate (debug only) |
| `--train-temporal` | run_paper_suite | Train temporal head before night eval |
| `--temporal-checkpoint` | run_paper_suite | Reuse existing temporal ckpt |
