# SleepFM-Unify

Shared–private factorization + mixed contrastive loss + modality dropout + optional night-level temporal context, **on top of the existing SleepFM encoders**. The package name stays `sleepfm/`.

Paper-aligned **baseline** is unchanged: `contrastive_mode: leave_one_out` with `unify.enabled: false`.

## Method

Each modality encoder (1D EffNet) still produces a backbone vector of size `embedding_dim` (512). Unify adds two linear heads:

\[
z_m = [z_m^{\mathrm{shared}};\; z_m^{\mathrm{private}}]
\]

Default `shared_dim=256`, `private_dim=256` so the downstream concat per modality remains 512-d (comparable to SleepFM).

| Piece | Where it is used |
|-------|------------------|
| Shared subspace | Pairwise + leave-one-out InfoNCE (cross-modal alignment) |
| Private subspace | Kept out of contrastive loss; orthogonality vs shared |
| Mixed loss | \(\lambda_{\mathrm{LOO}}\mathcal{L}_{\mathrm{LOO}} + \lambda_{\mathrm{pair}}\mathcal{L}_{\mathrm{pair}} + \lambda_{\mathrm{orth}}\mathrm{mean}((Z_s^\top Z_p)^2) + \lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}} + \lambda_{\mathrm{miss}}\mathcal{L}_{\mathrm{miss}}\) (implementation uses mean squared Gram entries after row-L2; not an unnormalized Frobenius sum) |
| Modality dropout | Batch-level drop of one present modality; LOO mean over **remaining** modalities only |
| \(\mathcal{L}_{\mathrm{miss}}\) | InfoNCE(mean of remaining shared, dropped shared) when a modality was dropped |
| Temporal (optional) | GRU/Transformer over a window of **shared** epoch embeddings; adjacent-epoch contrastive + masked epoch MSE |

Missing keys in a batch are skipped (zero-fill + `present_mask` in the dataset). Training does not crash if ECG or respiratory is absent.

Downstream default: concatenate shared\(\|\)private per modality (`downstream_space: concat`). Retrieval uses **shared** embeddings from the **same checkpoint**.

## How to run

### Baseline SleepFM (paper LOO)

```powershell
python scripts/generate_synthetic_data.py --demo
python scripts/pretrain.py --demo
python scripts/evaluate_downstream.py --checkpoint outputs/pretrain/best.pt
python scripts/eval_retrieval.py --checkpoint outputs/pretrain/best.pt --split pretrain
```

Full-scale (256 Hz × 30 s) omits `--demo`. Same commands on a real `data_dir` after export.

### SleepFM-Unify

```powershell
python scripts/pretrain.py --config configs/unify.yaml --data-dir data/synthetic
# equivalent: python scripts/pretrain.py --unify --data-dir data/synthetic
python scripts/evaluate_downstream.py --config configs/unify.yaml --checkpoint outputs/unify/best.pt
python scripts/eval_retrieval.py --checkpoint outputs/unify/best.pt --split pretrain
```

Night-level temporal head:

```powershell
python scripts/pretrain.py --config configs/unify_temporal.yaml --data-dir data/synthetic
python scripts/evaluate_night.py --checkpoint outputs/unify_temporal/best.pt --data-dir data/synthetic
```

`--night` (or `temporal.enabled: true`) forces a non-zero `loss_weights.temporal` default (`0.2` unless `temporal.loss_weight` is set), so the temporal head is not left unused when `unify.yaml` still has `temporal: 0.0`. GRU paths pack padded sequences; Transformer continues to use `src_key_padding_mask`.

`evaluate_night.py` reloads `temporal_state_dict` when the checkpoint has a temporal head and contextualizes each night’s epoch sequence before pooling. Checkpoints without temporal weights fall back to mean-pooling epoch embeddings. When `stage_id` labels exist, the script also reports Cohen’s κ (and linear-weighted κ) on epoch sequences.

Mixed Unify `pretrain_loss` encodes shared/private embeddings **once per step**; LOO, pairwise, orthogonality, miss, and temporal terms reuse that encode.

Retrieval uses the **full split** as the paired gallery by default (not in-batch). Cap it with `--max-gallery N` for large CinC/SHHS runs; omit the flag on synthetic demos. Caps use **seeded RNG subsample** (`--gallery-mode rng`, default) rather than a loader-order prefix; pass `--gallery-mode prefix` only for legacy comparisons.

### Real PSG export (CinC / SHHS / MESA)

No PhysioNet/NSRR files are shipped. The exporter + channel tables + fixture test are complete without terabytes of data.

```powershell
# Dry-run (prints mapping / access notes; writes nothing)
python scripts/export_edf.py --dataset cinc2018 --input-dir data/raw/cinc2018 --dry-run
python scripts/export_nsrr.py --dataset shhs --input-dir data/raw/shhs --dry-run

# Tiny fixture dataset (schema-identical, no DUA)
python scripts/export_edf.py --dataset cinc2018 --fixture --output-dir data/cinc2018_fixture --validate

# After download (see access below)
python scripts/export_edf.py --dataset cinc2018 --input-dir data/raw/cinc2018 --output-dir data/cinc2018 --validate
python scripts/export_nsrr.py --dataset shhs --input-dir data/raw/shhs --output-dir data/shhs --validate
python scripts/export_nsrr.py --dataset mesa --input-dir data/raw/mesa --output-dir data/mesa --validate
```

Channel tables: `configs/channels/cinc2018.yaml`, `shhs.yaml`, `mesa.yaml`. Unmatched leads are **zero-padded** so the on-disk schema stays 10 / 2 / 7 (`docs/DATA_SCHEMA.md`). Padded slots are documented in each YAML (`missing_slots`).

Then pretrain as usual:

```powershell
python scripts/pretrain.py --config configs/default.yaml --data-dir data/cinc2018 --output-dir outputs/cinc_loo
python scripts/pretrain.py --config configs/unify.yaml --data-dir data/cinc2018 --output-dir outputs/cinc_unify
```

### Experiment CLIs (same checkpoint)

```powershell
python scripts/eval_modality_ablation.py --checkpoint outputs/unify/best.pt --data-dir data/synthetic
python scripts/eval_fewshot.py --checkpoint outputs/unify/best.pt --data-dir data/synthetic --ks 1,2,4
python scripts/eval_transfer.py --checkpoint outputs/unify/best.pt --data-dir data/synthetic
python scripts/evaluate_night.py --checkpoint outputs/unify/best.pt --data-dir data/synthetic
```

## Data access (user action required)

| Dataset | What you need |
|---------|----------------|
| [PhysioNet CinC 2018](https://physionet.org/content/challenge-2018/1.0.0/) | PhysioNet account; download `training/` (`.mat`+`.hea` or EDF) |
| [NSRR SHHS](https://sleepdata.org/datasets/shhs) | NSRR account + **executed DUA**; EDFs + `*-nsrr.xml` |
| [NSRR MESA](https://sleepdata.org/datasets/mesa) | NSRR account + **executed DUA**; EDFs + `*-nsrr.xml` |

CinC is primarily an **arousal** challenge: full AASM stages / respiratory events may be missing unless you add a sidecar `.xml`/`.csv`/`.json`. See **CinC label coverage gate** under Official checkpoint notes below. SHHS/MESA NSRR XML is the preferred annotation source.

Optional readers for awkward hospital EDFs: `pip install mne pyedflib wfdb`. Fixtures and standard 16-bit EDF use the built-in reader (no extra packages).

## Next steps when you have data

```powershell
# 1) Inspect raw download (exit 1 if raw_ready=False)
python scripts/check_data_ready.py --data-dir data/raw/cinc2018 --dataset cinc2018 --stage raw
python scripts/check_data_ready.py --data-dir data/raw/shhs --dataset shhs --stage raw
# After export, gate on index.json (pretrain_ready):
python scripts/check_data_ready.py --data-dir data/cinc2018 --stage pretrain

# 2) Export → validate → paper suite (or stepwise CLIs above)
python scripts/export_edf.py --dataset cinc2018 --input-dir data/raw/cinc2018 --output-dir data/cinc2018 --validate
python scripts/run_paper_suite.py --data-dir data/cinc2018 --max-gallery 5000 --output-dir outputs/paper_suite

# Synthetic CI path (no PhysioNet/NSRR):
python scripts/run_paper_suite.py --demo
```

`check_data_ready` prints explicit flags: `raw_ready` (PSG files present for export) vs `pretrain_ready` / `exported_ready` (`index.json` present). Exit code follows `--stage` (`raw` default; `pretrain`/`exported` for schema-ready). Do not treat exit 0 on a raw tree as “ready to claim CinC/SHHS paper metrics.”

### Official SleepFM checkpoint

```powershell
python scripts/download_checkpoint.py --convert
python scripts/load_official_checkpoint.py --checkpoint outputs/official_checkpoint/best.pt
```

Maps `sleep_stages→bas`, `ekg→ecg`, `respiratory→respiratory`. CinC demo weights use ~5/1/3 channels; paper clinic configs stay 10/2/7.

**Channel meta check (fail-fast):** adapter + `evaluate_downstream` / `evaluate_night` / `run_paper_suite` compare checkpoint channels to `index.json` `meta.channels`. Official `5/1/3` vs export `10/2/7` prints `CHANNEL MISMATCH` and aborts unless you pass `--allow-channel-mismatch` (documented override for intentional partial loads). Convert with inferred channels:

```powershell
python scripts/load_official_checkpoint.py --checkpoint path/to/official_best.pt
# or explicit: --channels bas=5,ecg=1,respiratory=3
# paper montage against CinC weights fails without --allow-channel-mismatch
```

### CinC label coverage gate

CinC is primarily an **arousal** challenge: full AASM stages / respiratory events may be missing unless you add a sidecar `.xml`/`.csv`/`.json`. Export writes `meta.label_coverage` + `meta.label_gate`. Evaluate / paper suite **do not claim** staging or SDB/AHI metrics when labels are degenerate (Wake-only / zero apnea). Override with `--force-metrics` only for debugging.

SHHS/MESA NSRR XML is the preferred annotation source.

### Paper suite

```powershell
# Fast CI path (dual LOO+Unify only; night uses mean-pool unless temporal ckpt given)
python scripts/run_paper_suite.py --demo

# Real temporal night head (opt-in; still uses demo epoch counts when --demo)
python scripts/run_paper_suite.py --demo --train-temporal
python scripts/run_paper_suite.py --data-dir data/cinc2018 --train-temporal --max-gallery 5000
# or reuse a unify_temporal checkpoint:
python scripts/run_paper_suite.py --demo --skip-pretrain --checkpoint outputs/x/best.pt --temporal-checkpoint outputs/unify_temporal/best.pt
```

Optional readers for awkward hospital EDFs: `pip install mne pyedflib wfdb`. Fixtures and standard 16-bit EDF use the built-in reader (no extra packages).

### Supervised baselines (not U-Sleep)

```powershell
python scripts/train_supervised.py --demo --model effnet          # concat EffNet, train split → test metrics
python scripts/train_supervised.py --demo --model seq --window 8  # SeqStagingBaseline (CNN+GRU)
```

`SeqStagingBaseline` is an in-repo sequence model for fairer night-level comparison. We do **not** vendor U-Sleep; if you install an external `u-sleep` package yourself, treat it as optional and out of tree.

## GPU / paths

SHHS-scale 256 Hz pretraining needs a GPU (24 GB is a reasonable start). The synthetic demo and unit tests are CPU-only. Set `data_dir` / `--input-dir` to wherever you unpack the DUA downloads.

## Tests

```powershell
python scripts/smoke_test.py
python scripts/run_tests.py
```
