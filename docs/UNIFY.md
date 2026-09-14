# SleepFM-Unify

Robust multimodal pretraining for **heterogeneous / missing PSG** on top of SleepFM encoders
(shared–private factorization is a *tool*, not the novelty claim — FOCAL already has shared/private).
The package name stays `sleepfm/`.

Paper-aligned **baseline** is unchanged: `contrastive_mode: leave_one_out` with `unify.enabled: false`.

## Method

Each modality encoder (1D EffNet) still produces a backbone vector of size `embedding_dim` (512). Unify adds two linear heads:

\[
z_m = [z_m^{\mathrm{shared}};\; z_m^{\mathrm{private}}]
\]

Default `shared_dim=256`, `private_dim=256` so the downstream concat per modality remains 512-d (comparable to SleepFM).

| Piece | Where it is used |
|-------|------------------|
| Shared subspace | Pairwise + leave-one-out InfoNCE (cross-modal alignment) on **L2-normalized** shared |
| Private subspace | Orthogonal to shared; **VICReg on raw (pre-L2) private** — never on unit vectors |
| Orthogonality | Center columns, column-L2 normalize, then \(\mathrm{mean}((S^\top P)^2)\) — matches `orthogonality_loss` |
| Mixed loss | \(\lambda_{\mathrm{LOO}}\mathcal{L}_{\mathrm{LOO}} + \lambda_{\mathrm{pair}}\mathcal{L}_{\mathrm{pair}} + \lambda_{\mathrm{priv}}\mathcal{L}_{\mathrm{private}} + \lambda_{\mathrm{orth}}\mathcal{L}_{\mathrm{orth}} + \lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}} + \lambda_{\mathrm{miss}}\mathcal{L}_{\mathrm{miss}}\) |
| Modality dropout | Default **sample-wise** random **non-empty proper** subsets (full set excluded when \(K\ge 2\)); `modality_dropout_mode: batch` keeps legacy drop-one |
| \(\mathcal{L}_{\mathrm{miss}}\) | InfoNCE(remaining shared mean, dropped shared) only where dropped modality was **originally present** |
| Temporal (optional) | GRU/Transformer over `masked_modality_mean` of **post-dropout** shared epoch embeddings |

**Private VICReg (math).** Let \(p\in\mathbb{R}^{N\times D}\) be **raw** private projections (before row L2-normalize), restricted to present rows. With per-dimension std \(s_d=\sqrt{\mathrm{Var}_n(p_{n,d})+\varepsilon}\):

\[
\mathcal{L}_{\mathrm{var}}=\frac{1}{D}\sum_{d=1}^{D}\mathrm{ReLU}(\gamma - s_d),\qquad \gamma=1
\]

Optional covariance term (\(\lambda_{\mathrm{cov}}\) = `loss_weights.private_cov`): center \(P_c\), form \(C=P_c^\top P_c/(N-1)\), then

\[
\mathcal{L}_{\mathrm{cov}}=\frac{1}{D}\sum_{i\neq j} C_{ij}^2,\qquad
\mathcal{L}_{\mathrm{private}}=\mathcal{L}_{\mathrm{var}}+\lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{cov}}.
\]

Do **not** feed L2-normalized unit vectors into \(\mathcal{L}_{\mathrm{var}}\) with \(\gamma=1\): after row-normalize, typical per-dim std is \(\lesssim 1/\sqrt{D}\), so the hinge saturates near \(1-1/\sqrt{D}\) and is not a meaningful anti-collapse signal.

**Mask-aware encode:** rows with `present_mask=0` are skipped in the encoder (no BatchNorm pollution); embeddings for missing modalities are exactly zero. Downstream / night / retrieval multiply by `present_mask` again after encode.

Optional `channel_mask` per modality (dataset → collate → encode) zeros padded/absent leads before the 1D CNN.

| Channel mask | Status |
|--------------|--------|
| Emit masks from index / missing slots; collate stacks them | **Done** |
| Zero padded leads in `encode_backbone` (no fake signal into BN/conv) | **Done** |
| Fixed montage schema 10/2/7 with documented pads | **Done** |
| Mask-aware soft channel attention (`ChannelAwareMaskedPool`, reweight mode) | **Done** (optional; `channel_aware_pool: true` in `configs/unify.yaml`) |
| True variable-channel encoders that **resize** the channel axis | **TODO** (future) |

Downstream default: concatenate shared\(\|\)private per modality (`downstream_space: concat`). Probe helpers / CLIs support `space=shared|private|concat` (`scripts/eval_space_probe.py`, `evaluate_downstream.py --space`, paper suite `--space-probe`). Retrieval uses **shared** embeddings from the **same checkpoint**.

### Downstream evaluation protocol

1. Fit logistic regression on **train** embeddings.
2. Select L2 `C` by scoring on **valid** (fit train → score valid; **no** resubstitution on valid alone).
3. Refit on train with best `C`; evaluate **once** on test.

Split isolation (paper/strict): all pairs among pretrain/valid/train/test must be disjoint on `path`, `participant_id`, and `night_id`/`recording_id` when present — leaks raise `RuntimeError` (`assert_paper_isolation`; also wired in `run_paper_suite.py`). Missing `participant_id` is **fail-closed**. Missing `night_id` is reported as **N/A** (not a silent PASS and not a leak).

Few-shot (paper mode): ≥10 participant-level sampling repeats; report **mean±95% CI** (Student-$t$). Demo/`--demo` may use 2 repeats.

### Novelty positioning

FOCAL already introduces shared/private + orthogonality for multimodal time series. CIMSleepNet imagines missing modalities; PhysioOmni/Omni-Sleep add physiological hierarchy; SleepFM ICML 2024 / Nature Med SleepFM 2026 scale LOO; OSF/SleepBench standardize evaluation. SleepFM-Unify’s claim is **robust SleepFM-style LOO under incomplete/heterogeneous PSG** (sample-wise missing, mask-aware BN, \(\mathcal{L}_{\mathrm{miss}}\) with original presence, optional channel masks, honesty gates) — **not** “novel shared-private factorization.”

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

Retrieval uses the **full split** as the paired gallery by default (not in-batch). Cap it with `--max-gallery N` for large CinC/SHHS runs; omit the flag on synthetic demos. Caps **sample indices from the full dataset before encoding** (`--gallery-mode rng`, default); pass `--gallery-mode prefix` only for legacy comparisons. For each directed pair A→B, only samples where **both** modalities are present enter the gallery/query; random Recall@k baseline is \(\min(k,N_{\mathrm{pair}})/N_{\mathrm{pair}}\). Scripts report directed pairs (e.g. BAS→ECG) plus a macro average. Similarity is chunked to avoid \(N\times N\) OOM.

**Formal real-data checkpoints:** any Unify checkpoint trained *before* the raw-private VICReg + retrieval co-presence fixes must be **retrained** before quoting CinC/SHHS/MESA numbers. This repo does not invent real PSG metrics. See `docs/EXPERIMENT_CHECKLIST.md` and `docs/DEVELOPMENT_STATUS.md`.

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
# Fast CI: python scripts/eval_fewshot.py ... --demo
python scripts/eval_space_probe.py --checkpoint outputs/unify/best.pt --data-dir data/synthetic
python scripts/eval_transfer.py --checkpoint outputs/unify/best.pt --data-dir data/synthetic
python scripts/evaluate_night.py --checkpoint outputs/unify/best.pt --data-dir data/synthetic
python scripts/evaluate_downstream.py --checkpoint outputs/unify/best.pt --space shared
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

# Paper / real-data path: few-shot repeats default to 10; optional space probe
python scripts/run_paper_suite.py --data-dir data/cinc2018 --max-gallery 5000 --space-probe
python scripts/run_paper_suite.py --demo --fewshot-repeats 2 --space-probe

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
