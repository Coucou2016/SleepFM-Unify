# SleepFM + 4DVarNet (JAMES 2024 reproduction)

This repository hosts two independent research reproductions:

| Package | Paper |
|---------|--------|
| `sleepfm/` | Multi-modal contrastive learning for sleep (ECG, EEG/BAS, respiratory) |
| `fourdvarnet/` | **Inversion of Sea Surface Currents From Satellite-Derived SST-SSH Synergies With 4DVarNets** — Fablet, Chapron, Le Sommer, Sévellec, *JAMES* (2024), [doi:10.1029/2023MS003609](https://doi.org/10.1029/2023MS003609) |

Official reference implementation: [CIA-Oceanix/4dvarnet-james-uv-ssc](https://github.com/CIA-Oceanix/4dvarnet-james-uv-ssc).

---

## 4DVarNet: sea surface currents from SST + SSH

### Method (brief)

4DVarNets combine a **variational assimilation cost** (Eq. 8) with a **learned gradient solver** unfolded over K steps (Eq. 9):

\[
U_\Phi(x,y,z) = \lambda_1 \|y-x\|_\Omega^2 + \lambda_2 \|\mathcal{G}(x)-\mathcal{H}(z)\|^2 + \gamma \|x-\Phi(x)\|^2
\]

- **State** \(x = (\mathrm{SSH}, u, v)\) over a time window  
- **Observations** \(y\): DUACS-like SSH + gappy along-track altimetry; currents are masked (never directly observed)  
- **SST** \(z\): informs \(x\) via trainable convolutional operators \(\mathcal{G}\), \(\mathcal{H}\)  
- **Prior** \(\Phi\): two-scale U-Net; **solver**: ConvLSTM on \(\nabla_x U\) + learned step size  

Training minimizes SSH, SSH-gradient, \((u,v)\), divergence, and \(\Phi\)-regularization losses (Eqs. 10–14).

### Synthetic OSSE (default)

Without NATL60 / ocean-data-challenge files, the pipeline builds a **documented synthetic Gulf-Stream-like benchmark**:

- QG streamfunction → SSH → geostrophic \((u,v)\) + ageostrophic perturbation  
- SQG-style SST linked to SSH gradients  
- Along-track altimetry masks + DUACS smoothing  

For **real data**, place preprocessed `train.npz`, `val.npz`, `test.npz` under `data/fourdvarnet_osse/` with keys: `y_obs`, `obs_mask`, `z_sst`, `ssh`, `u`, `v`, `ssh_seq`, `u_seq`, `v_seq` (see `fourdvarnet/data/dataset.py`). NATL60 Wasabi URLs and layout notes: `fourdvarnet/data/natl60.py`. Synthetic OSSE applies **train-split z-score normalization** (stats in `meta.npz`); geostrophic terms denormalize SSH before `g/f` derivatives.

### Quick start

```powershell
cd E:\Projects\20260522-SleepFM
pip install -r requirements.txt

# Smoke test (generate demo data, train 2 epochs, forward pass)
python scripts/fourdvarnet_smoke_test.py

# Full synthetic pipeline
python scripts/fourdvarnet_generate_data.py
python scripts/fourdvarnet_train.py
python scripts/fourdvarnet_eval.py --checkpoint outputs/fourdvarnet/best.pt
python scripts/fourdvarnet_infer.py --checkpoint outputs/fourdvarnet/best.pt
```

Config: `configs/fourdvarnet.yaml` (grid size, `n_iter`, learning rate, loss weights).

### Evaluation metrics

- RMSE on SSH, \(u\), \(v\)  
- Vector correlation of \((u,v)\)  
- Explained variance  
- Heuristic spectral resolved time scales (`lambda_t_*`) in the spirit of Le Guillou et al. (2020)  

Paper targets (NATL60 OSSE): ~2.5–3 day and ~0.5°–0.7° scales, ~47% ageostrophic divergence skill with SST+SSH vs SSH-only — **not expected on the small synthetic demo** without real training data and full architecture scale (~1.4M params at 1/20°).

### Limitations

- Demo grid is smaller than paper (48×48 vs 200×200 at 1/20°).  
- No Hydra / SWOT-specific configs from the official repo.  
- Real NATL60 paths must be supplied by the user for quantitative comparison to published figures.

---

## SleepFM: multi-modal sleep foundation model

**Paper:** Thapa et al., *SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals* — [ICML 2024](https://proceedings.mlr.press/v235/thapa24a.html), [OpenReview workshop](https://openreview.net/forum?id=cDXtscWCKC), [arXiv:2405.17766](https://arxiv.org/abs/2405.17766).

| Component | Description |
|-----------|-------------|
| **BAS** | Brain activity (EEG/EOG/EMG); 10 channels |
| **ECG** | 2 channels |
| **Respiratory** | 7 channels |
| **Preprocessing** | 30 s epochs @ 256 Hz |
| **Encoders** | 1D EfficientNet-style CNNs → 512-d embeddings |
| **Pretraining** | Pairwise or **leave-one-out** contrastive learning |
| **Downstream** | Logistic regression on concatenated embeddings |

Research links and datasets: [`docs/MATERIALS.md`](docs/MATERIALS.md). Official code: [rthapa84/sleepfm-codebase](https://github.com/rthapa84/sleepfm-codebase).

### Quick start (synthetic demo)

```powershell
cd E:\Projects\20260522-SleepFM
pip install -r requirements.txt
pip install -e .

python scripts/generate_synthetic_data.py --demo
python scripts/validate_data.py --data-dir data/synthetic --strict-participants
python scripts/pretrain.py --demo
python scripts/evaluate_downstream.py --checkpoint outputs/pretrain/best.pt
python scripts/smoke_test.py
python -m pytest tests/ -q
```

Config: `configs/default.yaml`. Unify method + real-PSG export: [`docs/UNIFY.md`](docs/UNIFY.md). Full-scale training omits `--demo` (256 Hz × 30 s, batch 32, 20 epochs).

### SleepFM-Unify (shared–private + mixed loss)

Default config keeps the paper LOO baseline (`unify.enabled: false`). To train Unify:

```powershell
python scripts/pretrain.py --config configs/unify.yaml --data-dir data/synthetic
python scripts/evaluate_downstream.py --checkpoint outputs/unify/best.pt
python scripts/eval_retrieval.py --checkpoint outputs/unify/best.pt --split pretrain
python scripts/eval_modality_ablation.py --checkpoint outputs/unify/best.pt
python scripts/eval_fewshot.py --checkpoint outputs/unify/best.pt --ks 1,2,4
```

Real PSG (after PhysioNet / NSRR download):

```powershell
python scripts/check_data_ready.py --data-dir data/raw/cinc2018 --dataset cinc2018
python scripts/export_edf.py --dataset cinc2018 --input-dir data/raw/cinc2018 --output-dir data/cinc2018 --validate
python scripts/export_nsrr.py --dataset shhs --input-dir data/raw/shhs --output-dir data/shhs --validate
python scripts/export_edf.py --fixture --output-dir data/cinc2018_fixture --validate
python scripts/run_paper_suite.py --data-dir data/cinc2018 --max-gallery 5000
```

Synthetic end-to-end paper suite (CI):

```powershell
python scripts/run_paper_suite.py --demo
```

Official demo weights (optional; CinC ~5/1/3 channels):

```powershell
python scripts/download_checkpoint.py --convert
python scripts/load_official_checkpoint.py --checkpoint outputs/official_checkpoint/best.pt
```

Supervised baselines: `python scripts/train_supervised.py --demo --model effnet` or `--model seq` (`SeqStagingBaseline`, not U-Sleep).

### Verification checklist

| Check | Command | Expected on synthetic demo |
|-------|---------|----------------------------|
| Data schema + split isolation | `python scripts/validate_data.py --data-dir data/synthetic` | Pass; no pretrain↔train path/participant overlap |
| Unit tests | `python scripts/run_tests.py` | All pass (~30s CPU) |
| End-to-end smoke | `python scripts/smoke_test.py` | `SMOKE TEST PASSED` |
| Pretrain → downstream | demo commands above | Loss finite; staging macro AUROC often ~0.35–0.55 (chance ~0.5 for 5-class macro OVR) |
| Retrieval sanity | `python scripts/eval_retrieval.py --checkpoint outputs/pretrain/best.pt --split pretrain` | Recall@k above random baseline after training |
| Official weights (optional) | `python scripts/download_checkpoint.py` | Needs GitHub access; not required for synthetic pipeline |

**Paper targets (real clinic PSG, LOO pretrain):** staging macro AUPRC ~0.72, SDB AUPRC ~0.77 — not reproducible on synthetic data. See [`docs/MATERIALS.md`](docs/MATERIALS.md) and [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md).

**Reproducibility:** set `seed` in config; use `set_seed(..., deterministic=True)` for stricter CUDA reproducibility (slower).

### Layout

- `sleepfm/models/` — EffNet encoders, `MultiModalSleepFM`, optional Unify heads + temporal encoder
- `sleepfm/data/` — synthetic PSG, EDF/MAT/NPZ export, `SleepEpochDataset`, night windows
- `sleepfm/training/` — contrastive / mixed Unify pretraining
- `sleepfm/eval/` — staging, apnea, retrieval, ablation, few-shot, night-level probes
- `scripts/` — CLI entrypoints
- `configs/channels/` — CinC 2018 / SHHS / MESA lead maps

### Real PSG data

Authors' clinic cohort is not public. Use [PhysioNet CinC 2018](https://physionet.org/content/challenge-2018/1.0.0/) (PhysioNet account), or [SHHS](https://sleepdata.org/datasets/shhs) / [MESA](https://sleepdata.org/datasets/mesa) via NSRR DUA. Export with `scripts/export_edf.py` / `scripts/export_nsrr.py` to 30 s `.npy` epochs + `index.json` (`pretrain` / `valid` / `train` / `test`, participant-level). Details: [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md), [`docs/UNIFY.md`](docs/UNIFY.md), [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md), [`docs/MATERIALS.md`](docs/MATERIALS.md). Local inventory: `python scripts/protocol_checklist.py`.

### Citation

```bibtex
@inproceedings{thapa2024sleepfm,
  title={SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals},
  author={Rahul Thapa and Bryan He and Magnus Ruud Kjaer and Hyatt Moore and Gauri Ganjoo and Emmanuel Mignot and James Zou},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
