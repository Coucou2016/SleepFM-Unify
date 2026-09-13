# SleepFM-Unify

Robust multimodal pretraining for **heterogeneous / missing PSG** on SleepFM encoders
(shared–private factorization is a *tool*, not the novelty claim).

**Public repo:** [https://github.com/Coucou2016/SleepFM-Unify](https://github.com/Coucou2016/SleepFM-Unify)  
**Method docs:** [`docs/UNIFY.md`](docs/UNIFY.md) · **Paper draft:** [`docs/paper/paper.md`](docs/paper/paper.md) · **Data access:** [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md)  
**License:** MIT ([`LICENSE`](LICENSE)) · **Citation:** [`CITATION.cff`](CITATION.cff)

Upstream SleepFM (Thapa et al., ICML 2024): [PMLR](https://proceedings.mlr.press/v235/thapa24a.html) · [arXiv:2405.17766](https://arxiv.org/abs/2405.17766) · [official code](https://github.com/rthapa84/sleepfm-codebase).

| Component | Description |
|-----------|-------------|
| **BAS** | Brain activity (EEG/EOG/EMG); schema 10 channels |
| **ECG** | 2 channels |
| **Respiratory** | 7 channels |
| **Preprocessing** | 30 s epochs @ 256 Hz |
| **Encoders** | 1D EfficientNet-style CNNs → 512-d embeddings |
| **Baseline** | Paper LOO contrastive (`unify.enabled: false`) |
| **Unify** | Missing-PSG stack + shared/private heads + mixed loss |
| **Downstream** | Logistic regression on concat / shared / private spaces |

### Quick start (synthetic demo)

```powershell
pip install -r requirements.txt
pip install -e .

python scripts/generate_synthetic_data.py --demo
python scripts/validate_data.py --data-dir data/synthetic --strict-participants
python scripts/pretrain.py --demo
python scripts/evaluate_downstream.py --checkpoint outputs/pretrain/best.pt
python scripts/smoke_test.py
python scripts/run_tests.py
```

Config: `configs/default.yaml`. Full-scale training omits `--demo` (256 Hz × 30 s, batch 32, 20 epochs).

### SleepFM-Unify

```powershell
python scripts/pretrain.py --config configs/unify.yaml --data-dir data/synthetic
python scripts/evaluate_downstream.py --checkpoint outputs/unify/best.pt
python scripts/evaluate_downstream.py --checkpoint outputs/unify/best.pt --space shared
python scripts/eval_space_probe.py --checkpoint outputs/unify/best.pt
python scripts/eval_retrieval.py --checkpoint outputs/unify/best.pt --split pretrain
python scripts/eval_modality_ablation.py --checkpoint outputs/unify/best.pt
python scripts/eval_fewshot.py --checkpoint outputs/unify/best.pt --ks 1,2,4
# Fast CI few-shot: add --demo (2 repeats). Paper default: 10 repeats + mean±95% CI.
```

Real PSG (after PhysioNet / NSRR download — credentials required; not shipped):

```powershell
python scripts/check_data_ready.py --data-dir data/raw/cinc2018 --dataset cinc2018
python scripts/export_edf.py --dataset cinc2018 --input-dir data/raw/cinc2018 --output-dir data/cinc2018 --validate
python scripts/export_nsrr.py --dataset shhs --input-dir data/raw/shhs --output-dir data/shhs --validate
python scripts/run_paper_suite.py --data-dir data/cinc2018 --max-gallery 5000 --space-probe
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

### Verification checklist

| Check | Command | Expected on synthetic demo |
|-------|---------|----------------------------|
| Data schema + split isolation | `python scripts/validate_data.py --data-dir data/synthetic` | Pass; no pretrain↔train path/participant overlap |
| Unit tests | `python scripts/run_tests.py` | All pass (~30s CPU) |
| End-to-end smoke | `python scripts/smoke_test.py` | `SMOKE TEST PASSED` |
| Pretrain → downstream | demo commands above | Loss finite; staging macro AUROC often ~0.35–0.55 (chance ~0.5 for 5-class macro OVR) |
| Retrieval sanity | `python scripts/eval_retrieval.py --checkpoint outputs/pretrain/best.pt --split pretrain` | Recall@k above random baseline after training |

**Paper targets (real clinic PSG, LOO pretrain):** staging/SDB numbers from upstream SleepFM papers are **prior art only** — local CinC/SHHS metrics remain **待补充**. See [`docs/MATERIALS.md`](docs/MATERIALS.md) and [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md).

**Reproducibility:** set `seed` in config; use `set_seed(..., deterministic=True)` for stricter CUDA reproducibility (slower).

### Layout

- `sleepfm/models/` — EffNet encoders, `MultiModalSleepFM`, optional Unify heads + temporal encoder
- `sleepfm/data/` — synthetic PSG, EDF/MAT/NPZ export, `SleepEpochDataset`, night windows
- `sleepfm/training/` — contrastive / mixed Unify pretraining
- `sleepfm/eval/` — staging, apnea, retrieval, ablation, few-shot, night-level probes
- `scripts/` — CLI entrypoints
- `configs/channels/` — CinC 2018 / SHHS / MESA lead maps
- `fourdvarnet/` — **legacy** unrelated ocean-assimilation reproduction (not part of SleepFM-Unify claims); see note below

### Real PSG data

Authors' clinic cohort is not public. Use [PhysioNet CinC 2018](https://physionet.org/content/challenge-2018/1.0.0/) (PhysioNet account), or [SHHS](https://sleepdata.org/datasets/shhs) / [MESA](https://sleepdata.org/datasets/mesa) via NSRR DUA. Export with `scripts/export_edf.py` / `scripts/export_nsrr.py`. Details: [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md). Local inventory: `python scripts/protocol_checklist.py`.

### Citation

```bibtex
@software{sleepfm_unify,
  title = {SleepFM-Unify},
  url = {https://github.com/Coucou2016/SleepFM-Unify},
  year = {2026},
  note = {Heterogeneous / missing-PSG robustness on SleepFM encoders}
}

@inproceedings{thapa2024sleepfm,
  title={SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals},
  author={Rahul Thapa and Bryan He and Magnus Ruud Kjaer and Hyatt Moore and Gauri Ganjoo and Emmanuel Mignot and James Zou},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

---

## Legacy note: `fourdvarnet/` (unrelated)

This repo historically also ships a **separate** synthetic OSSE reproduction of Fablet et al., *JAMES* 2024 ([doi:10.1029/2023MS003609](https://doi.org/10.1029/2023MS003609); official [CIA-Oceanix/4dvarnet-james-uv-ssc](https://github.com/CIA-Oceanix/4dvarnet-james-uv-ssc)). It is **not** part of the SleepFM-Unify paper claim. Quick entrypoints: `scripts/fourdvarnet_smoke_test.py`, `configs/fourdvarnet.yaml`. Prefer keeping SleepFM work under `sleepfm/` / `docs/UNIFY.md`.
