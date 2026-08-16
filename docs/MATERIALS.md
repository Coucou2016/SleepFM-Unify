# SleepFM — Research Materials

## Paper & versions

| Resource | URL |
|----------|-----|
| OpenReview (AAAI 2024 Clinical FMs) | https://openreview.net/forum?id=cDXtscWCKC |
| arXiv | https://arxiv.org/abs/2405.17766 |
| ICML 2024 proceedings | https://proceedings.mlr.press/v235/thapa24a.html |
| ICML poster | https://icml.cc/virtual/2024/poster/34078 |

**OpenReview submission note:** Authors marked *"No, we will not be making any data and/or code public"* on that workshop track; the **ICML release** and GitHub repos are the public artifacts.

## Official / related code

| Repository | Notes |
|------------|-------|
| [rthapa84/sleepfm-codebase](https://github.com/rthapa84/sleepfm-codebase) | Primary open pipeline; CinC 2018 demo; checkpoint in `sleepfm/checkpoint/` |
| [zou-group/sleepfm-clinical](https://github.com/zou-group/sleepfm-clinical/tree/sleepfm_release) | Larger clinical foundation model (585k+ hours); disease prediction |

**Pretrained weights:** Small checkpoint bundled in official repo (not full paper model). No HuggingFace weights found.

**Loading official weights into this repo:** Official `best.pt` uses keys `respiratory_state_dict` / `sleep_stages_state_dict` / `ekg_state_dict` (+ aliases `resp_*` / `sleep_*`) and CinC-demo channel counts (~5 / 1 / 3), not paper clinic 10 / 2 / 7.

```powershell
python scripts/download_checkpoint.py --convert   # needs network
# or after manual copy:
python scripts/load_official_checkpoint.py --checkpoint outputs/official_checkpoint/best.pt
# MultiModalSleepFM.from_checkpoint(...) also auto-detects official layout
```

See `sleepfm/models/official_adapter.py` for the mapping report (what maps / shape mismatches).

## Datasets mentioned

| Dataset | Access |
|---------|--------|
| Internal Stanford clinic PSG | Not released (~14k participants, 100k+ hours) |
| [PhysioNet CinC 2018](https://physionet.org/content/challenge-2018/1.0.0/) | **PhysioNet account** (create at physionet.org, then download `training/`). Official SleepFM external demo. |
| [SHHS](https://sleepdata.org/datasets/shhs) | **NSRR account + executed DUA** (often days–weeks). Download EDFs + `annotations-events-nsrr` XML. |
| [MESA](https://sleepdata.org/datasets/mesa) | **NSRR account + executed DUA**. Same layout: EDFs + NSRR XML. |

Exporter: `python scripts/export_edf.py --dataset cinc2018 --input-dir <raw> --output-dir data/cinc2018` and `python scripts/export_nsrr.py --dataset shhs|mesa ...`. Channel maps and padded leads: `configs/channels/`. Method stack: [`docs/UNIFY.md`](UNIFY.md).

Without a DUA, use the synthetic demo or `python scripts/export_edf.py --fixture` (schema-identical, not paper numbers).

## Architecture summary

1. **Input:** 30-second clips, resampled to **256 Hz** per channel group.
2. **Encoders:** Three independent **1D EfficientNet** CNNs (Ouyang et al. 2022 style); first conv adapts to channel count (10 / 2 / 7).
3. **Embedding:** 512-D L2-normalized vectors per modality.
4. **Contrastive objectives:**
   - **Pairwise:** All modality pairs \((i,j)\); symmetric InfoNCE in batch.
   - **Leave-one-out:** Modality \(i\) vs mean embedding of all other modalities (paper’s best for downstream).
5. **Training:** SGD, lr 0.001, momentum 0.9, decay ×0.1 every 5 epochs, batch 32, max 20 epochs, trainable temperature \(\tau\) (init 0).
6. **Downstream:** Concatenate modality embeddings → **logistic regression** (L2, balanced classes, LBFGS) on the **`train`** split; evaluate on **`test`**. Contrastive pretraining uses only **`pretrain`** (separate cohort in the paper).

## Reported metrics (paper / ICML)

- Retrieval Recall@10 vs 90k negatives: ~500–8000× random.
- Sleep staging macro AUPRC: **0.72** (LOO) vs **0.48** (supervised CNN).
- SDB AUPRC: **0.77** (LOO) vs **0.61** (CNN).

This workspace implements the same **algorithmic structure**; synthetic/demo data will not reproduce published numbers.
