# SleepFM-Unify: Shared–Private Factorization for Multimodal Sleep Foundation Models

**Draft manuscript (methods paper)**  
**Status:** engineering-complete codebase + synthetic demos; **CinC / SHHS / MESA quantitative claims 待补充**  
**Target framing:** Nature Machine Intelligence–style methods article / ICML-compatible experimental spine  
**nature-skills:** `nature-writing` (methods + nat-mach-intell), `nature-figure` (Python / SciencePlots)

---

## Abstract

Polysomnography (PSG) records heterogeneous brain, cardiac and respiratory signals that sleep foundation models such as SleepFM align with leave-one-out (LOO) contrastive learning. Pure cross-modal alignment can blur modality-specific structure and under-specify behaviour when channels are missing. We present **SleepFM-Unify**, a shared–private factorization layered on SleepFM encoders: each modality yields a shared subspace used for LOO/pairwise InfoNCE and a private subspace kept out of contrastive alignment, regularized by orthogonality, modality dropout and a missing-modality reconstruction term, with an optional night-level temporal head. We release configs, export/validation gates, and a paper experiment suite. **Quantitative CinC/SHHS results are 待补充** pending user DUA downloads; synthetic demos are reported only as engineering smoke tests (AUROC near chance) and are not scientific claims.

---

## Introduction

Sleep staging and sleep-disordered breathing assessment rely on multimodal PSG. SleepFM showed that large-scale LOO contrastive pretraining across brain-activity (BAS), ECG and respiratory streams yields transferable embeddings for staging, SDB detection and cross-modal retrieval. Follow-on foundation models (e.g. Nature Medicine–scale SleepFM; Omni-Sleep’s CNS/ANS hierarchy; CIMSleepNet’s missing-modality imagination) emphasize robustness under montage heterogeneity and incomplete inputs.

A remaining tension is that LOO alignment encourages factors that are *predictable across modalities*, which can suppress private cues that matter for unimodal or partially observed inference. SleepFM-Unify addresses this by **explicit shared–private factorization** on top of the existing SleepFM backbone, without replacing the paper LOO baseline (`unify.enabled: false`).

**Contributions (bounded):**

1. Shared–private projection heads with mixed LOO + pairwise + orthogonality + miss (+ optional temporal) losses.
2. Modality dropout training with `present_mask`-aware $\mathcal{L}_{\mathrm{miss}}$.
3. Evaluation honesty gates: channel meta fail-fast, CinC label coverage, apnea-epoch-rate vs clinical AHI wording, seeded RNG gallery caps.
4. Reproducible CLI paper suite for LOO vs Unify comparisons once real data are present.

---

## Related work

**Sleep foundation models.** SleepFM (Thapa et al., ICML 2024; arXiv:2405.17766) introduced multimodal leave-one-out contrastive learning (LOO-CL) on BAS/ECG/respiratory PSG clips and showed LOO outperforming pairwise alignment on staging, SDB detection and cross-modal retrieval. A later SleepFM line (Nature Medicine, doi:10.1038/s41591-025-04133-4; medRxiv 2025.02.04.25321675) scaled LOO-CL to disease-risk prediction with SHHS transfer — **do not conflate those clinical C-Index numbers with the ICML 2024 staging/retrieval setting**, and do not paste them into our CinC/SHHS tables until we re-run locally. Omni-Sleep (arXiv:2607.07720) imposes a CNS/ANS physiological hierarchy with intra-/inter-system contrastive terms and latent temporal modeling. CIMSleepNet (NeurIPS 2024) targets incomplete multimodal staging via modal imagination (MAIM) plus semantic/modal calibration contrastive learning.

**Shared–private multimodal learning.** Disentangling shared and modality-specific subspaces is a recurring theme (shared–private streams; low-rank shared/specific edits). Unify instantiates this idea *inside* SleepFM’s encoder interface rather than replacing the backbone or inventing a new physiological ontology: contrastive terms act only on the shared head, while private capacity remains for downstream concat and missing-modality regimes.

**Positioning one-liner.** Relative to SleepFM (flat LOO space), Omni-Sleep (CNS/ANS hierarchy) and CIMSleepNet (imagination under missingness), SleepFM-Unify is a **lightweight shared–private factorization + mixed loss + dropout/miss objective on frozen SleepFM-compatible encoders**, designed for fair LOO-vs-Unify ablations once real PSG exports exist.

---

## Method

### Backbone (unchanged SleepFM)

Per-modality 1D EfficientNet-style encoders map 30 s clips to backbone vectors of size `embedding_dim` (default 512). Baseline contrastive mode is leave-one-out InfoNCE.

### Shared–private factorization

When Unify is enabled, each backbone vector is mapped by linear heads:

\[
z_m = [z_m^{\mathrm{shared}};\, z_m^{\mathrm{private}}],\qquad
\dim(z_m^{\mathrm{shared}})=\dim(z_m^{\mathrm{private}})=256
\]

by default, preserving a 512-d concat for downstream logistic heads comparable to SleepFM.

### Mixed objective

\[
\mathcal{L}=\lambda_{\mathrm{LOO}}\mathcal{L}_{\mathrm{LOO}}
+\lambda_{\mathrm{pair}}\mathcal{L}_{\mathrm{pair}}
+\lambda_{\mathrm{orth}}\mathrm{mean}\big((Z_s^\top Z_p)^2\big)
+\lambda_{\mathrm{miss}}\mathcal{L}_{\mathrm{miss}}
+\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}
\]

- Shared embeddings enter LOO and pairwise InfoNCE.
- Private embeddings are excluded from contrastive terms.
- Orthogonality uses mean squared Gram entries after row centering / column normalization (see `docs/UNIFY.md`).
- Modality dropout drops one present modality; $\mathcal{L}_{\mathrm{miss}}$ is InfoNCE between the mean of remaining shared embeddings and the dropped modality’s shared embedding, masked by `present_mask`.
- Optional temporal head (GRU/Transformer) operates on shared epoch sequences with adjacent-epoch contrastive and masked MSE.

### Downstream and retrieval

Default downstream space concatenates shared$\|$private per modality. Retrieval uses **shared** embeddings from the same checkpoint. Gallery caps use seeded RNG subsample (`limit_gallery(..., mode="rng")`), not a loader-order prefix.

---

## Experiments

### Datasets (access-gated)

| Dataset | Role | Status in this draft |
|---------|------|----------------------|
| Synthetic demo | CI / smoke | Present (`data/synthetic`) |
| CinC 2018 fixture | Schema export test | Present (`data/cinc2018_fixture`) |
| PhysioNet CinC 2018 | Paper pretrain/eval | **待补充** (DUA download) |
| NSRR SHHS / MESA | Transfer / night labels | **待补充** (DUA download) |

### Protocol (when data land)

1. `check_data_ready --stage raw` → export → `check_data_ready --stage pretrain` → validate.
2. Pretrain LOO baseline vs Unify (`configs/default.yaml` vs `configs/unify.yaml`).
3. Downstream staging / apnea probes, retrieval Recall@k, modality ablation, few-shot, night-level probes.
4. Label coverage gate blocks degenerate CinC staging/SDB claims; channel meta blocks 5/1/3 vs 10/2/7 silent mismatch.

### Results

#### Table 1. Synthetic demo smoke metrics (NOT paper claims)

| Setting | Note | Staging macro AUROC | Retrieval |
|---------|------|---------------------|-----------|
| Synthetic LOO / Unify | Labels random-ish | ≈0.5 (chance) | Demo only |
| CinC 2018 | Real PSG | **待补充** | **待补充** |
| SHHS | Real PSG | **待补充** | **待补充** |

Loss curves in Fig. 2 are from a **local 5-epoch Unify pretrain on synthetic data** (engineering verification). Orthogonality Gram heatmaps (Fig. 4) are forward-pass diagnostics on the same demo checkpoint.

#### Ablations (planned; numbers 待补充 on real data)

LOO baseline; Unify full; Unify − orth; Unify − miss; Unify + temporal; modality dropout robustness.

---

## Discussion

Unify is designed to keep SleepFM’s LOO retrieval semantics in the shared space while retaining private capacity for downstream concat. Missing-modality training is first-class. Honesty constraints (label gates, AHI wording, channel meta) are part of the scientific interface: they prevent over-claiming when CinC arousal-only labels or montage mismatches appear.

Limitations: no real CinC/SHHS numbers in this draft; night `ahi_bin` uses apnea-epoch-rate cut-points as placeholders; synthetic metrics near chance must not be cited as method superiority.

---

## Methods (reproducibility)

- Package: `sleepfm/` (Python). Configs: `configs/unify.yaml`, `configs/unify_temporal.yaml`.
- Figures: `scripts/plot_unify_figures.py` with SciencePlots + Times New Roman / SimHei.
- Tests: `scripts/run_tests.py`, `scripts/smoke_test.py`.
- Paper suite: `scripts/run_paper_suite.py --demo` (synthetic) or `--data-dir` after export.

### Code / data availability

Public code/docs snapshot: [https://github.com/Coucou2016/SleepFM-Unify](https://github.com/Coucou2016/SleepFM-Unify) (`main`; large `data/` and `outputs/` excluded). PhysioNet/NSRR downloads require user accounts and DUAs. Synthetic and fixture datasets are generated in-repo.

---

## Figures

- **Fig. 1** Architecture (shared–private heads + mixed losses).
- **Fig. 2** Synthetic Unify pretrain loss curves (demo).
- **Fig. 3** Ablation schematic with chance baseline (demo numbers only).
- **Fig. 4** Shared×private Gram diagnostic (demo forward pass).
- **Fig. 5** Modality-dropout robustness schematic (demo).
- **Fig. 6** Experiment pipeline.

---

## References (selected)

1. Thapa R. et al. SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals. *ICML* 2024; PMLR 235:48019–48037; arXiv:2405.17766.
2. A multimodal sleep foundation model for disease prediction. *Nature Medicine* (doi:10.1038/s41591-025-04133-4); preprint medRxiv 10.1101/2025.02.04.25321675. (Distinct scale/task from [1].)
3. CIMSleepNet — Robust Sleep Staging over Incomplete Multimodal Physiological Signals via Contrastive Imagination. *NeurIPS* 2024; https://github.com/SQAIYY/CIMSleepNet.
4. Hou et al. Omni-Sleep: A Sleep Foundation Model via Hierarchical Contrastive Learning of CNS–ANS Dynamics. arXiv:2607.07720.
5. Shared–private multimodal factorization relatives (see `docs/reports/chatgpt-consultation-2026-08-16.md`).

---

## Assumptions / missing inputs

- Real CinC/SHHS/MESA metrics: **待补充**.
- ChatGPT advisor pass: browser MCP still blocked this turn; literature independently verified via WebSearch + nature-skills. Paste brief for ChatGPT: `docs/reports/chatgpt-paste-brief-2026-08-16.md`.
- Clinical AHI: not computed; use `apnea_epoch_rate` wording.
