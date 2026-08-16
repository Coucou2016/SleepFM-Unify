# SleepFM-Unify: Shared–Private Factorization for Multimodal Sleep Foundation Models

**Draft manuscript (methods paper)**  
**Status:** engineering-complete codebase + synthetic demos; **CinC / SHHS / MESA quantitative claims 待补充**  
**Target framing:** Nature Machine Intelligence–style methods article / ICML-compatible experimental spine  
**nature-skills:** `nature-writing` (methods + nat-mach-intell), `nature-figure` (Python / SciencePlots)  
**Code:** https://github.com/Coucou2016/SleepFM-Unify (`main`)

---

## Abstract

Polysomnography (PSG) yields heterogeneous brain, cardiac and respiratory streams that sleep foundation models such as SleepFM align with leave-one-out (LOO) contrastive learning. Pure cross-modal alignment can blur modality-specific structure and under-specify behaviour when channels are missing. We introduce **SleepFM-Unify**, a shared–private factorization on SleepFM 1D EfficientNet encoders: shared subspaces carry mixed LOO and pairwise InfoNCE, private subspaces stay out of contrastive alignment under an orthogonality penalty, and modality dropout with a `present_mask`-aware missingness term trains incomplete montages; an optional night-level temporal head contextualizes shared epoch sequences. We release configs, export/validation gates and a dual LOO–Unify paper suite. **Quantitative CinC/SHHS/MESA results remain 待补充** pending user DUA downloads; synthetic demos are engineering smoke tests (AUROC near chance) and are not scientific claims.

---

## Introduction

Sleep staging and sleep-disordered breathing (SDB) assessment rely on multimodal PSG. SleepFM showed that large-scale LOO contrastive pretraining across brain-activity (BAS), ECG and respiratory streams yields transferable embeddings for staging, SDB detection and cross-modal retrieval. Follow-on models emphasize scale and structure under montage heterogeneity: a Nature Medicine SleepFM line scales LOO-CL to disease-risk prediction with SHHS transfer; Omni-Sleep imposes a CNS/ANS physiological hierarchy; CIMSleepNet targets incomplete multimodal staging via modal imagination.

A remaining tension is that LOO alignment encourages factors that are *predictable across modalities*, which can suppress private cues that matter for unimodal or partially observed inference. SleepFM-Unify addresses this by **explicit shared–private factorization** on top of the existing SleepFM backbone, without replacing the paper LOO baseline (`unify.enabled: false`).

**Contributions (evidence-gated):**

1. Shared–private projection heads with mixed LOO + pairwise + orthogonality + miss (+ optional temporal) losses, contrastive terms on shared only.
2. Modality dropout training with `present_mask`-aware $\mathcal{L}_{\mathrm{miss}}$.
3. Evaluation honesty gates: channel meta fail-fast, CinC label coverage, apnea-epoch-rate vs clinical AHI wording, seeded RNG gallery caps.
4. Reproducible CLI paper suite for LOO vs Unify comparisons once real exports exist.

**Non-claims.** Synthetic AUROC≈0.5 is demo-only. Night `ahi_bin` uses apnea-epoch-rate cut-points, not clinical AASM AHI. Nature Medicine disease C-Index values are not our local results.

---

## Related work

**Sleep foundation models.** SleepFM (Thapa et al., ICML 2024; PMLR 235:48019–48037; arXiv:2405.17766) introduced multimodal leave-one-out contrastive learning (LOO-CL) on BAS/ECG/respiratory PSG clips and showed LOO outperforming pairwise alignment on staging, SDB detection and cross-modal retrieval (e.g. reported macro AUROC 0.88 vs CNN 0.72 for staging in that paper — **prior art only**). A later SleepFM line (Nature Medicine, doi:10.1038/s41591-025-04133-4; ~585k hours / ~65k participants; disease C-Index reporting) scales LOO-CL to future disease risk and SHHS transfer — **do not conflate** those clinical numbers with the ICML 2024 staging/retrieval setting, and do not paste them into our CinC/SHHS tables. Omni-Sleep (arXiv:2607.07720) uses CNS/ANS partitions with intra-/inter-system contrastive terms and latent temporal modeling. CIMSleepNet (NeurIPS 2024) targets incomplete multimodal staging via MAIM plus semantic/modal calibration contrastive learning.

**Shared–private multimodal learning.** Disentangling shared and modality-specific subspaces is a recurring theme (shared–private streams; low-rank shared/specific edits). Unify instantiates this idea *inside* SleepFM’s encoder interface rather than replacing the backbone or inventing a new physiological ontology: contrastive terms act only on the shared head, while private capacity remains for downstream concat and missing-modality regimes.

**Positioning.** Relative to SleepFM (flat LOO space), Omni-Sleep (CNS/ANS hierarchy) and CIMSleepNet (imagination under missingness), SleepFM-Unify is a **lightweight shared–private factorization + mixed loss + dropout/miss objective on SleepFM-compatible encoders**, designed for fair LOO-vs-Unify ablations once real PSG exports exist.

---

## Method

### Backbone (unchanged SleepFM)

Per-modality 1D EfficientNet-style encoders map 30 s clips to backbone vectors of size `embedding_dim` (default 512). Baseline contrastive mode is leave-one-out InfoNCE (`configs/default.yaml`, `unify.enabled: false`).

### Shared–private factorization

When Unify is enabled, each backbone vector is mapped by linear heads:

\[
z_m = [z_m^{\mathrm{shared}};\, z_m^{\mathrm{private}}],\qquad
\dim(z_m^{\mathrm{shared}})=\dim(z_m^{\mathrm{private}})=256
\]

by default, preserving a 512-d concat for downstream logistic heads comparable to SleepFM. **Assumption:** shared and private factors are approximately linearly separable in the backbone embedding; we do not claim nonlinear disentanglement completeness.

### Mixed objective

\[
\mathcal{L}=\lambda_{\mathrm{LOO}}\mathcal{L}_{\mathrm{LOO}}
+\lambda_{\mathrm{pair}}\mathcal{L}_{\mathrm{pair}}
+\lambda_{\mathrm{orth}}\mathrm{mean}\big((Z_s^\top Z_p)^2\big)
+\lambda_{\mathrm{miss}}\mathcal{L}_{\mathrm{miss}}
+\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}
\]

Default Unify weights (see `configs/unify.yaml`): LOO and pairwise on shared embeddings; private excluded from contrastive terms; orthogonality uses mean squared Gram entries after row centering / column normalization (`docs/UNIFY.md`); modality dropout drops one present modality; $\mathcal{L}_{\mathrm{miss}}$ is InfoNCE between the mean of remaining shared embeddings and the dropped modality’s shared embedding, masked by `present_mask`; optional temporal head (GRU/Transformer) on shared epoch sequences with adjacent-epoch contrastive and masked MSE (`configs/unify_temporal.yaml`).

### Downstream and retrieval

Default downstream space concatenates shared$\|$private per modality. Retrieval uses **shared** embeddings from the same checkpoint. Gallery caps use seeded RNG subsample (`limit_gallery(..., mode="rng")`), not a loader-order prefix.

### Honesty gates (part of the method interface)

- Channel meta fail-fast when official 5/1/3 montage weights meet schema 10/2/7 without explicit override.
- CinC label-coverage gate blocks degenerate staging/SDB claims when annotations are arousal-only.
- Night probes report **apnea_epoch_rate**, never clinical AHI unless a true AHI column is supplied.

---

## Experiments

### Datasets (access-gated)

| Dataset | Role | Status in this draft |
|---------|------|----------------------|
| Synthetic demo | CI / smoke | Present (`data/synthetic`) |
| CinC 2018 fixture | Schema export test | Present (`data/cinc2018_fixture`) |
| PhysioNet CinC 2018 | Paper pretrain/eval | **待补充** (DUA download; see `docs/DATA_ACCESS.md`) |
| NSRR SHHS / MESA | Transfer / night labels | **待补充** (DUA download) |

### Protocol (when data land)

1. `check_data_ready --stage raw` → export → `check_data_ready --stage pretrain` → validate.
2. Pretrain LOO baseline vs Unify (`configs/default.yaml` vs `configs/unify.yaml`).
3. Downstream staging / apnea probes, retrieval Recall@k, modality ablation, few-shot, night-level probes.
4. Optional temporal: `configs/unify_temporal.yaml` + `evaluate_night.py`.
5. Multi-seed uncertainty on real data (planned; **待补充**).

### Experiment matrix (planned)

| Experiment | Baseline | Unify | Metric | Status |
|------------|----------|-------|--------|--------|
| Staging (macro AUROC/AUPRC) | LOO | Full | Macro AUROC | **待补充** |
| SDB / apnea probe | LOO | Full | AUROC | **待补充** |
| Retrieval Recall@1/@5 | LOO | Shared | Recall@k | **待补充** |
| Modality dropout robustness | LOO | Full ± miss | ΔAUROC | **待补充** |
| Ablation −orth / −miss / +temporal | — | Variants | Same | **待补充** |
| Few-shot (k∈{1,2,4}) | LOO | Full | Macro AUROC | **待补充** |
| Night κ / apnea_epoch_rate bin | LOO pool | +temporal | κ, AUROC | **待补充** |

### Results

#### Table 1. Synthetic demo smoke metrics (NOT paper claims)

| Setting | Note | Staging macro AUROC | Retrieval |
|---------|------|---------------------|-----------|
| Synthetic LOO / Unify | Labels random-ish | ≈0.5 (chance) | Demo only |
| CinC 2018 | Real PSG | **待补充** | **待补充** |
| SHHS | Real PSG | **待补充** | **待补充** |

Loss curves in Fig. 2 are from a **local few-epoch Unify pretrain on synthetic data** (engineering verification). Orthogonality Gram heatmaps (Fig. 4) are forward-pass diagnostics on the same demo checkpoint. Ablation bars in Fig. 3 and modality-dropout curves in Fig. 5 are **schematic / chance-level** and must not be cited as method superiority.

#### Ablations (planned; numbers 待补充 on real data)

LOO baseline; Unify full; Unify − orth; Unify − miss; Unify + temporal; modality dropout robustness.

---

## Discussion

Unify keeps SleepFM’s LOO retrieval semantics in the shared space while retaining private capacity for downstream concat. Missing-modality training is first-class. Honesty constraints (label gates, AHI wording, channel meta) are part of the scientific interface: they prevent over-claiming when CinC arousal-only labels or montage mismatches appear.

**Failure modes.** (i) Montage mismatch without override → load fails by design. (ii) CinC staging claims without coverage → gate blocks. (iii) Interpreting synthetic AUROC≈0.5 as a positive result → rejected by caption policy. (iv) Calling night apnea-epoch-rate “AHI” → wording violation.

Limitations: no real CinC/SHHS numbers in this draft; night `ahi_bin` uses apnea-epoch-rate cut-points as placeholders; transfer claims bounded to evaluated montages once data exist.

---

## Methods (reproducibility)

- Package: `sleepfm/` (Python). Configs: `configs/unify.yaml`, `configs/unify_temporal.yaml`.
- Figures: `scripts/plot_unify_figures.py` with SciencePlots + Times New Roman / SimHei.
- Tests: `scripts/run_tests.py`, `scripts/smoke_test.py`.
- Paper suite: `scripts/run_paper_suite.py --demo` (synthetic) or `--data-dir` after export.
- Protocol inventory: `scripts/protocol_checklist.py`; access steps: `docs/DATA_ACCESS.md`.
- Compute note (demo): CPU or single GPU; synthetic few-epoch pretrain completes in minutes on a workstation. Full CinC/SHHS wall-clock **待补充**.

### Code / data availability

Public code/docs snapshot: [https://github.com/Coucou2016/SleepFM-Unify](https://github.com/Coucou2016/SleepFM-Unify) (`main`; large `data/` and `outputs/` excluded). PhysioNet/NSRR downloads require user accounts and DUAs. Synthetic and fixture datasets are generated in-repo.

---

## Figures

- **Fig. 1** Architecture (shared–private heads + mixed losses). Schematic; no clinical metrics.
- **Fig. 2** Synthetic Unify pretrain loss curves (demo only).
- **Fig. 3** Ablation schematic with chance baseline (demo numbers only; CinC/SHHS 待补充).
- **Fig. 4** Shared×private Gram diagnostic (demo forward pass).
- **Fig. 5** Modality-dropout robustness schematic (demo; 待补充 on real data).
- **Fig. 6** Experiment pipeline from raw PSG to LOO vs Unify evaluation.

---

## References (selected)

1. Thapa R. et al. SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals. *ICML* 2024; PMLR 235:48019–48037; arXiv:2405.17766.
2. A multimodal sleep foundation model for disease prediction. *Nature Medicine* (doi:10.1038/s41591-025-04133-4). (Distinct scale/task from [1]; do not conflate metrics.)
3. CIMSleepNet — Robust Sleep Staging over Incomplete Multimodal Physiological Signals via Contrastive Imagination. *NeurIPS* 2024; https://github.com/SQAIYY/CIMSleepNet.
4. Hou et al. Omni-Sleep: A Sleep Foundation Model via Hierarchical Contrastive Learning of CNS–ANS Dynamics. arXiv:2607.07720.
5. Shared–private multimodal factorization relatives (see `docs/reports/chatgpt-consultation-2026-08-16.md` and Round 1 notes).

---

## Assumptions / missing inputs

- Real CinC/SHHS/MESA metrics: **待补充** (no credentials / raw downloads on this host as of 2026-08-16).
- ChatGPT advisor pass: browser MCP blocked this turn; five documented substitute rounds in `docs/reports/rounds/`. Paste briefs: `docs/reports/chatgpt-paste-brief-2026-08-16.md`.
- Clinical AHI: not computed; use `apnea_epoch_rate` wording.
