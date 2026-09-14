# SleepFM-Unify: Robust Multimodal Pretraining under Heterogeneous and Missing PSG

**Draft manuscript (methods paper)**  
**Status:** engineering-complete codebase + synthetic demos; **CinC / SHHS / MESA quantitative claims 待补充** (see `docs/DEVELOPMENT_STATUS.md`). Formal Unify checkpoints must be **retrained after** raw-private VICReg + retrieval co-presence fixes before any real-data narrative.  
**Target framing:** Nature Machine Intelligence–style methods article / ICML-compatible experimental spine  
**nature-skills:** `nature-writing` (methods + nat-mach-intell), `nature-figure` (Python / SciencePlots)  
**Code:** https://github.com/Coucou2016/SleepFM-Unify (`main`)

---

## Abstract

Polysomnography (PSG) is routinely incomplete: montages differ across clinics, leads drop out overnight, and public corpora expose mismatched channel sets. Sleep foundation models such as SleepFM align brain, cardiac and respiratory streams with leave-one-out (LOO) contrastive learning, but pure cross-modal alignment under-specifies behaviour when modalities or channels are missing. We introduce **SleepFM-Unify**, a robustness layer on SleepFM 1D EfficientNet encoders for **heterogeneous and arbitrarily missing PSG**: sample-wise modality dropout, `present_mask`-aware encoding (no BatchNorm pollution), a missingness InfoNCE term gated by original presence, optional per-lead `channel_mask` zeroing, and mixed LOO/pairwise objectives on a shared subspace with private capacity retained for downstream probes. Shared–private heads are an implementation tool (FOCAL already factorizes shared/private multimodal time series); **they are not our novelty claim**. We release configs, export/validation gates and a dual LOO–Unify paper suite. **Quantitative CinC/SHHS/MESA results remain 待补充** pending user DUA downloads; synthetic demos are engineering smoke tests (AUROC near chance) and are not scientific claims.

---

## Introduction

Sleep staging and sleep-disordered breathing (SDB) assessment rely on multimodal PSG. SleepFM showed that large-scale LOO contrastive pretraining across brain-activity (BAS), ECG and respiratory streams yields transferable embeddings for staging, SDB detection and cross-modal retrieval. Follow-on models emphasize scale, structure and incomplete montages: a Nature Medicine SleepFM line scales LOO-CL to disease-risk prediction with SHHS transfer; PhysioOmni-style and Omni-Sleep lines impose physiological hierarchies; CIMSleepNet targets incomplete multimodal staging via modal imagination; OSF/SleepBench-style benchmarks stress standardized evaluation under montage heterogeneity.

A remaining systems gap is **SleepFM-compatible training and evaluation that remains well-defined when entire modalities or individual leads are absent**, without replacing the paper LOO baseline (`unify.enabled: false`). SleepFM-Unify addresses this with mask-aware encoding, missingness objectives and evaluation honesty gates—not by claiming a new shared–private factorization theory.

**Contributions (evidence-gated):**

1. Heterogeneous / missing-PSG training stack on SleepFM encoders: sample-wise modality dropout, mask-aware BN skip, $\mathcal{L}_{\mathrm{miss}}$ gated by original presence, optional lead masks.
2. Shared–private projection heads as a **tool** (mixed LOO + pairwise on shared; private for concat / space probes), with orthogonality + VICReg-style private hinge—explicitly positioned as FOCAL-adjacent, not novel factorization.
3. Evaluation honesty gates: channel meta fail-fast, CinC label coverage, apnea-epoch-rate vs clinical AHI wording, seeded RNG gallery caps, strict split isolation (`RuntimeError` on leak), few-shot mean±95% CI (≥10 participant-level repeats in paper mode).
4. Reproducible CLI paper suite for LOO vs Unify comparisons once real exports exist (shared/private/concat probes included).

**Non-claims.** Synthetic AUROC≈0.5 is demo-only. Night probes use continuous `apnea_positive_epoch_rate` (regression), not clinical AASM AHI and not 5/15/30 severity bins. Nature Medicine disease C-Index values are not our local results. We do **not** claim novelty for shared–private factorization per se.

---

## Related work

**Sleep foundation models.** SleepFM (Thapa et al., ICML 2024; PMLR 235:48019–48037; arXiv:2405.17766) introduced multimodal leave-one-out contrastive learning (LOO-CL) on BAS/ECG/respiratory PSG clips and showed LOO outperforming pairwise alignment on staging, SDB detection and cross-modal retrieval (e.g. reported macro AUROC 0.88 vs CNN 0.72 for staging in that paper — **prior art only**). A later SleepFM line (Nature Medicine, doi:10.1038/s41591-025-04133-4; ~585k hours / ~65k participants; disease C-Index reporting) scales LOO-CL to future disease risk and SHHS transfer — **do not conflate** those clinical numbers with the ICML 2024 staging/retrieval setting, and do not paste them into our CinC/SHHS tables.

**Incomplete multimodal sleep.** CIMSleepNet (NeurIPS 2024) targets incomplete multimodal staging via modal imagination (MAIM) plus semantic/modal calibration contrastive learning. SleepFM-Unify instead keeps SleepFM’s LOO interface and trains under dropout/masks without an imagination decoder.

**Physiological hierarchy / omnibus models.** Omni-Sleep (arXiv:2607.07720) uses CNS/ANS partitions with intra-/inter-system contrastive terms. PhysioOmni-style work similarly emphasizes structured physiological priors across biosignals. Unify does not introduce a new ontology; it hardens SleepFM under missingness.

**Shared–private multimodal learning.** FOCAL and related shared–private factorizations for multimodal time series already separate aligned and modality-specific subspaces (often with orthogonality). Unify **instantiates** shared/private heads inside SleepFM’s encoder interface so contrastive terms can stay on shared embeddings while private capacity remains for downstream concat and missing-modality regimes. **Novelty is not “we invented shared–private.”**

**Benchmarks.** OSF / SleepBench-style efforts standardize sleep foundation evaluation under heterogeneous setups. Our contribution is a reproducible SleepFM-compatible training/eval stack with honesty gates; we do not replace SleepBench.

**Positioning (one sentence).** Relative to SleepFM (flat LOO), Nature Med SleepFM (scale + disease risk), CIMSleepNet (imagination under missingness), PhysioOmni/Omni-Sleep (physiological hierarchy), FOCAL (shared/private factorization), and OSF/SleepBench (benchmarking), SleepFM-Unify is a **lightweight missing/heterogeneous-PSG robustness layer on SleepFM-compatible encoders**, designed for fair LOO-vs-Unify ablations once real PSG exports exist.

---

## Method

### Backbone (unchanged SleepFM)

Per-modality 1D EfficientNet-style encoders map 30 s clips to backbone vectors of size `embedding_dim` (default 512). Baseline contrastive mode is leave-one-out InfoNCE (`configs/default.yaml`, `unify.enabled: false`).

### Missing / heterogeneous PSG stack

- **Modality dropout:** default sample-wise random non-empty subsets (`modality_dropout_mode: sample`); batch drop-one remains available.
- **Mask-aware encode:** rows with `present_mask=0` skip the encoder (no BatchNorm pollution); missing-modality embeddings stay exactly zero.
- **$\mathcal{L}_{\mathrm{miss}}$:** InfoNCE(remaining shared mean, dropped shared) only where the dropped modality was **originally present**.
- **Channel mask (partial):** dataset → collate → encode zeros padded/absent leads before the CNN. **Done:** lead zeroing + fixed montage schema (10/2/7 with documented pads). **TODO:** true variable-channel encoders / mask-aware channel attention beyond zero-fill.

### Shared–private heads (tool, not claim)

When Unify is enabled, each backbone vector is mapped by linear heads:

\[
z_m = [z_m^{\mathrm{shared}};\, z_m^{\mathrm{private}}],\qquad
\dim(z_m^{\mathrm{shared}})=\dim(z_m^{\mathrm{private}})=256
\]

by default, preserving a 512-d concat for downstream logistic heads comparable to SleepFM. Contrastive terms act on **L2-normalized shared** only; private stays for concat / `shared|private` space probes. Orthogonality: center columns, column-L2 normalize, then $\mathrm{mean}((S^\top P)^2)$. VICReg-style private variance hinge runs on **raw** (pre-L2) private projections — not on unit vectors. Optional temporal head (GRU/Transformer) on `masked_modality_mean` of shared epoch sequences.

### Mixed objective

\[
\mathcal{L}=\lambda_{\mathrm{LOO}}\mathcal{L}_{\mathrm{LOO}}
+\lambda_{\mathrm{pair}}\mathcal{L}_{\mathrm{pair}}
+\lambda_{\mathrm{priv}}\mathcal{L}_{\mathrm{private}}
+\lambda_{\mathrm{orth}}\mathcal{L}_{\mathrm{orth}}
+\lambda_{\mathrm{miss}}\mathcal{L}_{\mathrm{miss}}
+\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}
\]

Default Unify weights: see `configs/unify.yaml` and `docs/UNIFY.md`.

### Downstream and retrieval

Default downstream space concatenates shared$\|$private per modality. CLI / paper suite also probe **shared-only** and **private-only**. Retrieval uses **shared** embeddings from the same checkpoint. Gallery caps use seeded RNG subsample (`limit_gallery(..., mode="rng")`), not a loader-order prefix.

### Few-shot protocol

Participant-level $k$-shot linear probes with **≥10 repeats** in paper mode (`scripts/eval_fewshot.py`, `run_paper_suite.py`); report **mean±95% CI** (Student-$t$ on repeat means). Demo/CI may use 2 repeats.

### Honesty gates (part of the method interface)

- Channel meta fail-fast when official 5/1/3 montage weights meet schema 10/2/7 without explicit override.
- CinC label-coverage gate blocks degenerate staging/SDB claims when annotations are arousal-only.
- Night probes report **apnea_epoch_rate**, never clinical AHI unless a true AHI column is supplied.
- Split isolation: all pretrain/valid/train/test pairs disjoint on path, participant, and night/recording ids — leaks raise `RuntimeError`.

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

1. `check_data_ready --stage raw` → export → `check_data_ready --stage pretrain` → validate (strict isolation).
2. Pretrain LOO baseline vs Unify (`configs/default.yaml` vs `configs/unify.yaml`).
3. Downstream staging / apnea probes (concat / shared / private), retrieval Recall@k, modality ablation, few-shot (≥10 repeats, mean±95% CI), night-level probes.
4. Optional temporal: `configs/unify_temporal.yaml` + `evaluate_night.py`.
5. Multi-seed uncertainty on real data (planned; **待补充**).

### Experiment matrix (planned)

| Experiment | Baseline | Unify | Metric | Status |
|------------|----------|-------|--------|--------|
| Staging (macro AUROC/AUPRC) | LOO | Full | Macro AUROC | **待补充** |
| SDB / apnea probe | LOO | Full | AUROC | **待补充** |
| Retrieval Recall@1/@5 | LOO | Shared | Recall@k | **待补充** |
| Modality dropout robustness | LOO | Full ± miss | ΔAUROC | **待补充** |
| Space probe (concat / shared / private) | — | Unify | Macro AUROC | **待补充** |
| Ablation −orth / −miss / +temporal | — | Variants | Same | **待补充** |
| Few-shot (k∈{1,2,4}, ≥10 repeats) | LOO | Full | Macro AUROC mean±95% CI | **待补充** |
| Night κ / apnea_positive_epoch_rate | LOO pool | +temporal | κ, R²/MAE | **待补充** |

### Results

#### Table 1. Synthetic demo smoke metrics (NOT paper claims)

| Setting | Note | Staging macro AUROC | Retrieval |
|---------|------|---------------------|-----------|
| Synthetic LOO / Unify | Labels random-ish | ≈0.5 (chance) | Demo only |
| CinC 2018 | Real PSG | **待补充** | **待补充** |
| SHHS | Real PSG | **待补充** | **待补充** |

Loss curves in Fig. 2 are from a **local few-epoch Unify pretrain on synthetic data** (engineering verification). Orthogonality Gram heatmaps (Fig. 4) are forward-pass diagnostics on the same demo checkpoint. Ablation bars in Fig. 3 and modality-dropout curves in Fig. 5 are **schematic / chance-level** and must not be cited as method superiority.

#### Ablations (planned; numbers 待补充 on real data)

LOO baseline; Unify full; Unify − orth; Unify − miss; Unify + temporal; space probes; modality dropout robustness.

---

## Discussion

Unify keeps SleepFM’s LOO retrieval semantics in the shared space while retaining private capacity for downstream concat, and makes missing-modality / heterogeneous-montage training first-class. Honesty constraints (label gates, AHI wording, channel meta, strict isolation) are part of the scientific interface.

**Failure modes.** (i) Montage mismatch without override → load fails by design. (ii) CinC staging claims without coverage → gate blocks. (iii) Interpreting synthetic AUROC≈0.5 as a positive result → rejected by caption policy. (iv) Calling night apnea-epoch-rate “AHI” → wording violation. (v) Claiming novel shared–private factorization → rejected by Related work positioning.

Limitations: no real CinC/SHHS numbers in this draft; channel-aware pooling beyond lead zeroing is stub-only; night severity uses continuous apnea-epoch-rate (not clinical AHI bins); transfer claims bounded to evaluated montages once data exist; strong public baselines (CIMSleepNet, FOCAL ports, full SleepBench) not yet run locally. Formal checkpoints must be retrained after raw-private VICReg + retrieval co-presence fixes (see `docs/DEVELOPMENT_STATUS.md`).

---

## Methods (reproducibility)

- Package: `sleepfm/` (Python). Configs: `configs/unify.yaml`, `configs/unify_temporal.yaml`.
- Figures: `scripts/plot_unify_figures.py` with SciencePlots + Times New Roman / SimHei.
- Tests: `scripts/run_tests.py`, `scripts/smoke_test.py`.
- Paper suite: `scripts/run_paper_suite.py --demo` (synthetic) or `--data-dir` after export; paper few-shot defaults to 10 repeats; `--space-probe` for shared/private/concat.
- Space probe CLI: `scripts/eval_space_probe.py`.
- Protocol inventory: `scripts/protocol_checklist.py`; access steps: `docs/DATA_ACCESS.md`.
- Compute note (demo): CPU or single GPU; synthetic few-epoch pretrain completes in minutes on a workstation. Full CinC/SHHS wall-clock **待补充**.

### Code / data availability

Public code/docs snapshot: [https://github.com/Coucou2016/SleepFM-Unify](https://github.com/Coucou2016/SleepFM-Unify) (`main`; large `data/` and `outputs/` excluded). PhysioNet/NSRR downloads require user accounts and DUAs. Synthetic and fixture datasets are generated in-repo. License: MIT (see `LICENSE`); cite via `CITATION.cff`.

---

## Figures

- **Fig. 1** Architecture (missing-PSG stack + shared–private tool heads). Schematic; no clinical metrics.
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
5. FOCAL and shared–private multimodal factorization relatives (shared/private + orthogonality for multimodal time series; see literature notes in `docs/reports/`).
6. OSF / SleepBench-style sleep foundation benchmarking efforts (heterogeneous evaluation; not replaced here).

---

## Assumptions / missing inputs

- Real CinC/SHHS/MESA metrics: **待补充** (no credentials / raw downloads on this host as of 2026-09-14).
- Clinical AHI: not computed; use `apnea_epoch_rate` wording.
- Strong external baselines (CIMSleepNet, FOCAL ports, SleepBench): **待补充**.
