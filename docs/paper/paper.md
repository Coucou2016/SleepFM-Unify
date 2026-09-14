# SleepFM-Unify: Robust Multimodal Pretraining under Heterogeneous and Missing PSG

**Draft manuscript (methods paper)**  
**Status:** Methods and software ready; real-cohort quantitative results pending data access and post-correctness retrain (limitations below).  
**Code:** https://github.com/Coucou2016/SleepFM-Unify (`main`)

---

## Abstract

Polysomnography (PSG) is routinely incomplete: montages differ across clinics, leads drop out overnight, and public corpora expose mismatched channel sets. Sleep foundation models such as SleepFM align brain, cardiac and respiratory streams with leave-one-out (LOO) contrastive learning, but pure cross-modal alignment under-specifies behaviour when modalities or channels are missing. We introduce **SleepFM-Unify**, a robustness layer on SleepFM 1D EfficientNet encoders for **heterogeneous and arbitrarily missing PSG**: sample-wise modality dropout, presence-mask–aware encoding (no BatchNorm pollution), a missingness InfoNCE term gated by original presence, optional per-lead channel masks, and mixed LOO/pairwise objectives on a shared subspace with private capacity retained for downstream probes. Shared–private heads are an implementation tool (FOCAL already factorizes shared/private multimodal time series); **they are not our novelty claim**. We release configs, export/validation gates and a dual LOO–Unify evaluation suite. Quantitative results on CinC, SHHS and MESA are left for a data-complete follow-up; synthetic demos verify engineering only and are not scientific claims.

---

## Introduction

Sleep staging and sleep-disordered breathing (SDB) assessment rely on multimodal PSG. SleepFM showed that large-scale LOO contrastive pretraining across brain-activity (BAS), ECG and respiratory streams yields transferable embeddings for staging, SDB detection and cross-modal retrieval. Follow-on models emphasize scale, structure and incomplete montages: a Nature Medicine SleepFM line scales LOO-CL to disease-risk prediction with SHHS transfer; PhysioOmni-style and Omni-Sleep lines impose physiological hierarchies; CIMSleepNet targets incomplete multimodal staging via modal imagination; OSF/SleepBench-style benchmarks stress standardized evaluation under montage heterogeneity.

A remaining systems gap is **SleepFM-compatible training and evaluation that remains well-defined when entire modalities or individual leads are absent**, without replacing the paper LOO baseline. SleepFM-Unify addresses this with mask-aware encoding, missingness objectives and evaluation honesty constraints—not by claiming a new shared–private factorization theory.

**Contributions (evidence-gated):**

1. Heterogeneous / missing-PSG training stack on SleepFM encoders: sample-wise modality dropout, mask-aware BN skip, missingness InfoNCE gated by original presence, optional lead masks.
2. Shared–private projection heads as a **tool** (mixed LOO + pairwise on shared; private for concat / space probes), with orthogonality + VICReg-style private regularization on raw private embeddings—explicitly positioned as FOCAL-adjacent, not novel factorization.
3. Evaluation constraints that keep claims falsifiable: channel-montage fail-fast, CinC label-coverage gates, apnea-epoch-rate vs clinical AHI wording, seeded gallery caps, strict split isolation, few-shot mean±95% CI (≥10 participant-level repeats in paper mode).
4. A reproducible CLI suite for LOO vs Unify comparisons once real exports exist (shared/private/concat probes included).

**Non-claims.** Synthetic near-chance AUROC is demo-only. Night probes use continuous apnea-positive epoch rate (regression), not clinical AASM AHI and not severity bins. Nature Medicine disease C-Index values are not our local results. We do **not** claim novelty for shared–private factorization per se.

---

## Related work

**Sleep foundation models.** SleepFM (Thapa et al., ICML 2024; PMLR 235:48019–48037; arXiv:2405.17766) introduced multimodal leave-one-out contrastive learning (LOO-CL) on BAS/ECG/respiratory PSG clips and showed LOO outperforming pairwise alignment on staging, SDB detection and cross-modal retrieval. A later SleepFM line (Nature Medicine, doi:10.1038/s41591-025-04133-4) scales LOO-CL to future disease risk and SHHS transfer — **do not conflate** those clinical numbers with the ICML 2024 staging/retrieval setting.

**Incomplete multimodal sleep.** CIMSleepNet (NeurIPS 2024) targets incomplete multimodal staging via modal imagination plus semantic/modal calibration contrastive learning. SleepFM-Unify instead keeps SleepFM’s LOO interface and trains under dropout/masks without an imagination decoder.

**Physiological hierarchy / omnibus models.** Omni-Sleep (arXiv:2607.07720) uses CNS/ANS partitions with intra-/inter-system contrastive terms. PhysioOmni-style work similarly emphasizes structured physiological priors. Unify does not introduce a new ontology; it hardens SleepFM under missingness.

**Shared–private multimodal learning.** FOCAL and related shared–private factorizations for multimodal time series already separate aligned and modality-specific subspaces (often with orthogonality). Unify **instantiates** shared/private heads inside SleepFM’s encoder interface so contrastive terms can stay on shared embeddings while private capacity remains for downstream concat and missing-modality regimes. **Novelty is not “we invented shared–private.”**

**Benchmarks.** OSF / SleepBench-style efforts standardize sleep foundation evaluation under heterogeneous setups. Our contribution is a reproducible SleepFM-compatible training/eval stack with claim discipline; we do not replace SleepBench.

**Positioning.** Relative to SleepFM (flat LOO), Nature Med SleepFM (scale + disease risk), CIMSleepNet (imagination under missingness), PhysioOmni/Omni-Sleep (physiological hierarchy), FOCAL (shared/private factorization), and OSF/SleepBench (benchmarking), SleepFM-Unify is a **lightweight missing/heterogeneous-PSG robustness layer on SleepFM-compatible encoders**, designed for fair LOO-vs-Unify ablations once real PSG exports exist.

---

## Method

### Backbone (unchanged SleepFM)

Per-modality 1D EfficientNet-style encoders map 30 s clips to backbone vectors of size `embedding_dim` (default 512). Baseline contrastive mode is leave-one-out InfoNCE with Unify disabled.

### Missing / heterogeneous PSG stack

- **Modality dropout:** default sample-wise random non-empty **proper** subsets (full set excluded when two or more modalities are present), so the dropout probability is an actual corruption rate; batch drop-one remains available.
- **Mask-aware encode:** absent modalities skip the encoder (no BatchNorm pollution); missing-modality embeddings stay exactly zero.
- **Missingness loss:** InfoNCE(remaining shared mean, dropped shared) only where the dropped modality was **originally present**.
- **Channel mask (partial):** padded/absent leads are zeroed before the CNN. True variable-channel encoders / mask-aware channel attention beyond zero-fill remain future work.

### Shared–private heads (tool, not claim)

When Unify is enabled, each backbone vector is mapped by linear heads:

\[
z_m = [z_m^{\mathrm{shared}};\, z_m^{\mathrm{private}}],\qquad
\dim(z_m^{\mathrm{shared}})=\dim(z_m^{\mathrm{private}})=256
\]

by default, preserving a 512-d concat for downstream logistic heads comparable to SleepFM. Contrastive terms act on **L2-normalized shared** only. Orthogonality: center columns, column-L2 normalize, then $\mathrm{mean}((S^\top P)^2)$. VICReg-style private variance (and optional covariance) runs on **raw** private projections — not on unit vectors. Optional temporal head (GRU/Transformer) averages present shared epoch embeddings with a masked mean.

### Mixed objective

\[
\mathcal{L}=\lambda_{\mathrm{LOO}}\mathcal{L}_{\mathrm{LOO}}
+\lambda_{\mathrm{pair}}\mathcal{L}_{\mathrm{pair}}
+\lambda_{\mathrm{priv}}\mathcal{L}_{\mathrm{private}}
+\lambda_{\mathrm{orth}}\mathcal{L}_{\mathrm{orth}}
+\lambda_{\mathrm{miss}}\mathcal{L}_{\mathrm{miss}}
+\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}
\]

Default Unify weights are given in the accompanying configs and method notes.

### Downstream and retrieval

Default downstream space concatenates shared$\|$private per modality; probes also evaluate shared-only and private-only. Retrieval uses **shared** embeddings. For each directed pair A→B, only samples where **both** modalities are present enter the gallery/query; the random Recall@$k$ baseline is $k/N_{\mathrm{pair}}$. Gallery caps use seeded RNG subsampling.

### Few-shot protocol

Participant-level $k$-shot linear probes with **≥10 repeats** in paper mode; report **mean±95% CI** (Student-$t$ on repeat means).

### Claim discipline (part of the method interface)

- Channel montage fail-fast when incompatible official weights meet the fixed schema without an explicit override.
- CinC label-coverage gate blocks degenerate staging/SDB claims when annotations are arousal-only.
- Night probes report apnea-positive epoch rate, never clinical AHI unless a true AHI column is supplied.
- Split isolation: path and participant ids must be disjoint across pretrain/valid/train/test; leaks raise errors. Absent night ids are marked not-applicable rather than treated as proven isolation.

---

## Experiments

### Datasets

| Dataset | Role | Availability in this draft |
|---------|------|----------------------------|
| Synthetic demo | Engineering smoke | Released with code |
| CinC 2018 fixture | Schema export test | Released with code |
| PhysioNet CinC 2018 | Paper pretrain/eval | Pending user download |
| NSRR SHHS / MESA | Transfer / night labels | Pending DUA download |

### Protocol (when data land)

1. Export and validate with strict split isolation.
2. Pretrain LOO baseline vs Unify.
3. Downstream staging / apnea probes (concat / shared / private), co-presence retrieval, modality ablation, few-shot (≥10 repeats), night-level probes.
4. Optional temporal night head.
5. Multi-seed uncertainty on real data (planned).

### Experiment matrix (planned)

| Experiment | Baseline | Unify | Metric | Status |
|------------|----------|-------|--------|--------|
| Staging (macro AUROC/AUPRC) | LOO | Full | Macro AUROC | Pending real data |
| SDB / apnea probe | LOO | Full | AUROC | Pending real data |
| Retrieval Recall@1/@5 | LOO | Shared | Directed Recall@$k$ | Pending real data |
| Modality dropout robustness | LOO | Full ± miss | ΔAUROC | Pending real data |
| Space probe (concat / shared / private) | — | Unify | Macro AUROC | Pending real data |
| Ablation −orth / −miss / +temporal | — | Variants | Same | Pending real data |
| Few-shot (k∈{1,2,4}, ≥10 repeats) | LOO | Full | Macro AUROC mean±95% CI | Pending real data |
| Night κ / apnea-positive epoch rate | LOO pool | +temporal | κ, R²/MAE | Pending real data |

### Results

Synthetic demos produce near-chance staging AUROC and are reported only as engineering verification. Real CinC / SHHS / MESA numbers are not filled in this draft. Formal Unify checkpoints used for any future real-data narrative must be trained after the raw-private VICReg and retrieval co-presence correctness fixes described in the method notes.

#### Ablations (planned)

LOO baseline; Unify full; Unify − orth; Unify − miss; Unify + temporal; space probes; modality dropout robustness.

---

## Discussion

Unify keeps SleepFM’s LOO retrieval semantics in the shared space while retaining private capacity for downstream concat, and makes missing-modality / heterogeneous-montage training first-class. Claim discipline (label gates, AHI wording, channel meta, strict isolation) is part of the scientific interface.

**Failure modes.** (i) Montage mismatch without override → load fails by design. (ii) CinC staging claims without coverage → gate blocks. (iii) Interpreting synthetic near-chance AUROC as a positive result → rejected by caption policy. (iv) Calling night apnea-epoch-rate “AHI” → wording violation. (v) Claiming novel shared–private factorization → rejected by Related work positioning.

**Limitations.** No real CinC/SHHS numbers in this draft; channel-aware pooling beyond lead zeroing is incomplete; night severity uses continuous apnea-epoch-rate (not clinical AHI); transfer claims are bounded to evaluated montages once data exist; strong public baselines (CIMSleepNet, FOCAL ports, full SleepBench) are not yet run locally.

---

## Methods (reproducibility)

- Package: `sleepfm/` (Python). Configs for Unify and optional temporal heads are shipped with the repository.
- Figures: SciencePlots-compatible plotting scripts (optional `[paper]` extra).
- Tests and smoke scripts ship with the repository.
- Dual LOO–Unify paper suite CLI supports synthetic demos and real `data_dir` exports; paper few-shot defaults to 10 repeats with optional shared/private/concat probes.
- Compute note (demo): CPU or single GPU; synthetic few-epoch pretrain completes in minutes. Full CinC/SHHS wall-clock depends on hardware and download size.

### Code / data availability

Public code/docs: [https://github.com/Coucou2016/SleepFM-Unify](https://github.com/Coucou2016/SleepFM-Unify) (`main`; large `data/` and `outputs/` excluded). PhysioNet/NSRR downloads require user accounts and DUAs. Synthetic and fixture datasets are generated in-repo. License: MIT (see `LICENSE`); third-party / clinical checkpoint terms: `THIRD_PARTY_NOTICES.md`; cite via `CITATION.cff`.

---

## Figures

- **Fig. 1** Architecture (missing-PSG stack + shared–private tool heads). Schematic; no clinical metrics.
- **Fig. 2** Synthetic Unify pretrain loss curves (demo only).
- **Fig. 3** Ablation schematic with chance baseline (demo numbers only).
- **Fig. 4** Shared×private Gram diagnostic (demo forward pass).
- **Fig. 5** Modality-dropout robustness schematic (demo).
- **Fig. 6** Experiment pipeline from raw PSG to LOO vs Unify evaluation.

---

## References (selected)

1. Thapa R. et al. SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals. *ICML* 2024; PMLR 235:48019–48037; arXiv:2405.17766.
2. A multimodal sleep foundation model for disease prediction. *Nature Medicine* (doi:10.1038/s41591-025-04133-4). (Distinct scale/task from [1]; do not conflate metrics.)
3. CIMSleepNet — Robust Sleep Staging over Incomplete Multimodal Physiological Signals via Contrastive Imagination. *NeurIPS* 2024; https://github.com/SQAIYY/CIMSleepNet.
4. Hou et al. Omni-Sleep: A Sleep Foundation Model via Hierarchical Contrastive Learning of CNS–ANS Dynamics. arXiv:2607.07720.
5. FOCAL and shared–private multimodal factorization relatives (shared/private + orthogonality for multimodal time series).
6. OSF / SleepBench-style sleep foundation benchmarking efforts (heterogeneous evaluation; not replaced here).
