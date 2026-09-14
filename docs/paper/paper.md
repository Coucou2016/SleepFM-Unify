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
| PhysioNet CinC 2018 | Paper pretrain/eval | **Measured** CPU-8 (`docs/results/cinc2018_cpu8/`) and **full24** (`docs/results/cinc2018_full24_cpu/`) |
| NSRR SHHS / MESA | Transfer / night labels | **Blocked** — `NSRR_TOKEN` unset (NSRR DUA required) |

### Protocol (executed)

1. Open-access CinC 2018 training subset via `scripts/download_cinc2018_subset.py` (S3 `physionet-open`, no login; 24 subjects on disk).
2. Export + validate with strict participant isolation (`claim_staging` / `claim_apnea` gates **pass** on WFDB `.arousal` stages + respiratory events).
3. CPU-8 protocol: 8-subject export; stride index 670 / 7304 epochs; LOO+Unify; full paper suite (few-shot×10).
4. Full24 protocol: export all 24 → `data/cinc2018_full24` (21532 epochs); stride index 2100; LOO+Unify (`configs/cinc_full24_cpu.yaml`, `configs/unify_cinc_full24_cpu.yaml`); paper suite **lite** (few-shot×3, gallery 500); device=CPU (CUDA build unavailable).
5. Supervised baselines reported for CPU-8 only (same index as that protocol).

### Experiment matrix (measured CinC CPU-8)

Source: `docs/results/cinc2018_cpu8/measured_compact.json`. Values are **measured**, not invented.

| Experiment | LOO | Unify | Metric | Measured |
|------------|-----|-------|--------|----------|
| Staging (macro AUROC/AUPRC) | 0.454 / 0.289 | 0.495 / 0.235 | Macro AUROC / AUPRC | Yes (n_test=60 epochs, 1 participant) |
| SDB / apnea probe | **0.662** / 0.454 | 0.436 / 0.189 | AUROC / AUPRC | Yes (label gate pass) |
| Retrieval Recall@10 (macro) | — | 0.0198 (rand≈0.020) | Directed co-presence R@10 | Yes (gallery 500) |
| Modality subsets (7) | — | bas 0.514 … full 0.495 | Staging macro AUROC | Yes |
| Space probe (concat / shared / private) | — | 0.495 / 0.510 / 0.499 | Staging macro AUROC | Yes |
| Few-shot (k∈{1,2,4}, 10 repeats) | — | 0.4955±0.0000 | Macro AUROC mean±95% CI | Yes (degenerate: only 1 train participant) |
| Night staging κ | — | κ=−0.087 (acc 0.367) | Cohen κ | Yes (mean-pool; apnea-epoch-rate N/A: &lt;2 nights/split) |
| Supervised EffNet | 0.393 macro AUROC | — | Staging | Yes |
| SeqStagingBaseline (CNN+GRU) | 0.518 macro AUROC | — | Staging | Yes |

### Experiment matrix (measured CinC full24 CPU)

Source: `docs/results/cinc2018_full24_cpu/measured_compact.json`. **Measured** on 24-subject export; CPU stride 2100/21532; suite lite (few-shot×3).

| Experiment | LOO | Unify | Metric | Measured |
|------------|-----|-------|--------|----------|
| Staging (macro AUROC/AUPRC) | 0.496 / 0.224 | **0.557** / 0.231 | Macro AUROC / AUPRC | Yes (n_test=180 epochs, 3 participants) |
| SDB / apnea probe | **0.664** / 0.307 | 0.326 / 0.154 | AUROC / AUPRC | Yes (label gate pass) |
| Retrieval Recall@10 (macro) | 0.0214 | 0.0200 (rand≈0.020) | Directed co-presence R@10 | Yes (gallery 500) |
| Modality subsets (7) | — | bas 0.577 … full 0.557 | Staging macro AUROC | Yes |
| Space probe (concat / shared / private) | — | 0.557 / 0.562 / 0.561 | Staging macro AUROC | Yes |
| Few-shot (k=1, 3 repeats) | — | 0.5371±0.1927 | Macro AUROC mean±95% CI | Yes (lite repeats) |
| Night staging κ | — | κ=−0.009 (acc 0.089) | Cohen κ | Yes (2 train / 3 test nights; mean-pool) |

FOCAL / CIMSleepNet / OSF-SleepBench: **external citation only** (not installed/run here).

### Results

CPU-8 and full24 CinC numbers are real PhysioNet Challenge 2018 exports with AASM-stage and respiratory labels from WFDB `.arousal` files. Staging AUROCs remain near chance under short CPU schedules — reported honestly as scale-limited measured values, not clinic SOTA. On full24, Unify staging macro AUROC (0.557) exceeds LOO (0.496), while LOO apnea AUROC (0.664) still exceeds Unify (0.326) on this split. SHHS/MESA cells remain empty (`NSRR_TOKEN` unset). Synthetic demos remain engineering-only.

#### Ablations (full24 modality subsets + space probes)

Unify modality-ablation staging macro AUROC: bas 0.577; ecg 0.500; respiratory 0.562; bas+ecg 0.577; bas+respiratory 0.557; ecg+respiratory 0.563; all three 0.557. Space probe: concat 0.557, shared 0.562, private 0.561.
---

## Discussion

Unify keeps SleepFM’s LOO retrieval semantics in the shared space while retaining private capacity for downstream concat, and makes missing-modality / heterogeneous-montage training first-class. Claim discipline (label gates, AHI wording, channel meta, strict isolation) is part of the scientific interface.

**Failure modes.** (i) Montage mismatch without override → load fails by design. (ii) CinC staging claims without coverage → gate blocks. (iii) Interpreting synthetic near-chance AUROC as a positive result → rejected by caption policy. (iv) Calling night apnea-epoch-rate “AHI” → wording violation. (v) Claiming novel shared–private factorization → rejected by Related work positioning.

**Limitations.** Measured CinC CPU-8 and full24 results are scale-limited (stride-subsampled epochs, 5 CPU epochs; no CUDA training build here); staging remains near chance. SHHS/MESA are still blocked without `NSRR_TOKEN`. Channel-aware soft pooling is implemented and enabled for Unify; encoders that *resize* the channel axis remain future work. FOCAL/CIMSleepNet/OSF are external citations only. Transfer claims are bounded to evaluated montages.

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
- **Fig. 2** Unify pretrain loss on real CinC CPU-8 protocol (measured; not synthetic demo).
- **Fig. 3** Measured modality-ablation / baseline bars from `docs/results/cinc2018_cpu8/` (CPU-8 scale).
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
