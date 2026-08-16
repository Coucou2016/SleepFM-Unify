# ChatGPT / literature consultation notes (2026-08-16)

## ChatGPT status

| Item | Result |
|------|--------|
| Conversation URL | **None** — browser MCP still blocked this turn |
| Attempt (prior) | `browser_tabs` `new` → viewId; `browser_navigate` → `Browser view not found` |
| Attempt (this turn) | `tabs new` → viewId; navigate w/ viewId → not found; navigate w/o viewId → no tab; `newTab:true` → no tab |
| Upload | Not performed (text-only policy; also blocked) |
| Manual path | User/Cursor paste `docs/reports/chatgpt-paste-brief-2026-08-16.md` + GitHub URL |
| Fallback | Local WebSearch + nature-skills (`nature-writing` / `nature-figure`) |

**Unblock needed:** restore Cursor Simple Browser MCP tab↔navigate attachment, then re-run ChatGPT Pro/Plus with web search.

## GitHub for ChatGPT (authorized)

| Item | Value |
|------|-------|
| URL | https://github.com/Coucou2016/SleepFM-Unify |
| Visibility | **public** |
| Branch | `main` |
| Contents | code + docs (excl. large `data/`, `outputs/`, secrets) |

Tell ChatGPT: “Please read the public GitHub repo at https://github.com/Coucou2016/SleepFM-Unify for full code/docs” and still paste the focused TEXT brief (no ZIP).

## Literature survey (WebSearch, independently verified this turn)

### Core baselines to cite

1. **SleepFM (ICML 2024)** — Thapa et al. Multi-modal PSG foundation model; leave-one-out (LOO) contrastive learning across BAS/ECG/respiratory; staging / SDB / retrieval. PMLR 235:48019–48037; arXiv:2405.17766; code https://github.com/rthapa84/sleepfm-codebase.
2. **SleepFM disease prediction (Nature Medicine)** — doi:10.1038/s41591-025-04133-4; medRxiv 10.1101/2025.02.04.25321675. Large-scale LOO-CL; disease risk; SHHS transfer. **Do not conflate** with ICML 2024 staging/retrieval numbers or invent local CinC metrics.
3. **CIMSleepNet (NeurIPS 2024)** — Missing-modality robust staging via MAIM + SMCCL + multi-level temporal attention. https://github.com/SQAIYY/CIMSleepNet.
4. **Omni-Sleep (arXiv:2607.07720)** — CNS/ANS physiological prior; hierarchical contrastive (intra-system + inter-system) + latent temporal modeling; missing-modality robustness.

### Shared–private / factorization relatives (architecture imitation)

- MultiLoReFT-style shared vs modality-specific subspaces (low-rank edits on frozen encoders).
- FedGAMMA / multimodal shared–private streams with contrastive + orthogonality-style disentanglement themes.
- Classic multimodal shared–private factorization (e.g. private not pulled into cross-modal InfoNCE).

### Recommended paper architecture to imitate

**Primary template:** Nature Machine Intelligence / methods-paper hybrid, with ICML SleepFM experimental spine.

| Section | Job |
|---------|-----|
| Abstract (≤150–200w) | Problem → shared–private Unify on SleepFM → what we show on **real data when available** → boundary |
| Introduction | Sleep multimodal FM landscape → LOO aligns shared factors but can erase private cues / struggle with missingness → Unify factorization + mixed loss |
| Related work | SleepFM / Omni-Sleep / CIMSleepNet / shared–private multimodal |
| Method | Factorization heads; LOO+pairwise on shared; orth; modality dropout + L_miss; optional temporal |
| Experiments | CinC / SHHS / MESA (**待补充** until DUA data); LOO vs Unify; ablations; few-shot; retrieval; missing-modality |
| Discussion | What shared vs private buy; failure modes; clinical AHI vs apnea_epoch_rate |
| Methods depth | Configs, channel schema, label gates, compute |

**Do not** claim CinC/SHHS metrics until `run_paper_suite` on real exports.

## Innovation framing (Shared–Private Unify)

One-sentence argument (nature-writing):

> On top of SleepFM’s modality encoders, we show that **explicit shared–private factorization with mixed LOO/pairwise contrastive, orthogonality, and missing-modality objectives** yields alignments that remain compatible with LOO retrieval while preserving modality-specific information and improving robustness under modality dropout — with boundaries restricted to evaluated PSG montages and label coverage.

### Claimed contributions (bounded; evidence-gated)

1. **Shared–private heads** on SleepFM encoders (`z = [z_shared; z_private]`) so contrastive alignment does not force private factors into the LOO space.
2. **Mixed Unify loss** with documented weights: LOO + pairwise + orth + miss (+ optional temporal).
3. **Modality dropout + L_miss** that respects `present_mask` (verified in unit tests).
4. **Engineering gates** for honest evaluation: channel meta fail-fast, CinC label coverage, apnea_epoch_rate vs clinical AHI wording, RNG gallery caps.
5. **Reproducible paper suite** CLI for dual LOO+Unify demos without inventing clinic numbers.

### Non-claims

- Synthetic AUROC≈0.5 is **not** a paper result.
- Night `ahi` / `ahi_bin` = apnea-epoch-rate placeholders.
- Official SleepFM 5/1/3 vs schema 10/2/7 requires explicit mismatch override.
- Nature Medicine disease C-Index values are **not** our local results.

## nature-skills

- Found at `C:\Users\Administrator\.cursor\skills\nature-skills\`.
- Used: `nature-writing` (methods + nat-mach-intell framing), `nature-figure` (Python/SciencePlots backend; user-specified).
- Axes: task=manuscript, paper_type=methods, journal=nat-mach-intell (drafting contract), language=en (paper) + zh teacher tone (report).
