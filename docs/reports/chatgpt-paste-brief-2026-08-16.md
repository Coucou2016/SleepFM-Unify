# ChatGPT paste brief (2026-08-16) — TEXT ONLY

Please enable **web search**. Please read the public GitHub repo at
**https://github.com/Coucou2016/SleepFM-Unify** for full code/docs
(branch `main`). Do **not** ask for a ZIP upload.

## Role
You are advisor only. Cursor implements. Focus on (A) literature + Nature-style paper architecture/innovation, and (B) optional code/architecture review from the GitHub tree.

## Project one-liner
**SleepFM-Unify**: shared–private factorization on SleepFM 1D EffNet encoders
(`z=[z_shared; z_private]`), mixed LOO+pairwise InfoNCE on shared only,
orthogonality, modality dropout + L_miss (respects `present_mask`), optional
night temporal head. Baseline LOO remains `unify.enabled: false`.

## Ask (Chat 1 — literature + paper)
1. With web search, refine related-work vs SleepFM ICML 2024, Nature Medicine SleepFM disease paper (do not conflate metrics), CIMSleepNet NeurIPS 2024, Omni-Sleep arXiv:2607.07720.
2. Propose Nature Machine Intelligence–style methods framing + contribution bullets that are **evidence-gated**.
3. Flag what must stay **待补充** until CinC/SHHS/MESA DUA data exist.
4. Suggest honest experiment matrix (LOO vs Unify, ablations, missingness, retrieval, few-shot).

## Ask (Chat 2 — code review via GitHub)
1. Review `sleepfm/`, `configs/unify*.yaml`, `docs/UNIFY.md`, retrieval gallery RNG, `check_data_ready` stages.
2. List P0/P1 risks only (correctness, claim honesty, missing-modality), not style nits.

## Hard constraints
- No fabricated CinC/SHHS/MESA numbers.
- Night `ahi` = apnea_epoch_rate placeholder, not clinical AHI.
- Synthetic AUROC≈0.5 is demo only.

## Local status (Cursor)
- Public repo pushed (code/docs); **not deployed**.
- ~68 unit tests + smoke previously green; re-run this turn.
- Self-contained `docs/reports/report.html` (Base64 figures, inline CSS, no CDN).
