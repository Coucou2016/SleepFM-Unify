# ChatGPT paste briefs (2026-08-16) — TEXT ONLY · 5 chats

Browser MCP could not open ChatGPT this turn (see section-19). Paste each block
into a **separate** ChatGPT Pro/Plus chat with **web search on**. Point ChatGPT at
**https://github.com/Coucou2016/SleepFM-Unify** (`main`). **No ZIP uploads.**

---

## Chat 1 — Literature + architecture

Please enable web search. Read https://github.com/Coucou2016/SleepFM-Unify.

SleepFM-Unify: shared–private factorization on SleepFM 1D EffNet encoders
(`z=[z_shared; z_private]`), mixed LOO+pairwise InfoNCE on shared only,
orthogonality, modality dropout + L_miss (respects `present_mask`), optional
night temporal head. Baseline LOO remains `unify.enabled: false`.

1. Refine related-work vs SleepFM ICML 2024, Nature Medicine SleepFM disease
   (doi:10.1038/s41591-025-04133-4; do not conflate metrics), CIMSleepNet NeurIPS
   2024, Omni-Sleep arXiv:2607.07720.
2. Flag what must stay 待补充 until CinC/SHHS/MESA DUA data exist.
3. Suggest honest experiment matrix (LOO vs Unify, ablations, missingness,
   retrieval, few-shot).

Hard constraint: no fabricated CinC/SHHS numbers; night ahi = apnea_epoch_rate
placeholder; synthetic AUROC≈0.5 is demo only.

---

## Chat 2 — Innovation framing + NMI outline

Same GitHub URL. Advise Nature Machine Intelligence–style methods framing:
≤150-word abstract budget, evidence-gated contribution bullets, non-claims list,
Results vs Methods separation when only synthetic demos exist. Critique our
current `docs/paper/paper.md` outline if you can fetch it from GitHub.

---

## Chat 3 — Code review via GitHub

Review `sleepfm/`, `configs/unify*.yaml`, `docs/UNIFY.md`, retrieval gallery RNG,
`check_data_ready` stages, `run_paper_suite.py`. List **P0/P1 risks only**
(correctness, claim honesty, missing-modality), not style nits.

---

## Chat 4 — Methods / Results chapter review

Paste the Methods + Experiments + Results sections from
`docs/paper/paper.md` (or fetch from GitHub). Ask: Is the demo vs 待补充
boundary clear? Any overclaim risk? What to add for reproducibility without
inventing metrics?

---

## Chat 5 — Risk / honesty + figure captions

Review figure caption policy for fig01–fig06 (SciencePlots; synthetic labeled).
Check night AHI wording, channel meta 5/1/3 vs 10/2/7, CinC label coverage gates.
Propose caption edits only; do not invent clinical numbers.

---

## Local status (Cursor)

- Public repo: https://github.com/Coucou2016/SleepFM-Unify
- ChatGPT MCP: blocked (navigate/tab attachment failure)
- Substitute rounds: `docs/reports/rounds/round-01` … `round-05`
- Data: synthetic + cinc2018_fixture only; PhysioNet/NSRR credentials unset
- Protocol: `docs/DATA_ACCESS.md`, `scripts/protocol_checklist.py`
