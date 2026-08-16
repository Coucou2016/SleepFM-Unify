# Round 2 — Innovation framing + NMI outline (2026-08-16)

**Mode:** MCP-blocked substitute (nature-writing: methods + nat-mach-intell).  
**ChatGPT URL:** none.

## Inputs

- nature-skills axes: `task=manuscript`, `paper_type=methods`, `journal=nat-mach-intell`, sections abstract/intro/method/experiments/discussion.
- Stance: no invented metrics; evidence-gated contributions.

## Judgment

NMI-style methods article needs: ≤150-word unreferenced abstract; clear mechanism claim; fair LOO baseline; reproducibility; bounded transfer. With only synthetic demos, the paper must read as **engineering-complete methods draft** with **待补充** clinical tables — not a results paper.

One-sentence argument:

> Explicit shared–private factorization on SleepFM encoders, with mixed LOO/pairwise InfoNCE on the shared head plus orthogonality and present_mask-aware missingness, preserves LOO-compatible retrieval while retaining private capacity for concat downstream — pending real CinC/SHHS confirmation.

## Local changes

- Rewrote abstract to NMI length/budget discipline.
- Strengthened contribution bullets with explicit non-claims.
- Added experiment matrix (LOO vs Unify, ablations, missingness, retrieval, few-shot) with 待补充 cells.
- Aligned Methods reproducibility subsection with `docs/UNIFY.md`.

## Tests

Paper suite / unit tests in Round 5.
