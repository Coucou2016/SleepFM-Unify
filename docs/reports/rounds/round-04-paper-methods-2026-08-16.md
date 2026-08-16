# Round 4 — Paper methods / results revision (2026-08-16)

**Mode:** MCP-blocked substitute (local chapter review + revision).  
**ChatGPT URL:** none.

## Inputs

- Draft Methods + Results from `docs/paper/paper.md`.
- `docs/UNIFY.md` loss definitions and CLI.
- Synthetic-only evidence (loss curves, Gram demo).

## Judgment

Methods depth was adequate for engineering, but NMI methods papers need: explicit assumptions, evaluation protocol, hardware/software, failure modes, and a results section that separates **demo smoke** from **planned real-data tables**. Results must not imply superiority from chance AUROC.

## Local changes

- Expanded Methods: factorization assumptions, loss schedule, evaluation protocol, compute note.
- Results: Table 1 clearly DEMO; Table 2 planned real matrix all 待补充; ablation list protocol-complete.
- Discussion: failure modes (montage mismatch, CinC label coverage, AHI wording).
- Regenerated SciencePlots figures with reinforced honesty captions (Round 4→5).

## Tests

Figure regen + docs bundle + pytest in Round 5.
