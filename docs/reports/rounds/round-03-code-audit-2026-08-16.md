# Round 3 — Code / architecture audit via GitHub tree (2026-08-16)

**Mode:** MCP-blocked substitute (local audit of public tree ≡ https://github.com/Coucou2016/SleepFM-Unify).  
**ChatGPT URL:** none. Paste Chat 2 brief remains in `chatgpt-paste-brief-2026-08-16.md`.

## Inputs

- `sleepfm/models/sleepfm.py` (factorization, LOO/pairwise, L_miss, orth, dropout).
- `configs/unify.yaml`, `unify_temporal.yaml`, `docs/UNIFY.md`.
- `sleepfm/eval/retrieval.py` gallery RNG; `scripts/check_data_ready.py` stages.
- `scripts/run_paper_suite.py` demo path.

## Judgment — P0 / P1 only

| ID | Severity | Finding | Action |
|----|----------|---------|--------|
| P0-1 | P0 | No real CinC/SHHS on disk; credentials unset | Document user download steps; keep 待补充 |
| P0-2 | P0 | Night `ahi` = apnea_epoch_rate placeholder | Keep wording in paper/report; never call clinical AHI |
| P0-3 | P0 | Synthetic AUROC≈0.5 must not be paper claims | Figures/captions already mark demo; reinforced |
| P1-1 | P1 | Official SleepFM 5/1/3 vs schema 10/2/7 | Channel meta fail-fast already present |
| P1-2 | P1 | CinC arousal-only staging claims | Label coverage gate already present |
| P1-3 | P1 | L_miss / LOO respect `present_mask` | Verified in `pretrain_loss` / `_leave_one_out_loss` |
| P1-4 | P1 | Gallery prefix bias | Default `mode="rng"` seeded subsample |

No new correctness bugs found that require code surgery this round. Protocol completeness improved via `docs/DATA_ACCESS.md` + `scripts/protocol_checklist.py`.

## Local changes

- Added `scripts/protocol_checklist.py` (inventory + credential flags + next commands).
- Documented audit in this round file + section-19.

## Tests

`protocol_checklist.py` smoke + full suite in Round 5.
