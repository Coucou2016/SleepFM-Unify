# Round 1 — Literature + architecture (2026-08-16)

**Mode:** MCP-blocked substitute (WebSearch + local paper revision).  
**ChatGPT URL:** none (see MCP blocker evidence in section-19).

## Inputs

- Prior paste brief + consultation notes.
- WebSearch on SleepFM ICML 2024, Nature Medicine SleepFM disease, CIMSleepNet NeurIPS 2024, Omni-Sleep arXiv:2607.07720.
- Existing `docs/paper/paper.md` related-work draft.

## Judgment

| Source | Verdict for SleepFM-Unify |
|--------|---------------------------|
| SleepFM ICML 2024 (PMLR 235:48019–48037; arXiv:2405.17766) | Primary backbone + LOO baseline; cite staging/SDB/retrieval numbers **only as prior art**, never as our local results. |
| Nature Medicine SleepFM disease (doi:10.1038/s41591-025-04133-4) | Different scale/task (disease C-Index); **must not** fill our CinC/SHHS tables. |
| CIMSleepNet NeurIPS 2024 | Closest missing-modality relative (MAIM + SMCCL); Unify uses dropout + L_miss, not imagination. |
| Omni-Sleep arXiv:2607.07720 | CNS/ANS hierarchy; Unify is flat shared–private on SleepFM heads, not a new ontology. |

## Local changes

- Matured related-work + positioning one-liner in `docs/paper/paper.md`.
- Added `docs/DATA_ACCESS.md` documenting absent real downloads.
- Refreshed consultation + paste briefs for manual ChatGPT.

## Tests

Deferred to Round 5 acceptance (`run_tests.py` + `smoke_test.py`).
