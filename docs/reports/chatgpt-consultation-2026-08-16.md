# ChatGPT / literature consultation notes (2026-08-16 · maturity pass)

## ChatGPT status

| Item | Result |
|------|--------|
| Conversation URL | **None** — browser MCP blocked |
| Attempt 1 | `browser_navigate` https://chatgpt.com → `No browser tab available` |
| Attempt 2 | `browser_tabs` `new` (active) → viewId `359ac8`; navigate w/ viewId → `Browser view not found: 359ac8` |
| Attempt 3 | `browser_navigate` without viewId → `No browser tab available` |
| Policy | Do **not** fake ChatGPT URLs; use documented substitute rounds |
| Manual path | `docs/reports/chatgpt-paste-brief-2026-08-16.md` (5 chats) + GitHub URL |
| Fallback rounds | `docs/reports/rounds/round-01` … `round-05` |

## GitHub for ChatGPT (authorized)

| Item | Value |
|------|-------|
| URL | https://github.com/Coucou2016/SleepFM-Unify |
| Visibility | **public** |
| Branch | `main` |
| Contents | code + docs (excl. large `data/`, `outputs/`, secrets) |

## Literature survey (WebSearch, re-verified this turn)

1. **SleepFM (ICML 2024)** — Thapa et al. PMLR 235:48019–48037; arXiv:2405.17766; LOO-CL; staging/SDB/retrieval.
2. **SleepFM disease (Nature Medicine)** — doi:10.1038/s41591-025-04133-4; ~585k hours / ~65k participants; disease C-Index. **Do not conflate** with ICML staging numbers or invent local CinC metrics.
3. **CIMSleepNet (NeurIPS 2024)** — MAIM + SMCCL; https://github.com/SQAIYY/CIMSleepNet.
4. **Omni-Sleep (arXiv:2607.07720)** — CNS/ANS hierarchy; intra-/inter-system contrastive + latent temporal.

## Code audit summary (Round 3)

P0: no real data/creds; night AHI wording; synthetic≠claims.  
P1: channel meta gate; CinC label coverage; L_miss/`present_mask` verified; gallery RNG default.  
No new correctness bugs requiring surgery beyond Windows GBK CLI/test hardening.
