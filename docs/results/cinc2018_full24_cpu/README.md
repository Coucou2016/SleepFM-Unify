# CinC 2018 full24 CPU measured results

Measured metrics from PhysioNet Challenge 2018 **open training** subset
(**24 subjects** exported; CPU stride index for training/eval).

| Artifact | Role |
|----------|------|
| `measured_compact.json` | Paper/report table source |
| `summary.json` | Full paper-suite dump (lite: fewshot×3) |
| `01_*.json` … `09_*.json` | Per-step suite outputs |
| `05_downstream_loo.json` / `05_downstream_unify.json` | Split LOO vs Unify probes |
| `fig02_loss_history.json` | LOO + Unify pretrain losses |
| `06_retrieval_loo.txt` | LOO retrieval log (R@10 macro ≈ 0.0214) |

**Scale:** device=CPU; 2100 indexed / 21532 full-export epochs; batch 8; 5 pretrain epochs;
paper suite lite (`fewshot_repeats=3`, `max_gallery=500`). Torch CUDA unavailable on this host
(GTX 950M present but CPU-only PyTorch).

**Not included:** raw `.mat` / exported `.npy` / `.pt` checkpoints (gitignored).
SHHS/MESA: not run (`NSRR_TOKEN` unset).
