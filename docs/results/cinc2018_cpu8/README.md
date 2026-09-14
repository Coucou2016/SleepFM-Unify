# CinC 2018 CPU-8 measured results

Measured metrics from PhysioNet Challenge 2018 **open training** subset
(8 subjects used for the reported tables; 24 subjects downloaded locally).

| Artifact | Role |
|----------|------|
| `measured_compact.json` | Paper/report table source |
| `summary.json` | Full paper-suite dump |
| `01_*.json` … `09_*.json` | Per-step suite outputs |
| `supervised_effnet_metrics.json` | EffNet supervised baseline |
| `seq_staging_baseline_metrics.json` | SeqStagingBaseline |
| `fig02_loss_history.json` | Unify pretrain loss (measured) |
| `06_retrieval_loo.txt` | LOO retrieval log (R@10 macro ≈ 0.0209) |

**Not included:** raw `.mat` / exported `.npy` / `.pt` checkpoints (gitignored).
SHHS/MESA: not run (NSRR DUA missing).
