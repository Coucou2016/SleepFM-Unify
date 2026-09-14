# Third-party notices

This repository (`sleepfm` / SleepFM-Unify) builds on ideas and, where applicable,
reference implementations from the SleepFM research line.

## SleepFM (upstream)

- Thapa et al., *SleepFM: Multi-modal Representation Learning for Sleep Across
  Brain Activity, ECG and Respiratory Signals*, ICML 2024.
- Related Nature Medicine SleepFM disease-risk work (2026 line).
- Upstream reference code: https://github.com/rthapa84/sleepfm-codebase

Users must respect the license and citation requirements of any upstream
SleepFM code or checkpoints they download. Official checkpoints are **not**
redistributed in this repository.

## Other dependencies

Runtime Python packages (PyTorch, NumPy, scikit-learn, etc.) are covered by
their own licenses; see the environment / `pyproject.toml` dependency list.

## FourDVarNet (archival)

The in-tree `fourdvarnet/` package is an archival research tree and is **not**
part of the default `sleepfm` install (`pyproject.toml` excludes it from package
discovery and entry points). Treat it as unrelated legacy code unless you
explicitly opt in.
