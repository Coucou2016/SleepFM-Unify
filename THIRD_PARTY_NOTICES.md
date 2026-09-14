# Third-party notices

This repository (`sleepfm` / SleepFM-Unify) builds on ideas and, where applicable,
reference implementations from the SleepFM research line.

## SleepFM (upstream code / ICML line)

- Thapa et al., *SleepFM: Multi-modal Representation Learning for Sleep Across
  Brain Activity, ECG and Respiratory Signals*, ICML 2024.
- Upstream reference code: https://github.com/rthapa84/sleepfm-codebase

This **SleepFM-Unify** tree is released under **MIT** (see `LICENSE`). That
covers **our** code and docs in this repository only.

Upstream SleepFM **source** is typically MIT-compatible; always verify the
license file of any upstream revision you copy or vendor. Official or
third-party **checkpoints are not redistributed** here.

## Clinical / Nature Medicine SleepFM weights (not redistributed)

Later SleepFM disease-risk / clinical-scale checkpoints (e.g. Nature Medicine
line) may be released under **non-commercial (NC)** or other restricted terms
by their authors. Those weights are **not** included in this repo. If you
download them yourself, you must obey **their** license and citation
requirements; do not assume MIT from this repository applies to those files.

## Other dependencies

Runtime Python packages (PyTorch, NumPy, scikit-learn, etc.) are covered by
their own licenses; see `pyproject.toml` / `requirements.txt`. Optional extras:

- `pip install -e ".[paper]"` — matplotlib, SciencePlots
- `pip install -e ".[psg]"` — mne, pyedflib, wfdb

## FourDVarNet (archival)

The in-tree `fourdvarnet/` package is an archival research tree and is **not**
part of the default `sleepfm` install (`pyproject.toml` excludes it from package
discovery and entry points). Treat it as unrelated legacy code unless you
explicitly opt in (`pip install -e ".[fourdvarnet]"` is a no-op marker only).
