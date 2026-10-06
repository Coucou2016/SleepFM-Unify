# START HERE - this repository is INTENTIONALLY flat (no folders)

**This layout is deliberate, not an accident.** Every source file, test, config,
document, figure, the paper, the research report, the measured result files and
the audit trail are placed **directly in the repository root**, with **no
sub-directories at all**.

Generated: 2026-10-06 15:20 UTC
Source commit (of the original nested tree): `caaaa4906cefbb8855945e0c8764f60ac27db34b`
Mirror branch: `flat`
Files at the root of this branch: **226** = 224
mirrored files + this guide + `_flat_manifest.json` (~8.2 MB of
text/JSON/PDF)

> Reading tip: if you are an automated reviewer, fetch the branch listing once,
> then fetch each file by its flat name. Nothing requires recursive descent.

---

## 1. Why the flat layout

1. **Full-content reading by external reviewers.** Web-based reviewers and
   model-based reviewers (chatgpt.com, another coding agent, a browser) can
   enumerate a flat root with a single listing call instead of recursively
   guessing folder names. Nothing hides behind a directory they did not think
   to open.
2. **Reproducible cross-review.** Two independent agents reading the *same*
   flat listing will see the *same* file set, which makes disagreement
   debuggable ("you read `docs__paper__paper.md`, I read `paper.md`" cannot
   happen).
3. **Zero path ambiguity.** Deeply nested trees cause reviewers to cite paths
   that do not exist in their view. A one-level namespace removes that class of
   error.
4. **Easy archival.** A flat set of files can be zipped, copied into any
   chat/issue, or diffed against another repository without restructuring.

Trade-off we accept: the flat branch is **not** the canonical Python package
layout, so `pip install -e .` / `pytest` are **not** intended to run from this
branch. Use `main` for running code; use this branch for **reading and
reviewing**. Both branches hold the same file contents.

## 2. Naming convention (how to map a flat name back to a real path)

| Flat name form | Original path |
|---|---|
| `README.md`, `LICENSE`, `pyproject.toml` | root-level files keep their name |
| `sleepfm__models__sleepfm.py` | `sleepfm/models/sleepfm.py` |
| `docs__paper__paper.md` | `docs/paper/paper.md` |
| `docs__results__cinc2018_full24_cpu__measured_compact.json` | `docs/results/cinc2018_full24_cpu/measured_compact.json` |

Rule: replace every `/` with `__` (double underscore). Root files are unchanged.

## 3. Suggested reading order for a reviewer

1. `README.md` - project overview, scope, honest status.
2. `START_HERE_FLAT_LAYOUT.md` - this file (layout + inventory).
3. `docs__paper__paper.md` - the manuscript draft (methods + framing).
4. `docs__reports__report.md` - the parallel research report (long-form
   explanations of every figure and table).
5. `docs__UNIFY.md` - the exact method definition, losses and evaluation
   protocol.
6. `docs__DEVELOPMENT_STATUS.md` / `docs__EXPERIMENT_CHECKLIST.md` - what is
   verified, what is still blank, and why.
7. `docs__results__cinc2018_cpu8__measured_compact.json` and
   `docs__results__cinc2018_full24_cpu__measured_compact.json` - the **measured**
   numbers that back the tables (PhysioNet/CinC 2018, CPU protocol, honest
   scale).
8. `sleepfm__models__sleepfm.py` - the model: shared/private heads, mixed
   contrastive loss, missing-modality objective, channel-aware pooling.
9. `sleepfm__eval__downstream.py`, `sleepfm__eval__retrieval.py`,
   `sleepfm__eval__night.py` - evaluation paths (train/valid/test separation,
   co-presence-filtered retrieval, masked aggregation).
10. `tests__*.py` - the regression tests that pin the behaviour above.
11. `docs__reports__round-0*.md` and `docs__reports__section-19-final*.md` -
    the audit/iteration trail.

## 4. What is deliberately NOT here (and where it lives)

| Excluded | Reason | Where / how to get it |
|---|---|---|
| `*.pt` / `*.pth` checkpoints (~80-170 MB each) | far beyond what a text reviewer can read; Git-hostile | regenerate with `scripts__pretrain.py`; download official with `scripts__download_checkpoint.py` |
| Processed `*.npy` epochs (`data/cinc2018*`, ~16 GB) | size only; content is derived | re-export from raw with `scripts__export_edf.py` |
| Raw PSG / EDF (`data/raw/`, ~3.2 GB) | licensed third-party data, not redistributable | PhysioNet (CinC 2018) / NSRR (SHHS, MESA) under their terms - see `docs__DATA_ACCESS.md` |
| `.zip` source snapshots | redundant with the flat mirror | - |
| `.env`, keys, cookies, tokens | secrets policy | never committed |

Everything excluded is **derived or licensed**, so the mirror still contains
100% of what is needed to audit the science.

## 5. Honesty statement about the numbers

* The only **measured real-data** results in this mirror are the PhysioNet/CinC
  2018 CPU-protocol runs under `docs__results__cinc2018_cpu8__*` and
  `docs__results__cinc2018_full24_cpu__*`.
* SHHS / MESA results are **not present** because access requires an NSRR data
  use agreement. They are marked as pending, never invented.
* Synthetic-data numbers are demo/CI artefacts only and are labelled as such.

## 6. How to cross-review this mirror

1. Read `START_HERE_FLAT_LAYOUT.md` (this file) first - it is the map.
2. Cross-check each claim in `docs__paper__paper.md` against the measured JSON
   in `docs__results__*`.
3. Cross-check the code paths cited in the paper against the corresponding flat
   `sleepfm__*.py` files.
4. Cross-check the stated method against `docs__UNIFY.md` **and** the actual
   implementation; discrepancies are what this review is for.
5. Report findings by flat filename (e.g. `sleepfm__eval__retrieval.py`) so the
   reference is unambiguous.

## 7. Complete inventory (224 files)

| Flat file (this branch) | Original path (main branch) | Size |
|---|---|---|
| `.gitignore` | `.gitignore` | 0.8 KB |
| `CITATION.cff` | `CITATION.cff` | 1.5 KB |
| `LICENSE` | `LICENSE` | 1.5 KB |
| `README.md` | `README.md` | 7.6 KB |
| `THIRD_PARTY_NOTICES.md` | `THIRD_PARTY_NOTICES.md` | 1.8 KB |
| `configs__channels__cinc2018.yaml` | `configs/channels/cinc2018.yaml` | 2.3 KB |
| `configs__channels__mesa.yaml` | `configs/channels/mesa.yaml` | 1.7 KB |
| `configs__channels__shhs.yaml` | `configs/channels/shhs.yaml` | 1.8 KB |
| `configs__cinc_cpu.yaml` | `configs/cinc_cpu.yaml` | 0.3 KB |
| `configs__cinc_full24_cpu.yaml` | `configs/cinc_full24_cpu.yaml` | 0.4 KB |
| `configs__default.yaml` | `configs/default.yaml` | 2.1 KB |
| `configs__fourdvarnet.yaml` | `configs/fourdvarnet.yaml` | 1.1 KB |
| `configs__unify.yaml` | `configs/unify.yaml` | 0.6 KB |
| `configs__unify_cinc_cpu.yaml` | `configs/unify_cinc_cpu.yaml` | 0.2 KB |
| `configs__unify_cinc_full24_cpu.yaml` | `configs/unify_cinc_full24_cpu.yaml` | 0.3 KB |
| `configs__unify_temporal.yaml` | `configs/unify_temporal.yaml` | 0.3 KB |
| `data__cinc2018__cpu_protocol.json` | `data/cinc2018/cpu_protocol.json` | 2.3 KB |
| `data__cinc2018__index.json` | `data/cinc2018/index.json` | 179.1 KB |
| `data__cinc2018__index_full.json` | `data/cinc2018/index_full.json` | 1299.1 KB |
| `data__cinc2018_fixture__index.json` | `data/cinc2018_fixture/index.json` | 4.0 KB |
| `data__cinc2018_full24__cpu_protocol.json` | `data/cinc2018_full24/cpu_protocol.json` | 2.3 KB |
| `data__cinc2018_full24__index.json` | `data/cinc2018_full24/index.json` | 556.7 KB |
| `data__synthetic__index.json` | `data/synthetic/index.json` | 16.0 KB |
| `docs__DATA_ACCESS.md` | `docs/DATA_ACCESS.md` | 4.0 KB |
| `docs__DATA_SCHEMA.md` | `docs/DATA_SCHEMA.md` | 2.1 KB |
| `docs__DEVELOPMENT_STATUS.md` | `docs/DEVELOPMENT_STATUS.md` | 2.7 KB |
| `docs__EXPERIMENT_CHECKLIST.md` | `docs/EXPERIMENT_CHECKLIST.md` | 3.2 KB |
| `docs__MATERIALS.md` | `docs/MATERIALS.md` | 3.9 KB |
| `docs__UNIFY.md` | `docs/UNIFY.md` | 15.2 KB |
| `docs__figures__fig01_architecture.png` | `docs/figures/fig01_architecture.png` | 175.8 KB |
| `docs__figures__fig01_architecture.svg` | `docs/figures/fig01_architecture.svg` | 116.7 KB |
| `docs__figures__fig02_cinc_cpu8_loss_bars.png` | `docs/figures/fig02_cinc_cpu8_loss_bars.png` | 106.0 KB |
| `docs__figures__fig02_cinc_cpu8_loss_bars.svg` | `docs/figures/fig02_cinc_cpu8_loss_bars.svg` | 89.1 KB |
| `docs__figures__fig02_loss_curves.png` | `docs/figures/fig02_loss_curves.png` | 118.6 KB |
| `docs__figures__fig02_loss_curves.svg` | `docs/figures/fig02_loss_curves.svg` | 100.4 KB |
| `docs__figures__fig02_loss_history.json` | `docs/figures/fig02_loss_history.json` | 0.3 KB |
| `docs__figures__fig03_ablation_schematic.png` | `docs/figures/fig03_ablation_schematic.png` | 118.8 KB |
| `docs__figures__fig03_ablation_schematic.svg` | `docs/figures/fig03_ablation_schematic.svg` | 91.8 KB |
| `docs__figures__fig03_cinc_modality_ablation.png` | `docs/figures/fig03_cinc_modality_ablation.png` | 77.8 KB |
| `docs__figures__fig03_cinc_modality_ablation.svg` | `docs/figures/fig03_cinc_modality_ablation.svg` | 65.8 KB |
| `docs__figures__fig04_orthogonality.png` | `docs/figures/fig04_orthogonality.png` | 120.5 KB |
| `docs__figures__fig04_orthogonality.svg` | `docs/figures/fig04_orthogonality.svg` | 116.0 KB |
| `docs__figures__fig05_modality_dropout.png` | `docs/figures/fig05_modality_dropout.png` | 101.0 KB |
| `docs__figures__fig05_modality_dropout.svg` | `docs/figures/fig05_modality_dropout.svg` | 81.8 KB |
| `docs__figures__fig06_pipeline.png` | `docs/figures/fig06_pipeline.png` | 63.9 KB |
| `docs__figures__fig06_pipeline.svg` | `docs/figures/fig06_pipeline.svg` | 53.8 KB |
| `docs__paper__paper.html` | `docs/paper/paper.html` | 960.3 KB |
| `docs__paper__paper.md` | `docs/paper/paper.md` | 17.2 KB |
| `docs__paper__paper.pdf` | `docs/paper/paper.pdf` | 848.3 KB |
| `docs__reports__2026-08-15-unify-continue.md` | `docs/reports/2026-08-15-unify-continue.md` | 6.2 KB |
| `docs__reports__2026-08-15-unify-p1-close.md` | `docs/reports/2026-08-15-unify-p1-close.md` | 2.3 KB |
| `docs__reports__chatgpt-consultation-2026-08-16.md` | `docs/reports/chatgpt-consultation-2026-08-16.md` | 1.8 KB |
| `docs__reports__chatgpt-paste-brief-2026-08-16.md` | `docs/reports/chatgpt-paste-brief-2026-08-16.md` | 2.8 KB |
| `docs__reports__report.html` | `docs/reports/report.html` | 955.3 KB |
| `docs__reports__report.md` | `docs/reports/report.md` | 11.4 KB |
| `docs__reports__report.pdf` | `docs/reports/report.pdf` | 927.4 KB |
| `docs__reports__rounds__round-01-literature-2026-08-16.md` | `docs/reports/rounds/round-01-literature-2026-08-16.md` | 1.4 KB |
| `docs__reports__rounds__round-02-nmi-outline-2026-08-16.md` | `docs/reports/rounds/round-02-nmi-outline-2026-08-16.md` | 1.4 KB |
| `docs__reports__rounds__round-03-code-audit-2026-08-16.md` | `docs/reports/rounds/round-03-code-audit-2026-08-16.md` | 1.7 KB |
| `docs__reports__rounds__round-04-paper-methods-2026-08-16.md` | `docs/reports/rounds/round-04-paper-methods-2026-08-16.md` | 1.1 KB |
| `docs__reports__rounds__round-05-honesty-acceptance-2026-08-16.md` | `docs/reports/rounds/round-05-honesty-acceptance-2026-08-16.md` | 0.9 KB |
| `docs__reports__section-19-final-20260816.md` | `docs/reports/section-19-final-20260816.md` | 5.6 KB |
| `docs__reports__section-19-final.md` | `docs/reports/section-19-final.md` | 5.5 KB |
| `docs__results__cinc2018_cpu8__01_validate.json` | `docs/results/cinc2018_cpu8/01_validate.json` | 0.1 KB |
| `docs__results__cinc2018_cpu8__01b_label_coverage.json` | `docs/results/cinc2018_cpu8/01b_label_coverage.json` | 1.8 KB |
| `docs__results__cinc2018_cpu8__04_channel_check.json` | `docs/results/cinc2018_cpu8/04_channel_check.json` | 0.4 KB |
| `docs__results__cinc2018_cpu8__05_downstream.json` | `docs/results/cinc2018_cpu8/05_downstream.json` | 0.5 KB |
| `docs__results__cinc2018_cpu8__06_retrieval.json` | `docs/results/cinc2018_cpu8/06_retrieval.json` | 1.2 KB |
| `docs__results__cinc2018_cpu8__06_retrieval_loo.txt` | `docs/results/cinc2018_cpu8/06_retrieval_loo.txt` | 0.8 KB |
| `docs__results__cinc2018_cpu8__07_modality_ablation.json` | `docs/results/cinc2018_cpu8/07_modality_ablation.json` | 2.7 KB |
| `docs__results__cinc2018_cpu8__08_fewshot.json` | `docs/results/cinc2018_cpu8/08_fewshot.json` | 19.3 KB |
| `docs__results__cinc2018_cpu8__08b_space_probe.json` | `docs/results/cinc2018_cpu8/08b_space_probe.json` | 1.3 KB |
| `docs__results__cinc2018_cpu8__09_night.json` | `docs/results/cinc2018_cpu8/09_night.json` | 0.9 KB |
| `docs__results__cinc2018_cpu8__README.md` | `docs/results/cinc2018_cpu8/README.md` | 0.8 KB |
| `docs__results__cinc2018_cpu8__fig02_loss_history.json` | `docs/results/cinc2018_cpu8/fig02_loss_history.json` | 0.3 KB |
| `docs__results__cinc2018_cpu8__measured_compact.json` | `docs/results/cinc2018_cpu8/measured_compact.json` | 4.6 KB |
| `docs__results__cinc2018_cpu8__seq_staging_baseline_metrics.json` | `docs/results/cinc2018_cpu8/seq_staging_baseline_metrics.json` | 0.1 KB |
| `docs__results__cinc2018_cpu8__summary.json` | `docs/results/cinc2018_cpu8/summary.json` | 33.8 KB |
| `docs__results__cinc2018_cpu8__supervised_effnet_metrics.json` | `docs/results/cinc2018_cpu8/supervised_effnet_metrics.json` | 0.1 KB |
| `docs__results__cinc2018_full24_cpu__01_validate.json` | `docs/results/cinc2018_full24_cpu/01_validate.json` | 0.1 KB |
| `docs__results__cinc2018_full24_cpu__01b_label_coverage.json` | `docs/results/cinc2018_full24_cpu/01b_label_coverage.json` | 1.8 KB |
| `docs__results__cinc2018_full24_cpu__04_channel_check.json` | `docs/results/cinc2018_full24_cpu/04_channel_check.json` | 0.4 KB |
| `docs__results__cinc2018_full24_cpu__05_downstream.json` | `docs/results/cinc2018_full24_cpu/05_downstream.json` | 0.5 KB |
| `docs__results__cinc2018_full24_cpu__05_downstream_loo.json` | `docs/results/cinc2018_full24_cpu/05_downstream_loo.json` | 0.5 KB |
| `docs__results__cinc2018_full24_cpu__05_downstream_unify.json` | `docs/results/cinc2018_full24_cpu/05_downstream_unify.json` | 0.5 KB |
| `docs__results__cinc2018_full24_cpu__06_retrieval.json` | `docs/results/cinc2018_full24_cpu/06_retrieval.json` | 1.2 KB |
| `docs__results__cinc2018_full24_cpu__06_retrieval_loo.txt` | `docs/results/cinc2018_full24_cpu/06_retrieval_loo.txt` | 0.8 KB |
| `docs__results__cinc2018_full24_cpu__07_modality_ablation.json` | `docs/results/cinc2018_full24_cpu/07_modality_ablation.json` | 2.8 KB |
| `docs__results__cinc2018_full24_cpu__08_fewshot.json` | `docs/results/cinc2018_full24_cpu/08_fewshot.json` | 8.4 KB |
| `docs__results__cinc2018_full24_cpu__08b_space_probe.json` | `docs/results/cinc2018_full24_cpu/08b_space_probe.json` | 1.3 KB |
| `docs__results__cinc2018_full24_cpu__09_night.json` | `docs/results/cinc2018_full24_cpu/09_night.json` | 1.0 KB |
| `docs__results__cinc2018_full24_cpu__README.md` | `docs/results/cinc2018_full24_cpu/README.md` | 1.0 KB |
| `docs__results__cinc2018_full24_cpu__fig02_loss_history.json` | `docs/results/cinc2018_full24_cpu/fig02_loss_history.json` | 0.6 KB |
| `docs__results__cinc2018_full24_cpu__measured_compact.json` | `docs/results/cinc2018_full24_cpu/measured_compact.json` | 17.5 KB |
| `docs__results__cinc2018_full24_cpu__summary.json` | `docs/results/cinc2018_full24_cpu/summary.json` | 21.6 KB |
| `fourdvarnet____init__.py` | `fourdvarnet/__init__.py` | 0.2 KB |
| `fourdvarnet__data____init__.py` | `fourdvarnet/data/__init__.py` | 0.2 KB |
| `fourdvarnet__data__dataset.py` | `fourdvarnet/data/dataset.py` | 1.4 KB |
| `fourdvarnet__data__natl60.py` | `fourdvarnet/data/natl60.py` | 1.3 KB |
| `fourdvarnet__data__normalize.py` | `fourdvarnet/data/normalize.py` | 2.8 KB |
| `fourdvarnet__data__synthetic_osse.py` | `fourdvarnet/data/synthetic_osse.py` | 8.8 KB |
| `fourdvarnet__eval____init__.py` | `fourdvarnet/eval/__init__.py` | 0.1 KB |
| `fourdvarnet__eval__metrics.py` | `fourdvarnet/eval/metrics.py` | 3.1 KB |
| `fourdvarnet__models____init__.py` | `fourdvarnet/models/__init__.py` | 0.2 KB |
| `fourdvarnet__models__blocks.py` | `fourdvarnet/models/blocks.py` | 2.3 KB |
| `fourdvarnet__models__conv_lstm.py` | `fourdvarnet/models/conv_lstm.py` | 1.2 KB |
| `fourdvarnet__models__fourdvarnet.py` | `fourdvarnet/models/fourdvarnet.py` | 7.7 KB |
| `fourdvarnet__ops____init__.py` | `fourdvarnet/ops/__init__.py` | 0.2 KB |
| `fourdvarnet__ops__physics.py` | `fourdvarnet/ops/physics.py` | 1.6 KB |
| `fourdvarnet__training____init__.py` | `fourdvarnet/training/__init__.py` | 0.1 KB |
| `fourdvarnet__training__trainer.py` | `fourdvarnet/training/trainer.py` | 5.5 KB |
| `outputs__baseline_effnet.log` | `outputs/baseline_effnet.log` | 0.4 KB |
| `outputs__baseline_effnet_cinc__supervised_effnet_metrics.json` | `outputs/baseline_effnet_cinc/supervised_effnet_metrics.json` | 0.1 KB |
| `outputs__baseline_seq.log` | `outputs/baseline_seq.log` | 0.4 KB |
| `outputs__baseline_seq_cinc__seq_staging_baseline_metrics.json` | `outputs/baseline_seq_cinc/seq_staging_baseline_metrics.json` | 0.1 KB |
| `outputs__download_cinc_more.log` | `outputs/download_cinc_more.log` | 1.0 KB |
| `outputs__download_cinc_stderr.log` | `outputs/download_cinc_stderr.log` | 0.0 KB |
| `outputs__download_cinc_stdout.log` | `outputs/download_cinc_stdout.log` | 0.5 KB |
| `outputs__export_cinc.log` | `outputs/export_cinc.log` | 2.3 KB |
| `outputs__export_cinc24.log` | `outputs/export_cinc24.log` | 2.4 KB |
| `outputs__paper_suite_cinc.log` | `outputs/paper_suite_cinc.log` | 1.0 KB |
| `outputs__paper_suite_cinc__20260914_184730__01_validate.json` | `outputs/paper_suite_cinc/20260914_184730/01_validate.json` | 0.1 KB |
| `outputs__paper_suite_cinc__20260914_184730__01b_label_coverage.json` | `outputs/paper_suite_cinc/20260914_184730/01b_label_coverage.json` | 1.8 KB |
| `outputs__paper_suite_cinc__20260914_184730__04_channel_check.json` | `outputs/paper_suite_cinc/20260914_184730/04_channel_check.json` | 0.4 KB |
| `outputs__paper_suite_cinc__20260914_184730__05_downstream.json` | `outputs/paper_suite_cinc/20260914_184730/05_downstream.json` | 0.5 KB |
| `outputs__paper_suite_cinc__20260914_184730__06_retrieval.json` | `outputs/paper_suite_cinc/20260914_184730/06_retrieval.json` | 1.2 KB |
| `outputs__paper_suite_cinc__20260914_184730__07_modality_ablation.json` | `outputs/paper_suite_cinc/20260914_184730/07_modality_ablation.json` | 2.7 KB |
| `outputs__paper_suite_cinc__20260914_184730__08_fewshot.json` | `outputs/paper_suite_cinc/20260914_184730/08_fewshot.json` | 19.3 KB |
| `outputs__paper_suite_cinc__20260914_184730__08b_space_probe.json` | `outputs/paper_suite_cinc/20260914_184730/08b_space_probe.json` | 1.3 KB |
| `outputs__paper_suite_cinc__20260914_184730__09_night.json` | `outputs/paper_suite_cinc/20260914_184730/09_night.json` | 0.9 KB |
| `outputs__paper_suite_cinc__20260914_184730__summary.json` | `outputs/paper_suite_cinc/20260914_184730/summary.json` | 33.8 KB |
| `outputs__paper_suite_cinc_full24.log` | `outputs/paper_suite_cinc_full24.log` | 1.1 KB |
| `outputs__paper_suite_cinc_full24__20260914_213259__01_validate.json` | `outputs/paper_suite_cinc_full24/20260914_213259/01_validate.json` | 0.1 KB |
| `outputs__paper_suite_cinc_full24__20260914_213259__01b_label_coverage.json` | `outputs/paper_suite_cinc_full24/20260914_213259/01b_label_coverage.json` | 1.8 KB |
| `outputs__paper_suite_cinc_full24__20260914_213259__04_channel_check.json` | `outputs/paper_suite_cinc_full24/20260914_213259/04_channel_check.json` | 0.4 KB |
| `outputs__paper_suite_cinc_full24__20260914_213259__05_downstream.json` | `outputs/paper_suite_cinc_full24/20260914_213259/05_downstream.json` | 0.5 KB |
| `outputs__paper_suite_cinc_full24__20260914_213259__06_retrieval.json` | `outputs/paper_suite_cinc_full24/20260914_213259/06_retrieval.json` | 1.2 KB |
| `outputs__paper_suite_cinc_full24__20260914_213259__07_modality_ablation.json` | `outputs/paper_suite_cinc_full24/20260914_213259/07_modality_ablation.json` | 2.8 KB |
| `outputs__paper_suite_cinc_full24__20260914_213259__08_fewshot.json` | `outputs/paper_suite_cinc_full24/20260914_213259/08_fewshot.json` | 8.4 KB |
| `outputs__paper_suite_cinc_full24__20260914_213259__08b_space_probe.json` | `outputs/paper_suite_cinc_full24/20260914_213259/08b_space_probe.json` | 1.3 KB |
| `outputs__paper_suite_cinc_full24__20260914_213259__09_night.json` | `outputs/paper_suite_cinc_full24/20260914_213259/09_night.json` | 1.0 KB |
| `outputs__paper_suite_cinc_full24__20260914_213259__summary.json` | `outputs/paper_suite_cinc_full24/20260914_213259/summary.json` | 21.6 KB |
| `outputs__paper_suite_demo__20260816_060859_demo__01_validate.json` | `outputs/paper_suite_demo/20260816_060859_demo/01_validate.json` | 0.1 KB |
| `outputs__paper_suite_demo__20260816_060859_demo__01b_label_coverage.json` | `outputs/paper_suite_demo/20260816_060859_demo/01b_label_coverage.json` | 1.7 KB |
| `outputs__pretrain_loo_cinc.log` | `outputs/pretrain_loo_cinc.log` | 23.7 KB |
| `outputs__pretrain_loo_cinc_full24.log` | `outputs/pretrain_loo_cinc_full24.log` | 76.2 KB |
| `outputs__pretrain_unify_cinc.log` | `outputs/pretrain_unify_cinc.log` | 23.7 KB |
| `outputs__pretrain_unify_cinc_full24.log` | `outputs/pretrain_unify_cinc_full24.log` | 76.2 KB |
| `pyproject.toml` | `pyproject.toml` | 1.5 KB |
| `requirements.txt` | `requirements.txt` | 0.2 KB |
| `scripts____init__.py` | `scripts/__init__.py` | 0.0 KB |
| `scripts__build_docs_bundle.py` | `scripts/build_docs_bundle.py` | 18.5 KB |
| `scripts__build_flat_upload.py` | `scripts/build_flat_upload.py` | 14.3 KB |
| `scripts__check_data_ready.py` | `scripts/check_data_ready.py` | 8.6 KB |
| `scripts__download_checkpoint.py` | `scripts/download_checkpoint.py` | 3.3 KB |
| `scripts__download_cinc2018_subset.py` | `scripts/download_cinc2018_subset.py` | 6.2 KB |
| `scripts__eval_fewshot.py` | `scripts/eval_fewshot.py` | 2.5 KB |
| `scripts__eval_modality_ablation.py` | `scripts/eval_modality_ablation.py` | 1.2 KB |
| `scripts__eval_retrieval.py` | `scripts/eval_retrieval.py` | 3.4 KB |
| `scripts__eval_space_probe.py` | `scripts/eval_space_probe.py` | 3.0 KB |
| `scripts__eval_transfer.py` | `scripts/eval_transfer.py` | 1.4 KB |
| `scripts__evaluate_downstream.py` | `scripts/evaluate_downstream.py` | 6.2 KB |
| `scripts__evaluate_night.py` | `scripts/evaluate_night.py` | 4.2 KB |
| `scripts__export_edf.py` | `scripts/export_edf.py` | 3.4 KB |
| `scripts__export_nsrr.py` | `scripts/export_nsrr.py` | 1.7 KB |
| `scripts__fourdvarnet_eval.py` | `scripts/fourdvarnet_eval.py` | 2.9 KB |
| `scripts__fourdvarnet_generate_data.py` | `scripts/fourdvarnet_generate_data.py` | 1.5 KB |
| `scripts__fourdvarnet_infer.py` | `scripts/fourdvarnet_infer.py` | 2.3 KB |
| `scripts__fourdvarnet_smoke_test.py` | `scripts/fourdvarnet_smoke_test.py` | 1.2 KB |
| `scripts__fourdvarnet_train.py` | `scripts/fourdvarnet_train.py` | 1.2 KB |
| `scripts__generate_synthetic_data.py` | `scripts/generate_synthetic_data.py` | 1.7 KB |
| `scripts__inference.py` | `scripts/inference.py` | 1.8 KB |
| `scripts__load_official_checkpoint.py` | `scripts/load_official_checkpoint.py` | 4.5 KB |
| `scripts__plot_unify_figures.py` | `scripts/plot_unify_figures.py` | 17.4 KB |
| `scripts__pretrain.py` | `scripts/pretrain.py` | 5.5 KB |
| `scripts__protocol_checklist.py` | `scripts/protocol_checklist.py` | 3.3 KB |
| `scripts__push_flat_branch.py` | `scripts/push_flat_branch.py` | 7.7 KB |
| `scripts__push_git_tree_via_api.py` | `scripts/push_git_tree_via_api.py` | 4.9 KB |
| `scripts__push_via_github_api.py` | `scripts/push_via_github_api.py` | 7.0 KB |
| `scripts__run_paper_suite.py` | `scripts/run_paper_suite.py` | 27.8 KB |
| `scripts__run_tests.py` | `scripts/run_tests.py` | 0.4 KB |
| `scripts__smoke_test.py` | `scripts/smoke_test.py` | 5.5 KB |
| `scripts__train_supervised.py` | `scripts/train_supervised.py` | 9.7 KB |
| `scripts__validate_data.py` | `scripts/validate_data.py` | 0.7 KB |
| `sleepfm____init__.py` | `sleepfm/__init__.py` | 0.1 KB |
| `sleepfm__data____init__.py` | `sleepfm/data/__init__.py` | 0.4 KB |
| `sleepfm__data__channel_map.py` | `sleepfm/data/channel_map.py` | 5.7 KB |
| `sleepfm__data__dataset.py` | `sleepfm/data/dataset.py` | 4.6 KB |
| `sleepfm__data__edf_export.py` | `sleepfm/data/edf_export.py` | 32.4 KB |
| `sleepfm__data__label_coverage.py` | `sleepfm/data/label_coverage.py` | 8.6 KB |
| `sleepfm__data__night_dataset.py` | `sleepfm/data/night_dataset.py` | 7.8 KB |
| `sleepfm__data__splits.py` | `sleepfm/data/splits.py` | 10.6 KB |
| `sleepfm__data__synthetic.py` | `sleepfm/data/synthetic.py` | 11.0 KB |
| `sleepfm__data__validate.py` | `sleepfm/data/validate.py` | 3.8 KB |
| `sleepfm__eval____init__.py` | `sleepfm/eval/__init__.py` | 0.4 KB |
| `sleepfm__eval__downstream.py` | `sleepfm/eval/downstream.py` | 8.6 KB |
| `sleepfm__eval__experiments.py` | `sleepfm/eval/experiments.py` | 8.5 KB |
| `sleepfm__eval__night.py` | `sleepfm/eval/night.py` | 11.9 KB |
| `sleepfm__eval__retrieval.py` | `sleepfm/eval/retrieval.py` | 8.7 KB |
| `sleepfm__models____init__.py` | `sleepfm/models/__init__.py` | 0.4 KB |
| `sleepfm__models__channel_meta.py` | `sleepfm/models/channel_meta.py` | 7.5 KB |
| `sleepfm__models__channel_pool.py` | `sleepfm/models/channel_pool.py` | 4.2 KB |
| `sleepfm__models__encoders.py` | `sleepfm/models/encoders.py` | 10.9 KB |
| `sleepfm__models__official_adapter.py` | `sleepfm/models/official_adapter.py` | 15.1 KB |
| `sleepfm__models__sleepfm.py` | `sleepfm/models/sleepfm.py` | 39.9 KB |
| `sleepfm__models__temporal.py` | `sleepfm/models/temporal.py` | 8.8 KB |
| `sleepfm__training____init__.py` | `sleepfm/training/__init__.py` | 0.1 KB |
| `sleepfm__training__trainer.py` | `sleepfm/training/trainer.py` | 8.0 KB |
| `sleepfm__utils____init__.py` | `sleepfm/utils/__init__.py` | 0.1 KB |
| `sleepfm__utils__config.py` | `sleepfm/utils/config.py` | 1.5 KB |
| `sleepfm__utils__seed.py` | `sleepfm/utils/seed.py` | 0.7 KB |
| `tests__conftest.py` | `tests/conftest.py` | 0.7 KB |
| `tests__test_channel_label_gates.py` | `tests/test_channel_label_gates.py` | 6.6 KB |
| `tests__test_check_data_ready.py` | `tests/test_check_data_ready.py` | 1.0 KB |
| `tests__test_downstream.py` | `tests/test_downstream.py` | 4.7 KB |
| `tests__test_edf_export.py` | `tests/test_edf_export.py` | 4.9 KB |
| `tests__test_experiments.py` | `tests/test_experiments.py` | 1.9 KB |
| `tests__test_model.py` | `tests/test_model.py` | 2.4 KB |
| `tests__test_night.py` | `tests/test_night.py` | 5.7 KB |
| `tests__test_official_adapter.py` | `tests/test_official_adapter.py` | 4.8 KB |
| `tests__test_paper_suite.py` | `tests/test_paper_suite.py` | 4.6 KB |
| `tests__test_retrieval.py` | `tests/test_retrieval.py` | 4.7 KB |
| `tests__test_splits.py` | `tests/test_splits.py` | 5.0 KB |
| `tests__test_unify.py` | `tests/test_unify.py` | 15.3 KB |
