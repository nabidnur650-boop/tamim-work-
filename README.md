# Boichitro-MoE

Boichitro-MoE is a reproducible research pipeline for Bangla dialect
normalization, dialect identification, tokenizer evaluation, dense-model
baselines, and mixture-of-experts experiments.

This repository is a compact GitHub-ready snapshot of the experiment. It
contains the source code, configurations, tests, research protocol, manuscript,
result summaries, tables, figures, and original notebooks. The 91 GB working
directory remains unchanged outside this repository.

## Important status

- The code and lightweight research outputs are included.
- Raw/processed datasets, model checkpoints, caches, full predictions, and
  training runs are intentionally excluded because of size and redistribution
  constraints.
- The native-speaker review and blinded human evaluation described in
  `reports/NATIVE_REVIEW_INSTRUCTIONS.md` remain publication requirements.
- Dataset sources, versions, licenses, and use decisions are recorded in
  `data/manifests/licenses.yaml` and `data/final/v1/DATASET_CARD.md`.
- No license is granted for third-party datasets or model artifacts by this
  repository.

## Repository contents

- `src/boichitro/`: model, data, metrics, inference, optimization, and protocol
  code
- `tools/`: dataset, training, evaluation, statistics, and reporting commands
- `configs/`: registered experiment and evaluation configurations
- `tests/`: regression and protocol tests
- `artifacts/tokenizers/frozen/`: frozen 32k WordPiece tokenizer and metadata
- `manuscript/`: manuscript draft and submission materials
- `reports/`: lightweight audits, result summaries, and evaluation reports
- `metrics/`, `tables/`, `figures/`: publication-oriented outputs
- `docs/`: experiment blueprint, research plan, and original workspace README
- `notebooks/`: original exploratory notebooks

## Environment

The recorded experiment used Python 3.12. Create an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the available tests from the repository root:

```bash
pytest -q
```

Tests that depend on omitted datasets are skipped automatically in the compact
snapshot and become active when the documented Parquet artifacts are restored.
GitHub Actions runs the same compact-repository test suite on every push and
pull request.

## Reproducing the pipeline

Place the two original local archives beside this repository using their
original names:

```text
archive(1).zip
archive (1).zip
```

Then run the data preparation stages:

```bash
python tools/audit_local_archives.py
python tools/build_preliminary_manifest.py
python tools/fetch_external_sources.py
python tools/build_final_dataset.py
python tools/plot_final_dataset.py
```

After the data, tokenizer, and pretraining caches are available, run the
resumable registered experiment:

```bash
PYTHONPATH=src python tools/run_full_pipeline.py
```

The full protocol and registered design are documented in
`docs/EXPERIMENT_BLUEPRINT.md` and `docs/Q1_RESEARCH_PLAN.md`.

## Results

Start with:

- `reports/ALL_RESULTS_AND_EVALUATION.md`
- `reports/ALL_RESULTS_AND_EVALUATION.pdf`
- `reports/Q1_JOURNAL_READINESS_AUDIT.md`
- `manuscript/BOICHITRO_MOE_Q1_MANUSCRIPT.md`

Detailed machine-readable summaries are available under `reports/model/`,
`reports/tokenizer/`, and `metrics/`.
