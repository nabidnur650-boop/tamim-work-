# Submission notes

This folder is the compact submission copy of the Boichitro-MoE experiment.
The original 91 GB workspace was not modified.

## Included

- Python source code and experiment tools
- registered YAML configurations
- tests
- frozen tokenizer and its lightweight metadata
- research plans and manuscript files
- lightweight metrics, reports, tables, and figures
- exploratory notebooks
- dataset card and source-license ledger

## Excluded

- raw and processed datasets
- model checkpoints and optimizer states
- training runs and caches
- full prediction files and intermediate artifacts
- local ZIP inputs
- unfinished human-review rows

These exclusions keep the repository small enough for ordinary GitHub use and
avoid redistributing third-party or review-sensitive data. The excluded
artifacts remain available only in the original workspace.

## Suggested GitHub settings

- Repository name: `boichitro-moe`
- Visibility: private until the data and human-review publication requirements
  are resolved
- Description: `Reproducible Bangla dialect normalization and identification
  experiments with dense and mixture-of-experts models`
