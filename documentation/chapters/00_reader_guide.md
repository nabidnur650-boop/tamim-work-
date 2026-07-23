# Reader guide, scope, and evidence rules

This monograph documents the complete Boichitro-MoE experiment workspace as it existed at **2026-07-24 05:09:34 KST**. It is designed to remain useful after the original 91 GiB workspace is removed. The package preserves explanatory text, configurations, source-code descriptions, run metadata, reports, tables, figure source data, and a complete path-and-size inventory.

The documentation is **not a replacement for the raw experiment**. It does not contain the multi-gigabyte Parquet corpora, packed pretraining blocks, model checkpoints, optimizer state, caches, or full prediction stores. Those artifacts are indexed so that their former role and location remain auditable, but they cannot be reconstructed from this document.

Evidence is labelled conservatively:

- **Development-only** means a result was used for model, schedule, fusion, or checkpoint selection.
- **Prior-test descriptive** identifies classical or smoke outputs created before the neural protocol freeze.
- **Locked evidence** would require an immutable protocol freeze followed by scripted test execution. That evidence is not present.
- **Human evidence** requires qualified native-speaker review. The dataset review is 0/230 complete.

The narrative distinguishes intentions from executions. The research plan and execution blueprint describe registered goals; manifests and reports describe what ran; the current filesystem snapshot determines what was present at documentation time.
