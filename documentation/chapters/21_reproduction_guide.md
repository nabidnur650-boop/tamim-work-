# Reproduction guide

The authoritative build order is:

1. Audit immutable local archives.
2. Build the preliminary manifest.
3. Acquire and verify pinned external sources.
4. Build the final dataset and figures.
5. Train and freeze tokenizer candidates.
6. Prepare and verify the pinned pretraining corpus and packed blocks.
7. Train the dense foundation and select continuation/upcycling/router pilots.
8. Run the three-seed M0–M3 task matrix.
9. Summarize development results and select only registered development artifacts.
10. Complete remaining bidirectional, ablation, and external studies.
11. Freeze code, configurations, data, tokenizer, checkpoints, calibration, and selection manifests.
12. Run one scripted locked evaluation, robustness, routing, statistics, and human evaluation.

The copied evidence snapshot retains the commands, configuration files, source code, reports, and run metadata needed to understand this sequence. Actual reproduction still requires the omitted data and model artifacts.
