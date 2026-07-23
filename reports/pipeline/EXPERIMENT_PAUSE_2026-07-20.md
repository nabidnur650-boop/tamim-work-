# Experiment pause handoff — 2026-07-20

Status: **RESUMED** on `2026-07-21`. The pause itself began at
`2026-07-20T22:01:36+09:00`.

Continuation note: all four registered retention candidates subsequently
completed. `ret35_balanced` was selected on validation, the main Stage-S
contract was updated and hash-checked, and the full pipeline resumed from
`main_task`.

No training process is running. The GPU was released after a clean `SIGINT`.
Locked neural test data has not been accessed. The dataset, tokenizer, foundation
models, completed task-pilot candidates, logs, validation predictions, and
checkpoints remain on disk.

## Completed before pause

- Dataset/tokenizer freeze and leakage audit.
- Dense, Switch, and standard-MoE foundation training and validation pilots.
- Default Stage-S development schedule retained as a rejected negative result:
  all four checkpoints failed the 5% replay guard; its best task point was
  44.7135 macro chrF++ with 15.9097% replay degradation.
- Registered Stage-S retention pilot implementation, fixed candidate config,
  pipeline integration, freeze-manifest integration, and unit tests.
- `ret25_balanced`: complete at 6,000,234 tokens; selected validation result
  41.1580 macro chrF++, 31.9175 worst-dialect chrF++, and 1.3365% replay
  degradation (guard pass).
- `ret25_conservative`: complete at 6,000,234 tokens; selected validation result
  41.0717 macro chrF++, 30.6214 worst-dialect chrF++, and 0.8305% replay
  degradation (guard pass).

## Interrupted work preserved

`ret35_balanced` was stopped at optimizer step 30 / 232 after 781,108 tokens.
Its partial contract, mixture report, trainer config, and log were moved without
deletion to:

`runs/pilots/boichitro_stage_s_retention_pilot_v1/interrupted/ret35_balanced_step30_user_pause_20260720/`

The canonical candidate path is intentionally absent, so the registered pilot
will rerun that candidate cleanly rather than silently treating a non-resumable
partial epoch as complete.

## Exact continuation order

1. From the project root, run:

   `PYTHONPATH=src python tools/run_stage_s_retention_pilot.py`

   The script verifies contracts, skips both completed 25%-replay candidates,
   and runs `ret35_balanced` followed by `ret35_conservative`.

2. Inspect `reports/model/stage_s_retention_pilot_selection.json`. Only a
   replay-guard-passing validation checkpoint can be selected.

3. Apply the selected mixture, Muon/AdamW/router rates, and six-checkpoint
   validation schedule to the main Stage-S protocol; record the selection-report
   hash in the main run manifest.

4. Resume M0–M3 task training for seeds 1701/2903/4307, then the optimizer
   pilot, inference benchmark, bidirectional ID specialist, external baselines,
   factorial and confirmatory ablations.

5. Freeze the protocol before any main neural test access, then run locked
   evaluation once, robustness/routing/statistics, human-evaluation packet
   preparation, and publication figures/tables.

## Frozen pilot code/config hashes at pause

- `configs/stage_s_retention_pilot.yaml`:
  `ed11707e1fec26e53b9a8dda4edbf162659dcad07bb35ddb012c6860f41d236a`
- `tools/run_stage_s_retention_pilot.py`:
  `10f9f0b824cc1a6fc12b64f77ec2db1d5d16bd5efa36b7671b327936929c4806`
- `configs/task_experiments_rejected_stage_s_default.yaml`:
  `f3d5d6e251fc72d9fa348d460b95cf741b62d8a1962d0f97d86ab0d6bbee0da5`

The legacy `reports/pipeline/full_pipeline_state.json` shows `FAILED` because
the earlier full-pipeline process was intentionally interrupted after detecting
the invalid default Stage-S schedule. This pause handoff is the authoritative
status for continuation.
