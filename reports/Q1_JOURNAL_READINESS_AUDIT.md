# Q1 journal readiness audit

Status: **INCOMPLETE**

Automated computational pipeline complete: **no**
Required human validation complete: **no**
Submission package complete: **no**
Overall submission-ready: **no**

A Q1 acceptance outcome cannot be guaranteed by an artifact audit. The audit only establishes reproducibility, protocol completion, and whether required human evidence is present.

## Checks

| Result | Scope | Check | Evidence |
|---|---|---|---|
| PASS | computational | Frozen data engineering gate | reports/data_gate.json |
| PASS | computational | Frozen tokenizer selected without test access | reports/tokenizer/TOKENIZER_FREEZE_REPORT.json (TOKENIZER_FROZEN) |
| PASS | computational | Foundation and matched continuations | 4/4 fixed-budget runs complete |
| FAIL | computational | Four models × three main seeds | 7/12 complete (8 run directories discovered) |
| PASS | computational | Fair source-blind baselines and validation-selected system fusion | reports/model/source_blind_normalization_baseline_audit.json, reports/model/id_fusion_selection.json, reports/model/normalization_fusion_selection_v2.json, reports/model/development_fusion_uncertainty.json |
| PASS | computational | No normalization evaluation input appears in ID-classifier training | reports/model/cross_task_input_firewall.json |
| FAIL | computational | Fixed fusion transferred without retuning to all M2/M3 development runs | reports/model/development_fusion_architecture_transfer.json (1/6 complete) |
| FAIL | computational | AdamW-only learning-rate pilot | reports/model/optimizer_pilot_selection.json (missing or invalid JSON) |
| FAIL | computational | Three-seed bidirectional ID specialist | 0/3 run manifests complete |
| FAIL | computational | Registered three-seed confirmatory ablations | 0/24 complete |
| FAIL | computational | Registered optimizer/MTP ablations | 0/6 complete |
| FAIL | computational | Registered 2^4 development factorial | 0/16 selected validation cells |
| FAIL | computational | Immutable protocol freeze before neural test access | no freeze manifest found |
| FAIL | computational | Scripted locked main evaluation | 0/12 evaluation manifests |
| FAIL | computational | Locked registered ablation evaluation | 0/30 or more evaluation manifests |
| FAIL | computational | Pinned external baselines | 0 locked external manifests |
| FAIL | computational | Registered perturbation robustness | reports/robustness/locked_robustness_v1/robustness_curves.csv |
| FAIL | computational | Locked routing specialization analysis | reports/routing/locked_test_v1/expert_specialization_metrics.parquet |
| FAIL | computational | Confirmatory uncertainty and multiplicity control | reports/statistics/locked_test_v1/confirmatory_statistics.csv |
| PASS | computational | At least 30 paired journal figures | manifest=44; paired files=44; figures/q1/figure_manifest.json |
| PASS | computational | Figure hash, format, resolution, caption, and source-data validation | figures/q1/VALIDATION_REPORT.json (398/398) |
| PASS | computational | Full regression test suite | reports/Q1_TEST_REPORT.json (107 passed) |
| PASS | computational | Reproducible paper tables | 10 table entries in tables/paper/table_manifest.json |
| FAIL | computational | Registered end-to-end pipeline | reports/pipeline/full_pipeline_state.json (RUNNING) |
| FAIL | human_submission | Stratified native-speaker dataset review | reports/native_review_report.json (0/230) |
| FAIL | human_submission | Blinded native-speaker system-output ratings | human_evaluation/blind_native_normalization_v1/human_evaluation_summary.csv |
| PASS | submission_package | Journal-neutral manuscript draft | manuscript/BOICHITRO_MOE_Q1_MANUSCRIPT.md |
| FAIL | submission_package | Author, affiliation, CRediT, funding, and conflict metadata | manuscript/AUTHOR_METADATA.yaml |
| FAIL | submission_package | Target journal and current author-guideline adaptation | manuscript/TARGET_JOURNAL.md |

## Remaining blockers

- **Four models × three main seeds** — 7/12 complete (8 run directories discovered)
- **Fixed fusion transferred without retuning to all M2/M3 development runs** — reports/model/development_fusion_architecture_transfer.json (1/6 complete)
- **AdamW-only learning-rate pilot** — reports/model/optimizer_pilot_selection.json (missing or invalid JSON)
- **Three-seed bidirectional ID specialist** — 0/3 run manifests complete
- **Registered three-seed confirmatory ablations** — 0/24 complete
- **Registered optimizer/MTP ablations** — 0/6 complete
- **Registered 2^4 development factorial** — 0/16 selected validation cells
- **Immutable protocol freeze before neural test access** — no freeze manifest found
- **Scripted locked main evaluation** — 0/12 evaluation manifests
- **Locked registered ablation evaluation** — 0/30 or more evaluation manifests
- **Pinned external baselines** — 0 locked external manifests
- **Registered perturbation robustness** — reports/robustness/locked_robustness_v1/robustness_curves.csv
- **Locked routing specialization analysis** — reports/routing/locked_test_v1/expert_specialization_metrics.parquet
- **Confirmatory uncertainty and multiplicity control** — reports/statistics/locked_test_v1/confirmatory_statistics.csv
- **Registered end-to-end pipeline** — reports/pipeline/full_pipeline_state.json (RUNNING)
- **Stratified native-speaker dataset review** — reports/native_review_report.json (0/230)
- **Blinded native-speaker system-output ratings** — human_evaluation/blind_native_normalization_v1/human_evaluation_summary.csv
- **Author, affiliation, CRediT, funding, and conflict metadata** — manuscript/AUTHOR_METADATA.yaml
- **Target journal and current author-guideline adaptation** — manuscript/TARGET_JOURNAL.md

## Claim discipline

- Do not describe the corpus as fully linguistically validated until the 230-row native review is complete.
- Do not substitute machine metrics for blinded native-speaker system ratings.
- Report validation-only pilot and ablation evidence separately from locked-test confirmatory evidence.
- Do not describe the work as accepted by, or guaranteed suitable for, a Q1 journal; quartile and editorial decisions are external.
