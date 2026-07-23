# Preliminary Data Gate

Status: RED_REMAINING_GATES

The exact-match build is reproducible and internally consistent, but these artifacts are not the final training manifest.

## Counts

| Item | Count |
|---|---:|
| all_manifest_rows | 136,768 |
| raw_rows | 85,667 |
| aligned_pair_rows | 22,364 |
| bangladial_rows | 63,303 |
| derived_rows | 51,101 |
| exact_components | 109,174 |
| normalization_task_eligible | 22,200 |
| normalization_train_preliminary | 17,190 |
| identification_task_eligible | 65,992 |
| identification_train_preliminary_non_bangladial | 16,985 |
| source_maps_multiple_targets_rows | 145 |
| angle_placeholder_rows | 3,888 |
| bangladial_rows_with_exact_vashantor_match | 15,248 |
| derived_rows_train_eligible | 0 |
| train_rows_in_protected_components | 0 |

## Preliminary normalization train rows

| Dialect | Rows |
|---|---:|
| BAR | 2,745 |
| CHI | 5,101 |
| KHU | 718 |
| MYM | 1,873 |
| NAR | 941 |
| NOA | 1,874 |
| RAN | 1,119 |
| SYL | 2,819 |

## Passed checks

- source_archive_counts: PASS
- derived_default_quarantine: PASS
- protected_exact_component_train_leakage_zero: PASS
- normalization_dialects_eight: PASS

## Remaining gates

- near_duplicate_components: NOT COMPLETE
- template_components: NOT COMPLETE
- bangladial_original_component_reconstruction: NOT COMPLETE
- license_ledger: NOT COMPLETE
- human_quality_audit: NOT COMPLETE
- external_ood_test_frozen: NOT COMPLETE
- general_bangla_pretraining_manifest: NOT COMPLETE

## Deterministic artifacts

| Artifact | SHA-256 |
|---|---|
| canonical_rows | df28845a6a6347250b85fe70e2ca6e0a4fd704cc34ea62f948811b666bc8fcf2 |
| exact_components | ee0623e82d07bb26b628a791e7ebb85f1838cd311fa45b5f9c683c28fb33f0f9 |
| quarantined_rows | 0fd232ff9b8f00faf6e8018a53bb139844424c82f9b79a6b1cd47f573e862764 |
| blocked_rows | b8e0785655a538ff01fa4aad5964b6072ca0d61ea58389c8532852ab36d5e16b |

## Decision

Do not start the final tokenizer or model from this preliminary manifest. Complete near/template deduplication, component provenance, licenses, human review, the OOD test, and the general Bangla corpus; then freeze a new final manifest hash.
