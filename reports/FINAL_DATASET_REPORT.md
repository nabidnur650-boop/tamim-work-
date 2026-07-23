# Final dataset audit

Status: **PASS_INTERNAL_DATA_ENGINEERING**
Publication status: **CONDITIONAL_NATIVE_REVIEW_REQUIRED**

## Core counts

| Item | Rows |
|---|---:|
| Authentic normalization | 54,598 |
| Synthetic normalization (train only) | 3,325 |
| Identification | 122,353 |
| Romanized OOD | 1,342 |
| Tokenizer unique training text | 100,236 |
| Exclusion decisions | 137,386 |
| Legacy derived ZIP still quarantined | 51,101 |

## Gate results

| gate | pass |
|---|---|
| thirteen_label_inventory_present | 1 |
| inventory_is_twelve_regional_plus_standard | 1 |
| no_compact_train_evaluation_overlap_normalization | 1 |
| no_compact_train_evaluation_overlap_identification | 1 |
| no_cross_label_identification_text | 1 |
| synthetic_train_only | 1 |
| synthetic_excluded_from_identification | 1 |
| locked_ood_sources_never_train | 1 |
| legacy_derived_archive_absent | 1 |
| all_task_licenses_resolved | 1 |
| native_human_review_complete | 0 |

## Exclusion decisions

| task | reason | rows |
|---|---|---|
| identification | cross_label_text_conflict | 20,724 |
| identification | fatal_text_quality | 57 |
| identification | hf_district_outside_frozen_taxonomy | 1,989 |
| identification | same_label_compact_duplicate | 38,142 |
| identification | train_near_protected_evaluation | 2 |
| normalization | conflicting_lower_priority_target | 118 |
| normalization | exact_pair_duplicate_lower_priority | 5,516 |
| normalization | failed_preliminary_local_normalization_gate | 164 |
| normalization | fatal_text_quality | 2 |
| normalization | ood_overlap_with_development | 2,655 |
| normalization | source_maps_multiple_targets_at_best_priority | 729 |
| normalization | train_near_protected_evaluation | 405 |
| romanized_normalization | bengali_parent_removed_from_onubad_ood | 546 |
| romanized_normalization | duplicate_romanized_parent | 72 |
| tokenizer | duplicate_tokenizer_text | 66,200 |
| tokenizer | exact_protected_evaluation_text | 65 |

## Correct interpretation

The corpus has 13 labels, consisting of 12 regional labels and Standard Bangla.
The data-engineering checks pass for internal experimentation. The native review
sheet contains 230 stratified rows and must be
completed before a Q1 manuscript calls the release linguistically validated.
