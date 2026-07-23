# Boichitro paper tables

## Dataset Counts

| quantity                                |   count |
|:----------------------------------------|--------:|
| excluded_row_decisions                  |  137386 |
| human_review_sample_rows                |     230 |
| identification_all                      |  122353 |
| legacy_derived_archive_quarantined_rows |   51101 |
| normalization_all                       |   57923 |
| normalization_authentic                 |   54598 |
| normalization_synthetic                 |    3325 |
| romanized_ood                           |    1342 |
| tokenizer_train_unique_texts            |  100236 |

## Normalization Dialect Splits

| dialect   |   train |   validation |   test |   test_ood |
|:----------|--------:|-------------:|-------:|-----------:|
| BAR       |    3135 |          333 |    491 |        608 |
| CHI       |   14564 |         1706 |   1879 |       2096 |
| KHU       |     817 |           89 |    101 |          0 |
| MYM       |    2148 |          250 |    375 |       1384 |
| NAR       |    1067 |          101 |    101 |          0 |
| NOA       |    2149 |          249 |    373 |       1400 |
| RAJ       |       0 |            0 |      0 |       1387 |
| RAN       |    1264 |          159 |    127 |          0 |
| SYL       |   14478 |         1402 |   1586 |       2104 |

## Identification Dialect Splits

| dialect   |   train |   validation |   test |   test_ood |   test_external |
|:----------|--------:|-------------:|-------:|-----------:|----------------:|
| BAR       |    4018 |          413 |    587 |        495 |              65 |
| CHI       |   16886 |         1991 |   2205 |       1654 |             162 |
| KHU       |     668 |           86 |     95 |          0 |               0 |
| KIS       |    7953 |          797 |    793 |          0 |             159 |
| MYM       |    2171 |          302 |    448 |        701 |               0 |
| NAR       |    8009 |          865 |    850 |          0 |             162 |
| NOA       |    1783 |          260 |    387 |        786 |               0 |
| NSD       |    5300 |          576 |    538 |          0 |             108 |
| RAJ       |     705 |           91 |     88 |        717 |               0 |
| RAN       |    6481 |          680 |    698 |          0 |             108 |
| STD       |   18768 |         2251 |   2452 |       1172 |               0 |
| SYL       |   15134 |         1407 |   1617 |       1278 |             283 |
| TAN       |    5021 |          519 |    506 |          0 |             104 |

## Tokenizer Selection

| candidate_id           |   mean_bpc |    std_bpc |   mean_worst_dialect_bpc |   tokens_per_character |   mean_tokens_per_second | selected   |
|:-----------------------|-----------:|-----------:|-------------------------:|-----------------------:|-------------------------:|:-----------|
| byte_bpe_balanced_16k  |    3.38073 | 0.0112921  |                  3.59336 |               0.685902 |                 153304   | False      |
| unigram_balanced_32k   |    2.7482  | 0.011373   |                  2.95175 |               0.238325 |                  72810.8 | False      |
| wordpiece_balanced_32k |    2.66754 | 0.00225104 |                  2.88985 |               0.226834 |                  71745.5 | False      |
| wordpiece_natural_32k  |    2.66    | 0.00883052 |                  2.87529 |               0.2245   |                  74345.5 | True       |

## Model Systems

| model_id        |   total_parameters |   active_parameters_per_token |   tokens_per_second |   peak_memory_gib |
|:----------------|-------------------:|------------------------------:|--------------------:|------------------:|
| M0_DENSE        |           83787290 |                      83787290 |            15336.6  |           2.82769 |
| M1_SWITCH       |          381107738 |                      83836442 |             5211.05 |           5.69047 |
| M2_STANDARD_MOE |          168771098 |                      83836442 |             8272.48 |           3.83799 |
| M3_BOICHITRO    |          168794695 |                      83860039 |             8224.14 |           3.84474 |

## Continuation Learning Rate Pilot

| id                   |   muon_lr |   adamw_lr |   initial_validation_bpc |   final_validation_bpc |   maximum_post_start_validation_bpc |   final_relative_bpc_regression |   maximum_relative_bpc_regression | eligible   | resolved_config                                                       | validation_log                                                    | training_report                                                   | final_checkpoint                                                 | selected   |
|:---------------------|----------:|-----------:|-------------------------:|-----------------------:|------------------------------------:|--------------------------------:|----------------------------------:|:-----------|:----------------------------------------------------------------------|:------------------------------------------------------------------|:------------------------------------------------------------------|:-----------------------------------------------------------------|:-----------|
| muon_1e3_adamw_1p5e5 |     0.001 |    1.5e-05 |                  1.23033 |                1.22681 |                             1.22795 |                     -0.00285835 |                       -0.00193151 | True       | reports/model/continuation_lr_pilot_configs/muon_1e3_adamw_1p5e5.yaml | runs/P_CONT_LR_MUON_1E3_ADAMW_1P5E5_10M/1701/validation_log.jsonl | runs/P_CONT_LR_MUON_1E3_ADAMW_1P5E5_10M/1701/training_report.json | runs/P_CONT_LR_MUON_1E3_ADAMW_1P5E5_10M/1701/final_checkpoint.pt | True       |
| muon_2e3_adamw_3e5   |     0.002 |    3e-05   |                  1.23033 |                1.23206 |                             1.23206 |                      0.00140597 |                        0.00140597 | False      | reports/model/continuation_lr_pilot_configs/muon_2e3_adamw_3e5.yaml   | runs/P_CONT_LR_MUON_2E3_ADAMW_3E5_10M/1701/validation_log.jsonl   | runs/P_CONT_LR_MUON_2E3_ADAMW_3E5_10M/1701/training_report.json   | runs/P_CONT_LR_MUON_2E3_ADAMW_3E5_10M/1701/final_checkpoint.pt   | False      |
| muon_5e3_adamw_7p5e5 |     0.005 |    7.5e-05 |                  1.23033 |                1.25411 |                             1.25411 |                      0.0193263  |                        0.0193263  | False      | reports/model/continuation_lr_pilot_configs/muon_5e3_adamw_7p5e5.yaml | runs/P_CONT_LR_MUON_5E3_ADAMW_7P5E5_10M/1701/validation_log.jsonl | runs/P_CONT_LR_MUON_5E3_ADAMW_7P5E5_10M/1701/training_report.json | runs/P_CONT_LR_MUON_5E3_ADAMW_7P5E5_10M/1701/final_checkpoint.pt | False      |
| muon_1e2_adamw_1p5e4 |     0.01  |    0.00015 |                  1.23033 |                1.29338 |                             1.29338 |                      0.0512441  |                        0.0512441  | False      | reports/model/continuation_lr_pilot_configs/muon_1e2_adamw_1p5e4.yaml | runs/P_CONT_LR_MUON_1E2_ADAMW_1P5E4_10M/1701/validation_log.jsonl | runs/P_CONT_LR_MUON_1E2_ADAMW_1P5E4_10M/1701/training_report.json | runs/P_CONT_LR_MUON_1E2_ADAMW_1P5E4_10M/1701/final_checkpoint.pt | False      |

## Stage S Retention Pilot

| candidate          |   normalization_fraction |   replay_fraction |   muon_lr |   adamw_lr |   selected_step |   macro_chrfpp |   worst_dialect_chrfpp |   replay_degradation_percent |   eligible_checkpoints | selected   |
|:-------------------|-------------------------:|------------------:|----------:|-----------:|----------------:|---------------:|-----------------------:|-----------------------------:|-----------------------:|:-----------|
| rejected_default   |                     0.55 |              0.1  |     0.006 |    0.0001  |             228 |        44.7135 |                34.2144 |                    15.9097   |                      0 | False      |
| ret25_balanced     |                     0.4  |              0.25 |     0.002 |    3e-05   |              44 |        41.158  |                31.9175 |                     1.3365   |                      6 | False      |
| ret25_conservative |                     0.4  |              0.25 |     0.001 |    1.5e-05 |             132 |        41.0717 |                30.6214 |                     0.830502 |                      6 | False      |
| ret35_balanced     |                     0.3  |              0.35 |     0.002 |    3e-05   |             232 |        41.1859 |                31.9101 |                     0.972356 |                      6 | True       |
| ret35_conservative |                     0.3  |              0.35 |     0.001 |    1.5e-05 |             156 |        40.7453 |                31.2849 |                     0.361816 |                      6 | False      |

## Upcycling Strategy Pilot

| strategy              | run_id            |   initial_validation_bpc |   final_validation_bpc |   maximum_regression_percent | eligible   | selected   |
|:----------------------|:------------------|-------------------------:|-----------------------:|-----------------------------:|:-----------|:-----------|
| abrupt_banked_release | P_M2_BANKED_20M   |                  1.23033 |                1.26218 |                      3.67811 | False      | False      |
| unbanked_transfer     | P_M2_UNBANKED_20M |                  1.39704 |                1.25587 |                      0       | False      | False      |
| random_initialization | P_M2_SCRATCH_20M  |                  3.51962 |                2.55943 |                      0       | False      | False      |
| annealed_cross_bank   | P_M2_ANNEALED_20M |                  1.23033 |                1.26411 |                      2.74516 | False      | False      |
| permanent_paired_bank | P_M2_PAIRED_20M   |                  1.23033 |                1.22872 |                      0       | True       | True       |

## Switch Router Pilot

| strategy                           | run_id                    |   initial_validation_bpc |   final_validation_bpc |   maximum_router_load_cv |   final_router_load_cv | eligible   | selected   |
|:-----------------------------------|:--------------------------|-------------------------:|-----------------------:|-------------------------:|-----------------------:|:-----------|:-----------|
| loss_free_straight_through         | P_M1_LOSS_FREE_ROUTER_10M |                  1.23033 |                1.22701 |                 0.506108 |              0.506108  | False      | False      |
| auxiliary_balance_straight_through | P_M1_AUX_ROUTER_10M       |                  1.23033 |                1.22694 |                 0.261698 |              0.0674601 | True       | True       |

## Development Fusion Uncertainty

| task           |   paired_mean_gain |   confidence_lower_95 |   confidence_upper_95 |   randomization_p_two_sided |   bootstrap_replicates |   randomization_replicates |
|:---------------|-------------------:|----------------------:|----------------------:|----------------------------:|-----------------------:|---------------------------:|
| normalization  |          2.66175   |             2.08575   |             3.26836   |                  0.00019996 |                   5000 |                       5000 |
| identification |          0.0390474 |             0.0309247 |             0.0478967 |                  0.00019996 |                   5000 |                       5000 |
