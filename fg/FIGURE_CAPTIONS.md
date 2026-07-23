# Q1 figure captions

Complete PNG/PDF pairs: **44**

## Figure 1. Tokenizer efficiency–quality frontier

Development-only proxy bits per character versus intrinsic tokenization cost. Error bars show seed standard deviation; the star marks the frozen tokenizer.

Evidence status: `development_only`. Figure ID: `fig_tokenizer_tradeoff`.

## Figure 2. Active-compute systems frontier

Measured GB10 training throughput versus total parameters for compute-matched dense and sparse architectures.

Evidence status: `development_only`. Figure ID: `fig_compute_pareto`.

## Figure 3. Classical dialect-identification floors

Regional macro-F1 of fixed character n-gram classifiers across validation, IID, source-OOD, and external-transcript tracks.

Evidence status: `prior_test_descriptive`. Figure ID: `fig_classical_id_floor`.

## Figure 4. Fair normalization floors and oracle disclosure

Macro chrF++ for source-blind copy/rewrite controls; the legacy gold-dialect rewrite is explicitly labeled as an oracle rather than a deployable comparator.

Evidence status: `prior_test_descriptive`. Figure ID: `fig_classical_norm_floor`.

## Figure 5. Continuation learning-rate stability

Validation BPC change from the mature-checkpoint LR pilot; the rejected warm restart is retained as a negative result.

Evidence status: `development_only`. Figure ID: `fig_continuation_lr_stability`.

## Figure 6. Normalization–retention trade-off

Validation macro chrF++ versus replay-NLL degradation during registered Stage-S schedule selection. The dashed line is the preregistered 5% guard.

Evidence status: `development_only`. Figure ID: `fig_stage_s_retention_tradeoff`.

## Figure 7. Boichitro-MoE architecture

Source-blind task input, dense prefix, sparse decoder blocks, router curricula, auxiliary heads, and hybrid optimization.

Evidence status: `development_only`. Figure ID: `fig_architecture`.

## Figure 8. Leakage-safe experimental protocol

Registered development sequence and immutable freeze boundary preceding one scripted locked evaluation.

Evidence status: `development_only`. Figure ID: `fig_protocol_flow`.

## Figure 9. Foundation and continuation learning curves

Smoothed training loss and held-out BPC for the dense foundation and matched 200M-token continuations.

Evidence status: `development_only`. Figure ID: `fig_training_curves`.

## Figure 10. Dense-to-MoE recovery pilot

Validation-only comparison of bank release and initialization strategies under a matched 20M-token budget.

Evidence status: `development_only`. Figure ID: `fig_upcycling_recovery`.

## Figure 11. Switch-router failure and repair

Held-out BPC and expert-load variation for the disclosed collapsed run and two registered repair candidates.

Evidence status: `development_only`. Figure ID: `fig_switch_router_recovery`.

## Figure 12. Dialect and split balance

Final admitted rows by dialect and split for the two supervised tasks; stacked segments expose sparse labels and held-out coverage.

Evidence status: `development_only`. Figure ID: `fig_data_split_dialect_balance`.

## Figure 13. Source composition

Largest provenance sources and their share of admitted normalization and identification rows.

Evidence status: `development_only`. Figure ID: `fig_data_source_composition`.

## Figure 14. Input-length distributions

Character-length distributions by dialect after frozen cleaning; boxes omit display outliers but source data retain them.

Evidence status: `development_only`. Figure ID: `fig_data_text_length`.

## Figure 15. Registered evaluation-track composition

Row counts in the frozen IID, source-held-out, external, and challenge tracks; only the twelve largest labels per task are displayed.

Evidence status: `development_only`. Figure ID: `fig_data_evaluation_tracks`.

## Figure 16. Synthetic-data footprint

Authentic and traceable train-only perturbation rows by dialect; synthetic rows are not evaluation evidence.

Evidence status: `development_only`. Figure ID: `fig_data_synthetic_footprint`.

## Figure 17. Romanized challenge coverage

Dialect support and median sequence lengths for the frozen real romanized source-held-out track.

Evidence status: `development_only`. Figure ID: `fig_data_romanized_coverage`.

## Figure 18. Dataset quality tiers

Frozen provenance/quality tiers for admitted task rows, retaining explicit uncertainty rather than imputing provenance.

Evidence status: `development_only`. Figure ID: `fig_data_quality_tiers`.

## Figure 19. Source–dialect coverage matrix

Log-scaled row counts for the twelve largest normalization sources, making missing source–dialect cells explicit.

Evidence status: `development_only`. Figure ID: `fig_data_source_dialect_coverage`.

## Figure 20. Tokenizer vocabulary scaling

Intrinsic token cost and fertility across family, corpus balance, and vocabulary size.

Evidence status: `development_only`. Figure ID: `fig_tokenizer_vocab_scaling`.

## Figure 21. Tokenizer cost by dialect

Tokens per character for the validation-shortlisted candidates across the frozen dialect inventory.

Evidence status: `development_only`. Figure ID: `fig_tokenizer_dialect_fertility`.

## Figure 22. Tokenizer dialect parity

Worst-to-best cost ratio versus Gini dispersion for every screened tokenizer; highlighted candidates entered the proxy study.

Evidence status: `development_only`. Figure ID: `fig_tokenizer_fairness`.

## Figure 23. Tokenizer proxy stability

Three-seed proxy-language-model quality and throughput for shortlisted tokenizers.

Evidence status: `development_only`. Figure ID: `fig_tokenizer_proxy_stability`.

## Figure 24. Pretraining corpus acceptance gate

Accepted share and mutually recorded rejection reasons for the pinned FineWeb-2 Bengali source.

Evidence status: `development_only`. Figure ID: `fig_pretraining_acceptance`.

## Figure 25. Pretraining shard balance

Token volume and deterministic validation allocation across immutable packed-source shards.

Evidence status: `development_only`. Figure ID: `fig_pretraining_shard_balance`.

## Figure 26. Model parameter decomposition

Token-active and inactive expert capacity for the dense, Switch, shared-expert MoE, and Boichitro architectures.

Evidence status: `development_only`. Figure ID: `fig_model_parameter_decomposition`.

## Figure 27. Measured systems efficiency

GB10 throughput, peak memory, and active parameter fraction under the same batch and sequence length.

Evidence status: `development_only`. Figure ID: `fig_model_throughput_memory`.

## Figure 28. Foundation continuation loss components

Logged LM and multi-token-prediction losses for compute-matched 200M-token continuations.

Evidence status: `development_only`. Figure ID: `fig_foundation_loss_components`.

## Figure 29. Foundation router dynamics

Expert-load variation and router entropy over matched Switch and shared-expert MoE continuation.

Evidence status: `development_only`. Figure ID: `fig_foundation_router_dynamics`.

## Figure 30. Router stability summary

Registered diagnostic summary for load balance, sustained threshold exceedance, and held-out BPC.

Evidence status: `development_only`. Figure ID: `fig_foundation_router_stability`.

## Figure 31. Normalization validation trajectories

Mean and seed dispersion of macro chrF++ at registered Stage-S validation checkpoints for completed main runs.

Evidence status: `development_only`. Figure ID: `fig_task_stage_s_trajectories`.

## Figure 32. Main-run retention trade-off

All main-run normalization checkpoints in development space; the dashed line is the fixed replay guard.

Evidence status: `development_only`. Figure ID: `fig_task_replay_tradeoff`.

## Figure 33. Identification validation trajectories

Mean and seed dispersion of regional macro-F1 during causal identification specialization.

Evidence status: `development_only`. Figure ID: `fig_task_stage_id_trajectories`.

## Figure 34. Main-run seed stability

Individual seeds and mean ± standard deviation at the validation-selected checkpoints.

Evidence status: `development_only`. Figure ID: `fig_task_seed_stability`.

## Figure 35. Validation normalization by dialect

Seed-mean chrF++ at each run's selected normalization checkpoint; blank cells denote unsupported dialects.

Evidence status: `development_only`. Figure ID: `fig_task_norm_dialect_heatmap`.

## Figure 36. Validation identification by class

Seed-mean per-class F1 at validation-selected causal identification checkpoints.

Evidence status: `development_only`. Figure ID: `fig_task_id_class_heatmap`.

## Figure 37. Validation identification confusion structure

Seed-mean row-normalized confusion at selected causal checkpoints, shown only for computationally complete main variants.

Evidence status: `development_only`. Figure ID: `fig_task_id_confusion`.

## Figure 38. Validation calibration

Reliability curves and ECE-15 for temperature-calibrated causal identification checkpoints.

Evidence status: `development_only`. Figure ID: `fig_task_calibration`.

## Figure 39. Task-stage training efficiency

Wall time, throughput, and peak memory for completed normalization and identification stages.

Evidence status: `development_only`. Figure ID: `fig_task_training_efficiency`.

## Figure 40. Task-stage optimization curves

Seed-aggregated smoothed training loss over adaptation, normalization, and causal identification stages.

Evidence status: `development_only`. Figure ID: `fig_task_stage_loss_components`.

## Figure 41. Validation checkpoint selection

Selected checkpoint locations and objectives across seeds, documenting early-selection variation without test access.

Evidence status: `development_only`. Figure ID: `fig_validation_checkpoint_selection`.

## Figure 42. Classical normalization error structure

Per-dialect chrF++ for source-blind controls. Legacy word-rewrite test values use gold dialect and are labeled only as oracle evidence.

Evidence status: `prior_test_descriptive`. Figure ID: `fig_classical_norm_dialect`.

## Figure 43. Classical identification error structure

Per-class F1 for the selected character n-gram SVM across IID, source-OOD, and external-transcript tracks.

Evidence status: `prior_test_descriptive`. Figure ID: `fig_classical_id_class`.

## Figure 44. Development-only source-blind fusion gains

Six paired M0/M1 runs. Normalization uses semantic-group-held-out selector predictions; identification uses the fixed calibrated neural/SVM blend. Settings were selected on development data, so this is exploratory evidence. Diamonds show mean ± seed SD.

Evidence status: `development_only`. Figure ID: `fig_development_source_blind_fusion`.
