# Executive summary

Boichitro-MoE is a provenance-first study of Bangla dialect normalization and dialect identification under source shift. The central scientific question is whether a compact, task-aware sparse decoder can learn dialect-relevant structure without using dataset-source shortcuts, while maintaining approximately matched token-active parameters against dense and sparse controls.

The data pipeline produced `boichitro_data_v1.0.0`, with 54,598 authentic normalization rows, 3,325 traceable train-only perturbations, 122,353 conflict-cleaned identification rows, 1,342 romanized source-held-out items, and 100,236 unique tokenizer-training texts. Automated engineering gates authorize internal modeling. Public redistribution and claims of linguistic validation remain blocked by the uncompleted native-speaker review.

The model family shares a 16-layer, width-512 causal decoder, grouped-query attention, RoPE, QK normalization, RMSNorm, a frozen 32k WordPiece tokenizer, and multi-token prediction. M0 is dense, M1 is Switch top-1, M2 is a shared-expert top-2 MoE, and M3 adds causal dialect evidence, task-conditioned routing, source-adversarial supervision, and GroupDRO.

At this documentation snapshot, all twelve M0–M3 development run manifests are present. This supersedes the older manuscript/readiness statement of seven completed runs. Development means are 41.951 versus 40.175 macro chrF++ for M2 and M3, and 0.7594 versus 0.7510 regional macro-F1. Thus the current development evidence does not support a claim that M3 improves on M2. Locked evaluation, registered confirmatory statistics, robustness, routing analysis, and human system ratings remain absent.

The strongest completed assets are the audited data pipeline, tokenizer freeze, dense foundation and continuation pilots, full three-seed development matrix, source-blind development fusion, systems benchmarks, 44 validated publication figures, ten reproducible paper-table families, and 107 recorded passing regression tests.
