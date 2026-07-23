# Research problem, questions, and falsifiable claim

Bangla dialect resources differ in collection source, annotation convention, orthography, and dialect coverage. When corpora are merged, dialect labels can become entangled with source identity. A classifier or normalizer may then exploit source-specific formatting and vocabulary rather than transferable dialect structure.

The primary task is source-blind dialect-to-Standard-Bangla normalization. At inference time, the system must not receive a gold dialect label or source identifier. Dialect identification is an auxiliary and independently evaluated task. General-language modeling supplies the shared foundation and replay-retention constraint.

The planned central comparison is M3 Boichitro-MoE versus M2 standard shared-expert MoE. The registered primary endpoint is source-independent normalization macro chrF++. A publishable positive claim requires a paired hierarchical confidence interval for M3−M2 entirely above zero and the associated registered randomization result to survive correction. A positive development point estimate would not be enough; the current point estimate is negative.

Supporting questions concern tokenizer fairness, router specialization, source invariance, worst-dialect behavior, replay retention, calibration, robustness, inference efficiency, and native-speaker adequacy. These outcomes cannot replace the primary endpoint.
