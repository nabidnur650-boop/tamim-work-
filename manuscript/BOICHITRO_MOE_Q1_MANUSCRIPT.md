# Boichitro-MoE: Source-Invariant Sparse Small Language Modeling for Bangla Dialect Normalization

**Manuscript type:** Original research article
**Authors:** AUTHOR DETAILS REQUIRED BEFORE SUBMISSION
**Affiliations:** AFFILIATION DETAILS REQUIRED BEFORE SUBMISSION
**Corresponding author:** REQUIRED BEFORE SUBMISSION
**Draft generated:** 2026-07-21T21:12:07.382633+00:00

## Submission status (not part of the blinded manuscript)

- Registered pipeline state: **RUNNING**
- Complete main development runs detected: **7/12**
- Complete locked M0–M3 × three-seed evidence: **no**
- Native dataset-review completion: **0.0%**
- Paired PNG/PDF figures: **44**
- Author identities, affiliations, contributions, funding, conflicts, and target-journal formatting: **required before submission**

This is a journal-neutral scientific draft. “Q1” is not an acceptance claim: journal quartiles change over time and editorial decisions remain external. Numerical confirmatory claims are included only when the frozen locked-evaluation manifests exist.

## Abstract

Bangla dialect technology is frequently evaluated on merged corpora in which dialect identity and dataset source are entangled, creating a risk that systems learn collection artifacts instead of transferable dialect structure. We present Boichitro-MoE, a compact task-aware mixture-of-experts decoder designed for source-blind dialect-to-Standard-Bangla normalization and causal dialect identification. The study begins with a provenance-first reconstruction of local and public resources, connected-component leakage controls, protected source-held-out evaluation, and an explicit quarantine of legacy derived data. The frozen benchmark contains 54,598 authentic normalization pairs, 3,325 traceable train-only perturbations, 122,353 conflict-cleaned identification rows, and 1,342 real romanized challenge items. A 300.0M-token dense Bangla foundation is continued into compute-matched dense, Switch top-1, and shared top-2 MoE controls. Boichitro adds causal dialect-evidence routing, task-conditioned late routing, source-adversarial supervision, and GroupDRO while retaining approximately matched active parameters. Locked neural results are not stated in this draft because the registered computational pipeline has not yet produced all twelve main evaluation manifests. The complete artifact contract saves per-example predictions, calibration, routing traces, hierarchical paired uncertainty estimates, and figure source data. The registered computational studies, protocol freeze, locked evaluation, and preregistered native-speaker review and blinded native evaluation remain incomplete.

**Keywords:** Bangla dialects; dialect normalization; mixture of experts; small language models; source shift; domain invariance; tokenizer fairness; reproducible NLP

## Research highlights

- A provenance-first Bangla dialect benchmark separates IID, source-OOD, external-transcript, and romanized challenge tracks.
- A custom wordpiece_natural_32k tokenizer is frozen through intrinsic and three-seed proxy evaluation without test-set selection.
- Dense, Switch, standard-MoE, and task-aware MoE systems are compared under fixed token budgets and measured active-compute diagnostics.
- Replay-constrained normalization prevents task gains from being reported when general-language retention exceeds the preregistered guard.
- Every empirical figure has paired high-resolution PNG/vector PDF output and machine-readable source data.

## 1. Introduction

Bangla is a pluricentric, regionally diverse language whose written dialect resources differ sharply in provenance, annotation practice, orthographic convention, and source domain. These differences create both an opportunity and a methodological hazard. A model may appear to recognize a dialect while exploiting a dataset signature, or it may normalize familiar source templates without learning transformations that transfer to an independent source. This problem is especially acute when published datasets are merged without connected-component deduplication, ancestry tracking, or a protected source-held-out test.

Dialect normalization is a stronger test than classification alone because the system must preserve meaning while producing fluent Standard Bangla under an unknown-dialect input contract. At the same time, a compact deployable system must allocate limited capacity across shared lexical, syntactic, and regional phenomena. Sparse mixture-of-experts models provide conditional capacity, but ordinary routers are free to specialize by source artifacts. Our central question is therefore not simply whether an MoE is larger or more accurate. We ask whether a task-aware sparse small language model can encode dialect-relevant variation while suppressing source shortcuts at approximately matched token-active compute.

The study is designed around falsifiability. The primary architectural comparison is Boichitro-MoE (M3) versus a standard shared-expert top-2 MoE (M2). The primary endpoint is macro chrF++ on the locked source-independent normalization track. The claim is supported only if the registered hierarchical paired confidence interval for M3 − M2 lies entirely above zero; a positive point estimate is insufficient. IID performance, worst-dialect performance, replay retention, identification, calibration, robustness, routing specialization, efficiency, and blinded human ratings are supporting evidence rather than substitutes for the primary endpoint.

### 1.1 Contributions

1. We reconstruct a traceable Bangla dialect benchmark with immutable row identifiers, licenses, source versions, semantic groups, explicit exclusion reasons, and train/evaluation firewalls.
2. We introduce a staged sparse decoder whose router receives causal dialect evidence early and task-conditioned bias late, while a gradient-reversal source head penalizes source-predictive representations.
3. We provide compute-matched dense, Switch, and standard-MoE controls, validation-only architecture/optimizer pilots, three-seed main experiments, registered factorial and confirmatory ablations, and pinned external baselines.
4. We use a one-way protocol freeze before neural test access and retain per-example predictions for paired hierarchical inference.
5. We release an auditable publication bundle with 44 paired figure exports at the time this draft was generated.

## 2. Related work

### 2.1 Bangla dialect resources and evaluation

Vashantor provides aligned regional and Standard-Bangla material; BanglaDial aggregates regional text for classification; ChatgaiyyaAlap contributes Chittagonian conversational pairs; BhasaBodh and related BanglaNLP resources add romanized or benchmark evidence. These resources are valuable but cannot be treated as independent merely because their filenames differ. The present work emphasizes ancestry recovery, exact/compact/near-duplicate controls, and source-held-out evaluation. BanglaBERT and multilingual encoders serve as external identification references, while sequence-to-sequence baselines are pinned before locked evaluation.

### 2.2 Sparse language models

Shared-expert and routed-expert designs increase total capacity while limiting active computation. DeepSeekMoE motivates shared experts and specialization; auxiliary-loss-free balancing avoids a training objective that can distort routing; upcycling transfers dense checkpoints into sparse models. OLMoE and related open studies highlight the need to report active parameters, total parameters, load balance, routing entropy, throughput, and stability rather than parameter count alone. Boichitro differs by making source invariance and task-conditioned routing explicit experimental targets in a low-resource dialect setting.

### 2.3 Tokenization and compact-model optimization

Tokenizer choice changes sequence length, dialect parity, memory, and the interpretation of token-normalized perplexity. We therefore compare WordPiece, Unigram, and byte-BPE candidates using tokens per character/byte, fertility, unknown rate, round-trip behavior, dialect dispersion, and a fixed-budget proxy language model. The final model uses Muon for eligible hidden matrices and AdamW for embeddings, norms, routers, and task heads, with an AdamW-only confirmatory control selected by a development-only LR pilot.

## 3. Data and benchmark construction

### 3.1 Frozen task inventory

The normalization task maps regional input to Standard Bangla under a source-blind `<dial_unknown>` prompt. Authentic and traceable perturbation rows total 57,923; only 3,325 perturbations are synthetic, and all are train-only. The identification task contains 122,353 rows across twelve regional labels plus Standard Bangla. A separate 1,342-row track evaluates real romanized inputs.

### 3.2 Provenance and exclusion controls

Each admitted row records provider, DOI/version, license, source row, dialect taxonomy, normalized text, semantic group, split origin, evaluation track, quality tier, and synthetic ancestry where relevant. The complete legacy derived archive (51,101 rows) is quarantined from the main training pool. Exact and compact duplicates, cross-label text conflicts, protected evaluation relatives, and quality failures are recorded rather than silently dropped. SimHash and character n-gram controls supplement exact matching.

### 3.3 Evaluation tracks

The protocol separates local group-IID tests from source-held-out normalization, source-OOD identification, external transcripts, and romanized challenges. RAJ normalization is a zero-shot challenge and is not silently averaged into the trained-dialect endpoint. Test examples are unavailable to neural model selection, calibration fitting, schedule choice, tokenizer choice, and ablation pruning.

### 3.4 Human data review

A stratified 230-row review packet tests dialect authenticity, source/target validity, fluency, and label quality. At draft generation, 0/230 rows were completed. Until this reaches 230/230, the corpus may support internal experiments but must not be described as fully linguistically validated or publicly redistribution-ready.

## 4. Tokenizer and general-language foundation

The selected tokenizer is `wordpiece_natural_32k` with an actual vocabulary of 32,000. Selection combined intrinsic efficiency and dialect-parity gates with three proxy seeds trained for two million tokens each. Test data were not used for selection. Bits per character are reported for cross-tokenizer comparisons because token-level perplexity is not comparable across different vocabularies.

The general-language source is the revision-pinned Bengali subset of FineWeb-2. After Unicode, Bengali-ratio, length, and direct/compact benchmark decontamination, 567,746 documents were accepted from 585,619 examined. The fixed training budget contains 300,004,991 tokens, with a disjoint 3,000,629-token validation set. Packed block order and all source hashes are immutable.

## 5. Model

### 5.1 Shared decoder backbone

All main systems use a 16-layer causal decoder with width 512, grouped-query attention (8 query heads, 2 key/value heads), RoPE, QK normalization, RMSNorm, a frozen custom tokenizer, and multi-token prediction. M0 is dense. M1 uses Switch top-1 routing. M2 uses one shared expert plus top-2 of eight routed experts. M3 begins from the same M2 continuation checkpoint and adds the proposed task/dialect routing signals.

### 5.2 Boichitro routing

The first four blocks remain dense. Sparse layers 5–8 receive a causal dialect-evidence curriculum derived only from the input prefix visible at that position. Middle sparse layers use learned loss-free load bias. Layers 13–16 add a task-conditioned router bias. Every sparse layer retains one shared expert and activates two routed experts. A causal dialect head supplies auxiliary regional supervision, while a gradient-reversal source head discourages representations that identify the dataset source. GroupDRO reweights observed dialect/source/authenticity groups within the registered bounds.

### 5.3 Active-compute controls

| System | Total parameters | Active parameters/token | Active fraction | Tokens/s | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|
| M0_DENSE | 83.8M | 83.8M | 100.0% | 15,337 | 2.83 |
| M1_SWITCH | 381.1M | 83.8M | 22.0% | 5,211 | 5.69 |
| M2_STANDARD_MOE | 168.8M | 83.8M | 49.7% | 8,272 | 3.84 |
| M3_BOICHITRO | 168.8M | 83.9M | 49.7% | 8,224 | 3.84 |

The benchmark uses batch size 4 and sequence length 512 on the NVIDIA GB10. Total capacity differs, but M0–M3 activate approximately 83.8M parameters per token. We report throughput and memory because nominal active parameters do not capture routing and kernel overhead.

## 6. Training and specialization

The dense foundation is trained for 300M tokens. M0, M1, and M2 then receive matched 200M-token continuations from the same foundation and deterministic block order. High-LR mature-checkpoint restart behavior is retained as a negative result; a validation-only pilot selected a lower continuation LR before full runs.

Task adaptation uses three seeds (1701, 2903, 4307). Stage A consumes 12M fixed tokens from general replay, dialect CLM, normalization, and romanized material. Stage S consumes 6M fixed tokens and selects normalization checkpoints only if replay degradation remains at or below 5%. The selected schedule is `ret35_balanced`; its pilot checkpoint reached 41.19 validation macro chrF++ with 0.97% replay-NLL degradation. The causal ID branch trains for at most 20 epochs with early stopping and temperature calibration. A separate bidirectional MNTP/contrastive branch is used only for identification.

## 7. Experimental design

### 7.1 Baselines

Normalization controls include identity copying and a training-only word-rewrite model. The fair rewrite infers a supported normalization dialect from source text; the legacy gold-dialect rewrite is retained only as an explicitly labeled oracle upper bound. Identification controls include character TF–IDF SVM/SGD systems. Main neural controls are M0 dense, M1 Switch, and M2 standard shared-expert MoE. A fixed validation-selected, source-blind candidate selector and neural/SVM probability blend are reported as separate system views, never substituted for raw neural outputs in architectural inference. Pinned external systems are evaluated by the same locked scripts and split contracts.

### 7.2 Development-only experiments

All architecture, LR, replay-retention, upcycling, router-repair, optimizer, and 2^4 factorial decisions use training/validation data only. The following table is diagnostic and must not be cited as locked test evidence.

| Model | Complete seeds | Validation macro chrF++ | Replay degradation (%) | Validation regional macro-F1 | ECE-15 |
|---|---:|---:|---:|---:|---:|
| M0 | 3 | 41.15 ± 0.10 | 1.08 ± 0.09 | 0.759 ± 0.007 | 0.097 ± 0.014 |
| M1 | 3 | 42.01 ± 0.24 | 1.90 ± 0.19 | 0.764 ± 0.005 | 0.114 ± 0.007 |
| M2 | 1 | 41.73 | 1.17 | 0.753 | 0.096 |

Fixed source-blind system views (development only):

| System view | Development score | Paired gain | 95% hierarchical CI |
|---|---:|---:|---:|
| Normalization selector V2 | 52.667 macro chrF++ | +2.662 | [2.086, 3.268] |
| Neural/SVM identification blend | 0.8009 regional macro-F1 | +0.0390 | [0.0309, 0.0479] |

References, gold dialects, source IDs, and evaluation-track labels are forbidden fusion inputs. Whole normalization semantic groups are held out together. Because the fusion settings were selected on these development rows, the intervals are selection-conditioned exploratory diagnostics, not confirmatory inference. Raw neural outputs remain primary for architectural inference.

Fixed no-retuning transfer to later architectures (shared validation rows; not independent confirmation):

| Model | Seed | Raw norm chrF++ | Fused norm chrF++ | Fused worst dialect | Raw ID F1 | Fused ID F1 |
|---|---:|---:|---:|---:|---:|---:|
| M2 | 1701 | 41.728 | 54.858 | 36.378 | 0.7533 | 0.8046 |

### 7.3 Locked evaluation and statistics

After all development choices, code/config/data/tokenizer/checkpoint/calibration hashes are frozen. One scripted pass produces normalization, identification, external, robustness, routing, and ablation predictions. Primary intervals use paired global-semantic-group bootstrap resampling: each group receives one multiplicity reused across seeds, architectures, and every dialect realization containing that group. Paired randomization likewise draws one treatment/control swap per global semantic group and reuses it across repeated outputs. Holm correction is applied within registered families. A claim of improvement requires the confidence interval—not merely the mean—to exclude zero in the favorable direction.

## 8. Results

### 8.1 Locked main results

> **Locked-result firewall:** no locked neural evaluation manifests were available when this draft was built. Numerical main-result claims are intentionally withheld until the registered pipeline completes.

If the table is populated, all values are descriptive seed aggregates from frozen evaluation manifests. Inferential claims must be taken from `reports/statistics/locked_test_v1/confirmatory_statistics.csv` and must respect familywise correction.

### 8.2 Dialect-level behavior and calibration

Per-dialect normalization chrF++, identification precision/recall/F1, row-normalized confusion, ECE-15, Brier score, and reliability bins are saved for each run. Worst-present-dialect metrics accompany macro metrics to expose failures hidden by dominant CHI, SYL, or Standard-Bangla groups. Validation and locked confusion matrices are never conflated.

### 8.3 Ablations

The registered factorial study tests dialect evidence, the dialect auxiliary head, source adversary, and task conditioning without test access. Three-seed confirmatory ablations remove or randomize one component at a time, including GroupDRO, synthetic perturbations, replay, Muon, and multi-token prediction. Component necessity is concluded only from the locked three-seed effect family; factorial pilot effects are mechanism-screening evidence.

## 9. Routing, robustness, and efficiency

Routing analysis measures expert-load CV, entropy, dialect–expert normalized mutual information, pairwise dialect JS divergence, task JS divergence, and source predictability across layers. A router that separates datasets but not dialects would contradict the proposed mechanism even if aggregate accuracy improved. Perturbation robustness evaluates registered character, whitespace, punctuation, and code-mixing families by severity. Efficiency is reported as measured end-to-end throughput, memory, total parameters, active parameters, and active fraction.

## 10. Blinded native-speaker evaluation

Machine overlap metrics cannot determine whether a normalization preserves meaning, produces fluent Standard Bangla, or hallucinates unsupported content. The pipeline creates blinded randomized packets for qualified native reviewers covering meaning preservation, fluency, overall quality, and unsupported-content flags. This section must remain a protocol description until completed ratings and inter-rater diagnostics are present. No machine metric is substituted for missing human evidence.

## 11. Reproducibility and artifact availability

The executable pipeline is `tools/run_full_pipeline.py`. It records stage state and separate logs under `reports/pipeline/`. Immutable artifacts include source hashes, licenses, split manifests, exclusion decisions, tokenizer hashes, packed block order, configs, checkpoints, calibration, per-example predictions, routing traces, statistical source tables, environment capture, and publication source data. The Q1 figure suite contains 44 PNG/PDF pairs and a hash manifest. Unit tests cover protocol firewalls, data leakage, tokenizer invariants, parameter ownership, training, evaluation, statistics, robustness, and human-packet handling.

## 12. Limitations

First, the frozen normalization inventory is uneven: only a subset of dialects has independent source-held-out parallel data, and RAJ is zero-shot. Second, published dialect labels sometimes collapse sociolinguistically heterogeneous varieties; the 13-label taxonomy is operational rather than an assertion that boundaries are discrete. Third, the 300M-token foundation is intentionally small and does not test scaling laws. Fourth, automated provenance controls cannot replace community-led linguistic validation. Fifth, the external transcript track covers only available sources and domains. Sixth, sparse routing diagnostics are correlational unless supported by the registered interventions and ablations. Seventh, performance on one written benchmark does not establish spoken-language robustness or suitability for consequential deployment.

## 13. Ethics, identity, and data governance

Dialect labels can stigmatize speakers and may encode geography, class, religion, age, or community membership. Identification outputs should not be used to infer identity, origin, or eligibility. Normalization can erase legitimate linguistic identity; the system is framed as optional transformation, not correction of “incorrect” speech. Raw-source licenses and redistribution conditions are preserved per source. Public redistribution and claims of full linguistic validation remain blocked until the stratified native review is complete. Any release should include the data statement, exclusions, intended uses, prohibited uses, model card, known dialect gaps, and a mechanism for community correction or takedown.

## 14. Conclusion

Boichitro-MoE tests a narrow, falsifiable proposition: task-aware and source-adversarial sparse routing may improve transfer to independent Bangla dialect sources at matched active compute. The contribution is equally the experimental control needed to decide that proposition honestly. Provenance reconstruction, protected OOD evaluation, replay-constrained specialization, one-way protocol freezing, paired inference, and native-speaker validation prevent a positive-looking internal score from becoming an unsupported linguistic claim. If the confirmatory M3 − M2 interval does not exclude zero, the audited benchmark, negative result, and routing diagnosis remain valid contributions.

## Declarations required before submission

- **Author contributions (CRediT):** REQUIRED.
- **Funding:** REQUIRED; state “none” if applicable.
- **Competing interests:** REQUIRED; state “none” if applicable.
- **Ethics/IRB determination for human ratings:** REQUIRED.
- **Informed consent and reviewer compensation:** REQUIRED.
- **Data availability statement:** finalize after license/native-review gate.
- **Code availability statement:** add repository and archival DOI.
- **Generative-AI disclosure:** REQUIRED according to the target journal's current policy.
- **Target-journal formatting and word/figure limits:** REQUIRED after journal selection.

## References (journal formatting to be finalized)

1. *Vashantor: A Large-scale Multilingual Benchmark Dataset for Automated Translation of Bangla Regional Dialects to Bangla Language.* [arXiv:2311.11142](https://arxiv.org/abs/2311.11142).
2. *BanglaDial: A merged and imbalanced text dataset for Bengali regional dialect analysis.* [PubMed Central record](https://pmc.ncbi.nlm.nih.gov/articles/PMC12597015/).
3. *ChatgaiyyaAlap: a Chittagonian–Standard Bangla resource.* [PubMed Central record](https://pmc.ncbi.nlm.nih.gov/articles/PMC11925091/).
4. *BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla.* [ACL Anthology](https://aclanthology.org/2022.findings-naacl.98/).
5. *BhasaBodh.* [ACL Anthology](https://aclanthology.org/2025.banglalp-1.9/).
6. *DeepSeekMoE: Towards Ultimate Expert Specialization.* [arXiv:2401.06066](https://arxiv.org/abs/2401.06066).
7. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.* [arXiv:2408.15664](https://arxiv.org/abs/2408.15664).
8. *Upcycling Large Language Models into Mixture of Experts.* [arXiv:2410.07524](https://arxiv.org/abs/2410.07524).
9. *OLMoE: Open Mixture-of-Experts Language Models.* [arXiv:2409.02060](https://arxiv.org/abs/2409.02060).
10. *Muon is Scalable for LLM Training.* [arXiv:2502.16982](https://arxiv.org/abs/2502.16982).
11. *Tokenizer Choice for LLM Training: Negligible or Crucial?* [ACL Anthology](https://aclanthology.org/2024.findings-naacl.247/).
12. *Training Compute-Optimal Large Language Models.* [arXiv:2203.15556](https://arxiv.org/abs/2203.15556).

## Figure and table provenance

All captions and file hashes are in `figures/q1/figure_manifest.json` and `figures/q1/FIGURE_CAPTIONS.md`. Table source data and LaTeX exports are under `tables/paper/`. Development-only and locked-test figures are explicitly labeled in the figure manifest.
