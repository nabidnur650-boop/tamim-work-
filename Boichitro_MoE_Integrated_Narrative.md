# Boichitro-MoE Full Experiment Technical Documentation

Documentation snapshot: **2026-07-24 05:09:34 KST**

> This Markdown edition contains the integrated narrative. The PDF/DOCX editions also contain the full visual atlas and all evidence cards.

## Reader guide, scope, and evidence rules

This monograph documents the complete Boichitro-MoE experiment workspace as it existed at **2026-07-24 05:09:34 KST**. It is designed to remain useful after the original 91 GiB workspace is removed. The package preserves explanatory text, configurations, source-code descriptions, run metadata, reports, tables, figure source data, and a complete path-and-size inventory.

The documentation is **not a replacement for the raw experiment**. It does not contain the multi-gigabyte Parquet corpora, packed pretraining blocks, model checkpoints, optimizer state, caches, or full prediction stores. Those artifacts are indexed so that their former role and location remain auditable, but they cannot be reconstructed from this document.

Evidence is labelled conservatively:

- **Development-only** means a result was used for model, schedule, fusion, or checkpoint selection.
- **Prior-test descriptive** identifies classical or smoke outputs created before the neural protocol freeze.
- **Locked evidence** would require an immutable protocol freeze followed by scripted test execution. That evidence is not present.
- **Human evidence** requires qualified native-speaker review. The dataset review is 0/230 complete.

The narrative distinguishes intentions from executions. The research plan and execution blueprint describe registered goals; manifests and reports describe what ran; the current filesystem snapshot determines what was present at documentation time.

## Executive summary

Boichitro-MoE is a provenance-first study of Bangla dialect normalization and dialect identification under source shift. The central scientific question is whether a compact, task-aware sparse decoder can learn dialect-relevant structure without using dataset-source shortcuts, while maintaining approximately matched token-active parameters against dense and sparse controls.

The data pipeline produced `boichitro_data_v1.0.0`, with 54,598 authentic normalization rows, 3,325 traceable train-only perturbations, 122,353 conflict-cleaned identification rows, 1,342 romanized source-held-out items, and 100,236 unique tokenizer-training texts. Automated engineering gates authorize internal modeling. Public redistribution and claims of linguistic validation remain blocked by the uncompleted native-speaker review.

The model family shares a 16-layer, width-512 causal decoder, grouped-query attention, RoPE, QK normalization, RMSNorm, a frozen 32k WordPiece tokenizer, and multi-token prediction. M0 is dense, M1 is Switch top-1, M2 is a shared-expert top-2 MoE, and M3 adds causal dialect evidence, task-conditioned routing, source-adversarial supervision, and GroupDRO.

At this documentation snapshot, all twelve M0–M3 development run manifests are present. This supersedes the older manuscript/readiness statement of seven completed runs. Development means are 41.951 versus 40.175 macro chrF++ for M2 and M3, and 0.7594 versus 0.7510 regional macro-F1. Thus the current development evidence does not support a claim that M3 improves on M2. Locked evaluation, registered confirmatory statistics, robustness, routing analysis, and human system ratings remain absent.

The strongest completed assets are the audited data pipeline, tokenizer freeze, dense foundation and continuation pilots, full three-seed development matrix, source-blind development fusion, systems benchmarks, 44 validated publication figures, ten reproducible paper-table families, and 107 recorded passing regression tests.

## Status reconciliation and chronology

Several artifacts were generated at different times and therefore disagree. The older Q1 readiness audit was created at `2026-07-21T21:13:32.502992+00:00` and recorded `7/12 complete (8 run directories discovered)`. The later development snapshot records 12/12 complete runs, and the filesystem contains 12 main run manifests.

The pipeline state file was last modified at `2026-07-23T07:15:29.847532+09:00` and still reports `RUNNING`. At documentation time, 0 matching live training processes were observed. The bidirectional specialist has 2/3 completed manifests; seed 4307 contains partial training evidence but no completed run manifest.

The correct interpretation is therefore:

1. Main M0–M3 development training is complete for three seeds each.
2. The manuscript and older audit are stale with respect to those main runs.
3. The end-to-end pipeline is not complete.
4. No immutable protocol-freeze manifest or locked neural evaluation set is present.
5. The bidirectional branch is partially complete.
6. Human validation remains unstarted.

All later claims in this monograph use this reconciled chronology.

## Research problem, questions, and falsifiable claim

Bangla dialect resources differ in collection source, annotation convention, orthography, and dialect coverage. When corpora are merged, dialect labels can become entangled with source identity. A classifier or normalizer may then exploit source-specific formatting and vocabulary rather than transferable dialect structure.

The primary task is source-blind dialect-to-Standard-Bangla normalization. At inference time, the system must not receive a gold dialect label or source identifier. Dialect identification is an auxiliary and independently evaluated task. General-language modeling supplies the shared foundation and replay-retention constraint.

The planned central comparison is M3 Boichitro-MoE versus M2 standard shared-expert MoE. The registered primary endpoint is source-independent normalization macro chrF++. A publishable positive claim requires a paired hierarchical confidence interval for M3−M2 entirely above zero and the associated registered randomization result to survive correction. A positive development point estimate would not be enough; the current point estimate is negative.

Supporting questions concern tokenizer fairness, router specialization, source invariance, worst-dialect behavior, replay retention, calibration, robustness, inference efficiency, and native-speaker adequacy. These outcomes cannot replace the primary endpoint.

## Historical audit and protocol repairs

The project began by auditing two local ZIP archives and two notebooks. The audit found publication-blocking problems in the earlier pipeline: a Vashantor column mismatch, held-out rows reintroduced through derived synthetic data, a missing Barishal validation file, duplicate loading of a regional corpus, template-like synthetic conflicts, uncertain ancestry inside merged BanglaDial material, model–data scale mismatch, and external diagnostics that did not support valid comparative claims.

The repaired workflow treats the original archives as immutable inputs. Every source is handled through an explicit adapter. The old derived archive is quarantined instead of silently reused. Source ancestry, licenses, dialect mapping, row origin, exclusion reason, split membership, semantic groups, and synthetic parentage are recorded in machine-readable manifests.

The monolithic notebooks are retained as historical material, not as the authoritative executable pipeline. Reusable modules under `src/boichitro`, command-line tools, YAML registries, regression tests, immutable hashes, and saved per-example outputs form the reproducible implementation.

## Data provenance, taxonomy, and canonical schema

The frozen taxonomy has thirteen labels: twelve regional varieties plus Standard Bangla. Labels are BAR, CHI, KHU, KIS, MYM, NAR, NOA, NSD, RAJ, RAN, SYL, TAN, and STD. Not every source supports every label, and the documentation preserves those missing cells rather than imputing coverage.

Canonical records include immutable row identifiers, task, source/provider, source version, source row, license record, dialect code, regional input, Standard-Bangla target when available, normalized text views, semantic component, original split provenance, frozen split, evaluation track, quality tier, synthetic flag, and ancestry.

Source adapters separate local Vashantor, regional corpora, Chatgaiyya material, Sylheti translation, pinned public datasets, external transcripts, romanized pairs, and general-language pretraining text. The data artifact ledger in this package records row counts, byte sizes, and published SHA-256 values for every frozen dataset output.

## Cleaning, deduplication, and leakage controls

Cleaning normalizes Unicode and whitespace while preserving linguistically meaningful content. Fatal text-quality failures are excluded with explicit reasons. The pipeline applies exact pair checks, compact text checks, cross-label conflict removal, source-priority rules, and protected-evaluation ancestry controls.

Near-duplicate protection combines 64-bit SimHash with character 4-gram Jaccard filtering. Candidate pairs within a maximum Hamming distance are checked against a 0.90 Jaccard threshold. Semantic relations are converted into connected components before split assignment, preventing related rows from straddling development and protected evaluation.

The direction of removal is asymmetric by design: training relatives are removed against protected IID evaluation; source-OOD material is protected against all development inputs. A separate cross-task firewall checks that normalization evaluation inputs do not appear among identification-training inputs. Synthetic perturbations are train-only and never constitute evaluation evidence.

## Frozen dataset composition and data gates

The final build contains 57,923 normalization rows, of which 54,598 are authentic and 3,325 are traceable train-only perturbations. The identification view contains 122,353 conflict-cleaned rows. The real romanized source-held-out track contains 1,342 rows. The tokenizer text view contains 100,236 unique texts.

The dataset report records 137,386 exclusion decisions. Large categories include same-label compact duplicates, cross-label identification conflicts, lower-priority exact normalization pairs, protected-evaluation relatives, source taxonomy exclusions, and quality failures. The full exclusion ledger is evidence of conservative filtering, not additional training data.

Automated gates pass for license resolution within the internal task scope, taxonomy, quarantine of the legacy derived archive, source-OOD protection, synthetic train-only use, compact-overlap checks, and cross-label conflict removal. The native-review gate fails because 0/230 sampled rows have been completed. Training is authorized internally; public redistribution and “linguistically validated” claims are not.

## Tokenizer study and immutable freeze

Tokenizer selection compares WordPiece, Unigram, and byte-BPE families at multiple vocabulary sizes and corpus-balance conditions. Intrinsic measures include tokens per character and byte, fertility, unknown rate, round-trip behavior, dialect dispersion, worst-to-best ratios, and Gini-style fairness summaries.

Candidates passing the intrinsic screen enter a fixed-budget proxy language-model study with seeds 1701, 2903, and 4307. Bits per character is the cross-tokenizer quality metric because token-normalized perplexity is not comparable across vocabularies. Throughput and stability provide additional practical evidence.

The selected tokenizer is `wordpiece_natural_32k`, with an actual vocabulary of 32,000. Its frozen files, metadata, and recorded hash are copied into this documentation package. Test data were not used for selection.

## General Bangla pretraining corpus and dense foundation

The general-language source is a revision-pinned Bengali subset of FineWeb-2. Source verification records immutable revisions and file hashes. Filtering applies Unicode checks, Bengali-script ratios, length constraints, and direct/compact benchmark decontamination.

The recorded corpus report examined 585,619 documents and accepted 567,746. The fixed foundation budget contains 300,004,991 tokens, with a disjoint 3,000,629-token validation allocation. Packed block order and shard provenance are immutable.

The dense foundation uses sixteen decoder layers, width 512, grouped-query attention with eight query heads and two key/value heads, RoPE, QK normalization, RMSNorm, multi-token prediction, and the frozen custom tokenizer. The fixed-budget foundation run is single-seed developmental evidence.

## Dense, Switch, standard-MoE, and Boichitro systems

M0 is the dense control. M1 uses Switch top-1 routing and substantially more inactive expert capacity. M2 uses a shared expert plus top-2 of eight routed experts. M3 starts from the M2 continuation and adds the proposed task/dialect routing mechanisms.

The architectures are designed around approximately 83.8 million active parameters per token. Total parameters differ: about 83.8M for M0, 381.1M for M1, and 168.8M for M2/M3. This is active-parameter matching, not wall-clock or energy matching.

The first four M3 blocks remain dense. Early sparse blocks receive causal dialect-evidence signals derived only from the visible prefix. Middle layers use learned load-balance bias. Late sparse blocks receive task-conditioned bias. Shared experts remain active alongside two routed experts.

## Boichitro routing, source adversary, and GroupDRO

The proposed contribution is not sparse capacity alone. It is the combination of causal dialect-evidence routing, late task-conditioned routing, source-adversarial supervision, and bounded group-robust optimization.

The dialect head encourages representations to preserve regional evidence without using future tokens. The gradient-reversal source head penalizes representations that predict dataset provenance. GroupDRO reweights observed dialect/source/authenticity groups within registered bounds. Router statistics track expert load, entropy, coefficient of variation, dropped-token behavior, and persistent collapse.

These mechanisms require direct validation. Current development task scores do not establish source invariance, and the locked routing-specialization outputs are absent. Therefore this document treats the mechanisms as implemented design features, not demonstrated scientific effects.

## Optimization, continuation learning rate, and upcycling

Eligible hidden matrices use Muon, while embeddings, normalization parameters, routers, and task heads use AdamW. A separately registered AdamW-only control prevents improvements from being attributed solely to the optimizer split.

Restarting a mature foundation checkpoint at the original high learning rate produced a monotonic validation regression. A validation-only pilot selected Muon 0.001 and AdamW 0.000015 for the continuation. The rejected high-rate run remains in the archive as negative evidence.

Dense-to-MoE transfer compared abrupt bank release, unbanked transfer, random initialization, annealed cross-bank release, and permanent complementary-bank routing. Only the permanent paired-bank strategy met the registered transient and endpoint regression guards. A separate Switch pilot selected auxiliary-balance straight-through routing after a loss-free variant exhibited unacceptable load variation.

## Task adaptation, replay retention, and identification

Task adaptation uses three seeds. Stage A consumes a fixed 12M-token mixture of general replay, dialect language modeling, normalization, and romanized material. Stage S consumes 6M tokens and selects normalization checkpoints only when general-language replay degradation remains at or below the preregistered 5% guard.

The selected `ret35_balanced` schedule allocates 30% normalization and 35% replay. It achieved 41.186 validation macro chrF++ in its pilot with 0.972% replay-NLL degradation. A higher-scoring default schedule was rejected because it incurred 15.91% degradation. This is a clear example of a protocol constraint overriding a superficially better task score.

Causal identification trains with early stopping and post-hoc temperature calibration. The separate bidirectional branch applies masked-next-token prediction and contrastive supervision before identification specialization. Two of its three seed manifests are complete at this snapshot.

## Baselines and source-blind development fusion

Normalization baselines include identity copying and a training-only word-rewrite system. The fair rewrite infers a supported dialect from source text. A legacy gold-dialect rewrite is retained strictly as an oracle diagnostic and is not a deployable comparator.

Identification controls include character TF–IDF SVM and SGD systems. These are strong IID baselines but collapse on source-OOD material, motivating source-robust modeling. External model baselines are pinned in configuration but have no completed locked manifests.

A fixed source-blind candidate selector and neural/SVM probability blend were selected on development data. They use source text, candidate outputs, and dialect probabilities inferred from source text; references, gold dialects, source IDs, and evaluation-track labels are forbidden inference features. The fixed fusion transfers without retuning across all M2/M3 development runs, but remains exploratory until locked confirmation.

## Evaluation tracks, metrics, and statistical protocol

Normalization reports chrF++, BLEU, TER, character error rate, exact match, worst-dialect performance, and replay retention. Identification reports accuracy, balanced accuracy, thirteen-class and regional macro-F1, MCC, ECE-15, Brier score, and worst-present-dialect F1.

Tracks separate validation, group-IID test, source-held-out test, external transcript, and romanized challenge material. RAJ normalization is a zero-shot challenge and is not silently averaged into the trained-dialect endpoint.

Confirmatory inference is designed around paired per-example predictions. Semantic groups define the resampling unit. Hierarchical paired bootstrap intervals, semantic-group paired randomization, and Holm correction within registered endpoint families control dependence and multiplicity. Those locked confirmatory outputs have not been generated.

## Current development results and interpretation

All twelve main development manifests are present. Mean normalization macro chrF++ is 41.151 for M0, 42.014 for M1, 41.951 for M2, and 40.175 for M3. M3 trails M2 by -1.776 points in this development summary.

Mean regional identification macro-F1 is 0.7592 for M0, 0.7645 for M1, 0.7594 for M2, and 0.7510 for M3. M3 trails M2 by -0.0084. These are validation-selected outcomes and cannot support a locked-test claim.

The fixed fusion substantially improves development normalization for both M2 and M3, reaching roughly 54–55 macro chrF++ across seeds, and raises identification regional macro-F1 to roughly 0.79–0.80. M2 remains slightly stronger than M3 under this fixed transfer. The appropriate conclusion is mixed-to-negative for the proposed architectural advantage, while the source-blind system-level fusion is a promising exploratory result.

## Systems and inference efficiency

On the NVIDIA GB10 training benchmark, M0 processes about 15,337 tokens/s with 2.83 GiB peak memory. M1 processes about 5,211 tokens/s with 5.69 GiB. M2 and M3 process about 8,272 and 8,224 tokens/s with roughly 3.84 GiB.

M3 adds little overhead relative to M2, but both sparse models are substantially slower than dense M0 despite similar active parameter counts. Grouped expert execution, routing, dispatch, and memory movement therefore matter to practical compute claims.

Task inference benchmarks include batch-one and batched normalization and identification. The results show substantial batching gains and distinct latency/memory trade-offs. Any deployment claim should report task, batch size, generated tokens, examples per second, latency, checkpoint hash, and hardware.

## Software architecture and reproducibility

The project separates reusable library code under `src/boichitro`, command-line tools under `tools`, YAML configuration under `configs`, regression tests under `tests`, immutable evidence under `reports`, metrics and predictions, and run artifacts.

The recorded regression suite reports 107 passed, 0 failed, and one non-failing forward-compatibility warning. The 44 source figures passed 398/398 checks for paired formats, hashes, resolution, captions, and source data.

Every configuration, Python module, run manifest, training report, and CSV evidence table receives a catalog card later in this monograph. The evidence snapshot copies small human-readable artifacts so the documentation remains inspectable after the original workspace is removed.

## Limitations, blockers, and prohibited claims

The study is not submission-ready. No protocol-freeze manifest is present. No locked M0–M3 neural evaluation manifests, locked ablation results, registered robustness curves, locked routing specialization outputs, or confirmatory statistics are present. External baselines are configured but not completed on the locked tracks.

The native dataset review is 0/230 and blinded native system ratings are absent. Therefore the corpus cannot be described as fully linguistically validated or redistribution-ready, and machine metrics cannot establish human adequacy.

The primary M3-versus-M2 hypothesis is currently unsupported by development scores. The document must not imply a positive Boichitro effect, Q1 acceptance, or completed computational evidence. The manuscript requires author metadata, declarations, target-journal selection, full references, and updated results.

## Preservation, deletion consequences, and recovery limits

The documentation folder preserves descriptions and small evidence, not the executable state of the full experiment. Deleting the original workspace and the complete backup will permanently remove raw and processed Parquet data, packed pretraining blocks, `.pt` checkpoints, optimizer state, caches, full predictions, and original archives unless they exist elsewhere.

GitHub normally contains the compact source repository and frozen tokenizer, not the 91 GiB research state. A PDF, DOCX, or ZIP cannot reproduce training without the omitted artifacts.

Before deleting the original or backup, retain at least one independent copy of the complete archive on an external disk or trusted object store. If deletion proceeds anyway, this documentation will preserve scientific context, tables, figures, configurations, code descriptions, hashes recorded by the experiment, run metadata, and a complete former-file inventory—but not recoverable model weights or corpora.

## Reproduction guide

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

## Evidence traceability and documentation conventions

Each appendix card gives the original project-relative path, byte size, SHA-256 for preserved small files, structural keys, and a concise role description. Tables preview machine-readable evidence but do not alter it. Figure plates retain the original title, caption, evidence class, source-data path, and published hash.

The complete file inventory records every former project file, its category, extension, size, modification time, inode, hardlink count, and whether it was copied or merely indexed. Hardlink counts matter because checkpoints can share storage even when their logical sizes are large.

Where generated summaries conflict with older narrative text, the dated status-reconciliation chapter governs. Original reports are included unchanged as historical records and are explicitly labeled by their source timestamp.
