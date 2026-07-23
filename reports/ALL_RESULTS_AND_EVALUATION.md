# Boichitro-MoE: consolidated results and Q1 evaluation

**Snapshot:** 2026-07-22 06:12 KST
**Report status:** interim, evidence-audited snapshot
**Pipeline status at snapshot:** `RUNNING`
**Submission status:** **NOT READY**

This report inventories every finalized result family currently present in the
workspace. Development, previously accessed test, smoke-test, and missing
locked evidence are kept separate. Values from unfinished checkpoints are not
treated as scientific results.

## Executive evaluation

The project is methodologically ambitious and unusually strong in provenance,
leakage control, artifact tracking, and reproducibility. However, its main
scientific claim is not yet testable: two M2 and all three M3 main runs,
protocol freezing, locked evaluation, confirmatory statistics, robustness,
routing analysis, and human evaluation are unfinished.

The raw architecture results remain mixed, but the previously weak system-level
evaluation has now been repaired under a source-blind contract:

- M1 improves over M0 by **+0.863 macro chrF++** on normalization validation,
  but loses **0.661 worst-dialect chrF++**, increases replay degradation by
  **0.818 percentage points**, and has worse selection-time calibration.
- M1 improves M0 identification regional macro-F1 by only **+0.0053**.
- The earlier **50.462** word-rewrite score used gold dialect metadata. It is
  now correctly labeled as an oracle, not a deployable comparator.
- The corrected inferred-supported-dialect rewrite scores **50.005 macro
  chrF++** and **30.388 worst-dialect chrF++** with no gold metadata.
- A semantic-group-held-out, five-fold, reference-free candidate selector
  reaches **52.667 macro chrF++** and **34.851 worst-dialect chrF++** over six
  M0/M1 runs. Its paired gain over the fair rewrite is **+2.662** (95%
  dependence-preserving hierarchical bootstrap CI **[2.086, 3.268]**; paired
  randomization **p=0.000200**).
- The frozen neural/SVM identification blend reaches **0.8009 regional
  macro-F1**, a paired **+0.0390** over raw neural predictions (95% CI
  **[0.0309, 0.0479]**; **p=0.000200**), with **0.0440 ECE-15**.

These fusion values are development-only and selection-conditioned: the same
development rows informed the selected family/threshold or blend weight. Their
intervals preserve shared semantic-group dependence across runs and dialects,
but do not include selection uncertainty and are exploratory, not confirmatory.
Raw neural outputs remain primary for architecture and source-OOD inference,
and the fixed fusion is saved as a separate system view for the later locked
evaluation.

Accordingly, the evidence presently supports the engineering contribution and
some modest dense-to-Switch development gains. It does **not** support
Boichitro-MoE superiority, source-OOD robustness, state-of-the-art performance,
or publication readiness.

**Current Q1 readiness rating: 5.5/10.** The evaluation design and deployable
development systems are materially stronger, but submission now would still
be premature because the central M3-vs-M2 locked claim and required human
evidence do not yet exist.

## Evidence status

| Result family | Finalized status | Admissible interpretation |
|---|---:|---|
| Data engineering and leakage gates | Pass internally | Training-ready; native linguistic validation pending |
| Tokenizer selection | Complete | Validation-only, three seeds, no test access |
| Dense foundation and three continuations | Complete | Descriptive, one foundation seed |
| Development pilots | Complete | Hyperparameter and topology selection only |
| M0 main task runs | 3/3 | Development validation only |
| M1 main task runs | 3/3 | Development validation only |
| Fair source-blind fusion | Complete on M0/M1 | Validation-only; fixed before locked evaluation |
| M2 main task runs | 1/3 finalized | Seed 1701 is complete; insufficient for an aggregate claim |
| M3 main task runs | 0/3 | No result |
| Fixed fusion transfer to M2/M3 | 1/6 | M2 seed 1701 diagnostic only; shared validation rows |
| Locked main neural evaluation | 0/12 | No confirmatory result |
| Confirmatory and optimization ablations | 0/30 | No causal component evidence |
| Human dataset review | 0/230 | No native-quality corpus claim |
| Blinded human system ratings | Missing | No adequacy/fluency claim |

The readiness audit passes **10 of 29** gates. The 19 failed gates are blocking,
not cosmetic.

## Dataset results

### Final admitted data

| Quantity | Rows |
|---|---:|
| Authentic normalization pairs | 54,598 |
| Traceable synthetic normalization pairs, train only | 3,325 |
| All normalization rows | 57,923 |
| Identification rows | 122,353 |
| Real romanized source-OOD challenge | 1,342 |
| Unique tokenizer-training texts | 100,236 |
| Recorded exclusion decisions | 137,386 |
| Quarantined legacy-derived rows | 51,101 |

Normalization contains 39,622 training, 4,289 validation, 5,033 IID-test, and
8,979 OOD rows. The OOD total consists of 7,592 supported-dialect source-OOD
rows plus 1,387 RAJ zero-shot rows. Identification contains 92,897 training,
10,238 validation, 11,264 IID-test, 6,803 source-OOD, and 1,151 external rows.

The engineering gate passes ten automatic checks: the 13-label inventory is
consistent, exact/compact train-evaluation overlap is absent, cross-label
conflicts are removed, synthetic data are train-only and excluded from ID,
locked OOD sources never train, the legacy archive is absent, and licenses are
resolved. The sole failed publication gate is native review: **0 of 230 rows**
have been reviewed.

### Dataset limitations

- Normalization training is concentrated in CHI (36.76%) and SYL (36.54%);
  together they account for 73.3% of training rows.
- RAJ has no normalization training or validation pair, so its 1,387-row score
  must be labelled zero-shot and cannot be merged into supported-dialect OOD.
- KHU, NAR, and RAN lack defensible independent-source normalization tests.
- The dataset is internally verified but cannot yet be described as
  native-speaker validated or cleared for public redistribution.

## Tokenizer results

Twelve intrinsic candidates passed the mechanical gates; four entered the
three-seed, 2M-token-per-seed proxy comparison.

| Candidate | Proxy BPC mean ± SD ↓ | Worst-dialect BPC ↓ | Tokens/character ↓ | Tokens/s ↑ | Selected |
|---|---:|---:|---:|---:|---:|
| byte-BPE balanced 16k | 3.3807 ± 0.0113 | 3.5934 | 0.6859 | 153,304 | No |
| Unigram balanced 32k | 2.7482 ± 0.0114 | 2.9518 | 0.2383 | 72,811 | No |
| WordPiece balanced 32k | 2.6675 ± 0.0023 | 2.8898 | 0.2268 | 71,746 | No |
| **WordPiece natural 32k** | **2.6600 ± 0.0088** | **2.8753** | **0.2245** | **74,345** | **Yes** |

The frozen choice is well supported within the screened set: it has the best
mean and worst-dialect proxy BPC and the lowest token cost among the shortlisted
non-byte tokenizers. Selection used no test data. The limitation is breadth:
the proxy study establishes the best local candidate, not superiority to every
modern Bangla tokenizer.

## Pretraining corpus and foundation results

The revision-pinned FineWeb-2 Bangla build saw 585,619 documents, accepted
567,746, and produced 300,004,991 training tokens plus 3,000,629 validation
tokens. Recorded rejections include 3,505 low-Bangla-ratio documents, 2,476
direct benchmark matches, 49 compact benchmark matches, 130 overlong documents,
and 69 invalid-Unicode documents. All five source parquet hashes were verified.

### Fixed-budget foundation comparison

| Run | Tokens | Final validation BPC ↓ | Change from 300M dense foundation | Training tok/s |
|---|---:|---:|---:|---:|
| F dense foundation | 300.0M | 1.230329 | — | 23,853 |
| M0 dense continuation | 200.0M | 1.203249 | −2.201% | 16,877 |
| M1 Switch continuation | 200.0M | **1.199294** | **−2.523%** | 14,100 |
| M2 shared top-2 MoE continuation | 200.0M | 1.202603 | −2.254% | 18,223 |

M1 is best by validation BPC, but its advantage over M0 is only 0.00396 BPC
(0.329%). M2 improves on M0 by 0.00065 BPC (0.054%). These are single-seed
foundation results, so they are descriptive and cannot establish a stable
architecture ranking.

Router diagnostics passed the non-finite and persistent-collapse guards. M1's
mean/final load CV is 0.223/0.258; M2's is 0.147/0.043. One isolated M1 CV
exceedance and no post-stability M2 exceedance were recorded.

## Development-pilot results

### Mature-checkpoint continuation learning rate

| Muon / AdamW LR | Final BPC ↓ | Relative change | Eligible | Decision |
|---|---:|---:|---:|---|
| 0.001 / 0.000015 | **1.226813** | −0.286% | Yes | Selected |
| 0.002 / 0.000030 | 1.232059 | +0.141% | No | Rejected |
| 0.005 / 0.000075 | 1.254107 | +1.933% | No | Rejected |
| 0.010 / 0.000150 | 1.293376 | +5.124% | No | Rejected |

The monotonic deterioration at larger restart rates is credible evidence that a
mature checkpoint required a low continuation LR.

### Dense-to-MoE recovery

| Strategy | Initial BPC | Final BPC ↓ | Maximum regression | Eligible |
|---|---:|---:|---:|---:|
| Abrupt bank release | 1.230335 | 1.262179 | +3.678% | No |
| Unbanked transfer | 1.397035 | 1.255871 | 0.000% | Negative control |
| Random initialization | 3.519622 | 2.559432 | 0.000% | Negative control |
| Annealed cross-bank | 1.230335 | 1.264109 | +2.745% | No |
| **Permanent paired bank** | 1.230335 | **1.228724** | **0.000%** | **Selected** |

Only permanent paired-bank routing satisfied the preregistered transient and
final regression guards. This is one of the clearest positive pilot findings.

### Switch-router repair

| Router strategy | Final BPC ↓ | Maximum load CV ↓ | Final load CV ↓ | Eligible |
|---|---:|---:|---:|---:|
| Loss-free straight-through | 1.227010 | 0.5061 | 0.5061 | No |
| **Auxiliary-balance straight-through** | **1.226940** | **0.2617** | **0.0675** | **Yes** |

The auxiliary-balance router is both marginally better in BPC and substantially
more balanced, justifying its selection.

### Stage-S retention schedule

| Candidate | Norm/replay fraction | Macro chrF++ ↑ | Worst dialect ↑ | Replay degradation ↓ | Selected |
|---|---:|---:|---:|---:|---:|
| Rejected default | 0.55 / 0.10 | **44.714** | **34.214** | 15.910% | No |
| ret25 balanced | 0.40 / 0.25 | 41.158 | 31.917 | 1.336% | No |
| ret25 conservative | 0.40 / 0.25 | 41.072 | 30.621 | 0.831% | No |
| **ret35 balanced** | **0.30 / 0.35** | **41.186** | **31.910** | **0.972%** | **Yes** |
| ret35 conservative | 0.30 / 0.35 | 40.745 | 31.285 | 0.362% | No |

The selected schedule correctly respects the 5% replay guard, but the trade-off
is large: it gives up 3.53 chrF++ relative to the rejected high-forgetting
schedule. The choice is protocol-consistent, not evidence that the selected
score is competitive.

## Main task development results

M0 and M1 have finalized three-seed development runs; M2 seed 1701 is also
finalized while its other seeds and all M3 runs remain in progress. All values
below are validation-selection results and must not be reported as locked test
results.

### Per-seed results

| Model | Seed | Norm macro chrF++ ↑ | Worst dialect ↑ | Replay degradation ↓ | ID regional F1 ↑ | ID macro-F1 ↑ | Accuracy ↑ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 1701 | 41.186 | 31.910 | 0.972% | 0.7605 | 0.7726 | 0.8436 | 0.0900 |
| M0 | 2903 | 41.232 | 32.184 | 1.128% | 0.7519 | 0.7648 | 0.8426 | 0.0882 |
| M0 | 4307 | 41.035 | 32.733 | 1.144% | 0.7652 | 0.7769 | 0.8497 | 0.1131 |
| M1 | 1701 | 42.278 | 31.534 | 1.898% | 0.7670 | 0.7788 | 0.8533 | 0.1175 |
| M1 | 2903 | 41.802 | 31.504 | 1.705% | 0.7678 | 0.7796 | 0.8498 | 0.1197 |
| M1 | 4307 | 41.963 | 31.806 | 2.094% | 0.7586 | 0.7706 | 0.8456 | 0.1062 |
| M2 | 1701 | 41.728 | 31.363 | 1.168% | 0.7533 | 0.7657 | 0.8404 | 0.0965 |

### Three-seed aggregates and paired M1−M0 effects

| Metric | M0 mean ± SD | M1 mean ± SD | Paired difference |
|---|---:|---:|---:|
| Normalization macro chrF++ ↑ | 41.151 ± 0.103 | **42.014 ± 0.242** | **+0.863 ± 0.267** |
| Worst-dialect chrF++ ↑ | **32.276 ± 0.419** | 31.615 ± 0.167 | **−0.661 ± 0.276** |
| Replay degradation ↓ | **1.081 ± 0.095%** | 1.899 ± 0.195% | **+0.818 ± 0.209 pp** |
| ID accuracy ↑ | 0.8453 ± 0.0038 | **0.8495 ± 0.0039** | +0.0042 ± 0.0073 |
| ID balanced accuracy ↑ | 0.7582 ± 0.0010 | **0.7684 ± 0.0031** | +0.0102 ± 0.0038 |
| ID macro-F1 (13) ↑ | 0.7714 ± 0.0062 | **0.7763 ± 0.0050** | +0.0049 ± 0.0107 |
| ID regional macro-F1 ↑ | 0.7592 ± 0.0067 | **0.7645 ± 0.0051** | +0.0053 ± 0.0113 |
| ID ECE ↓ | **0.0971 ± 0.0138** | 0.1145 ± 0.0073 | +0.0174 ± 0.0211 |
| ID Brier ↓ | **0.2496 ± 0.0062** | 0.2589 ± 0.0055 | +0.0093 ± 0.0117 |

With only three seeds and development reuse, these differences are descriptive.
The paired signs show a real trade-off: M1 improves mean task scores but is
worse on the normalization tail, retention, ECE, and Brier score.

### Normalization validation by dialect

| Dialect | Word rewrite | M0 mean ± SD | M1 mean ± SD | M1−rewrite |
|---|---:|---:|---:|---:|
| BAR | 50.978 | 42.634 ± 1.424 | 42.841 ± 0.803 | −8.137 |
| CHI | 62.192 | 50.019 ± 1.033 | 50.077 ± 0.358 | −12.115 |
| KHU | 34.604 | 34.057 ± 1.183 | **36.118 ± 1.102** | +1.513 |
| MYM | 77.089 | 52.674 ± 1.181 | 53.783 ± 0.751 | −23.306 |
| NAR | 41.184 | 35.703 ± 1.033 | 38.852 ± 2.032 | −2.332 |
| NOA | 51.727 | 38.058 ± 0.742 | 38.772 ± 0.532 | −12.955 |
| RAN | 30.575 | **32.311 ± 0.478** | 31.615 ± 0.167 | +1.039 |
| SYL | 55.349 | 43.753 ± 0.107 | 44.056 ± 0.292 | −11.293 |

M1 beats the word-rewrite control only on KHU and RAN. The largest deficit is
MYM (−23.31 chrF++). This is the most serious current performance finding and
must not be hidden by an architecture-only comparison.

### Identification validation by class

| Label | Support | M0 F1 mean ± SD | M1 F1 mean ± SD | M1−M0 |
|---|---:|---:|---:|---:|
| BAR | 413 | 0.8365 ± 0.0150 | 0.8451 ± 0.0098 | +0.0086 |
| CHI | 1,991 | 0.9446 ± 0.0005 | 0.9481 ± 0.0031 | +0.0035 |
| KHU | 86 | 0.4613 ± 0.0779 | 0.4613 ± 0.0442 | 0.0000 |
| KIS | 797 | 0.7559 ± 0.0059 | 0.7632 ± 0.0085 | +0.0073 |
| MYM | 302 | **0.7826 ± 0.0049** | 0.7559 ± 0.0080 | **−0.0267** |
| NAR | 865 | 0.7803 ± 0.0135 | 0.7842 ± 0.0110 | +0.0039 |
| NOA | 260 | 0.7836 ± 0.0109 | 0.7853 ± 0.0131 | +0.0017 |
| NSD | 576 | 0.6791 ± 0.0189 | 0.7015 ± 0.0172 | +0.0224 |
| RAJ | 91 | 0.7634 ± 0.0470 | 0.7696 ± 0.0159 | +0.0062 |
| RAN | 680 | 0.7670 ± 0.0070 | 0.7825 ± 0.0107 | +0.0155 |
| STD | 2,251 | 0.9185 ± 0.0006 | 0.9186 ± 0.0042 | +0.0001 |
| SYL | 1,407 | 0.8997 ± 0.0030 | 0.9058 ± 0.0045 | +0.0061 |
| TAN | 519 | 0.6561 ± 0.0070 | 0.6713 ± 0.0191 | +0.0152 |

KHU remains the weakest class at 0.461 F1, consistent with its very small
training and validation support. M1 improves most classes but materially harms
MYM.

## Source-blind evaluation repair (development only)

| System | Primary metric | Worst-group metric | Paired gain | 95% CI |
|---|---:|---:|---:|---:|
| Fair supported-dialect rewrite | 50.005 macro chrF++ | 30.388 chrF++ | reference | — |
| Normalization selector V2, six-run semantic-group OOF mean | **52.667 macro chrF++** | **34.851 chrF++** | **+2.662** | **[2.086, 3.268]** |
| Raw neural ID, six-run mean | 0.7618 regional macro-F1 | — | reference | — |
| Neural/SVM ID blend | **0.8009 regional macro-F1** | 0.4508 F1 | **+0.0390** | **[0.0309, 0.0479]** |

The normalization selector uses only source text, its neural and rewrite
candidates, and dialect probabilities inferred from source text. References,
gold dialects, source IDs, and evaluation-track labels are forbidden inference
features. All rows and six repeated outputs sharing a semantic group remain in
the same fold. Resampling assigns one multiplicity or swap to each global
semantic group and reuses it across runs and cross-dialect realizations. Both
selection-conditioned paired randomization diagnostics give two-sided
p=0.000200 with 5,000 draws. These p-values are exploratory because fusion
settings reused development data. The selected artifacts and their SHA-256
hashes are registered for protocol freeze; neither pilot accessed a test split.
An aggregate-only cross-task firewall additionally found zero exact, compact,
or registered SimHash/Jaccard near-duplicate overlap between all normalization
evaluation inputs and the 92,897 inputs used to train the dialect classifier;
it emitted no protected text and was not used for model selection.

The fixed artifacts have also transferred without retuning to the first later
architecture run (1/6 planned M2/M3 runs). On M2 seed 1701, normalization rises
from **41.728 raw** to **54.858 fused** macro chrF++ (worst dialect **36.378**),
and identification regional macro-F1 rises from **0.7533 raw** to **0.8046
fused** with **0.0489 ECE-15**. This is a useful architecture-transfer
diagnostic, but it reuses the same validation rows and one seed cannot establish
stability or independent confirmation.

## Fixed classical baselines

These baselines were evaluated before the neural protocol freeze. They are
useful descriptive floors, but the test sets are not pristine. This limitation
is permanently disclosed and cannot be repaired by deleting outputs.

### Normalization

| Model / track | Rows | Macro chrF++ ↑ | Worst dialect ↑ | Macro BLEU ↑ | TER ↓ | CER ↓ | Exact match ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Copy / validation | 4,289 | 38.401 | 23.764 | 10.021 | 63.106 | 0.348 | 0.0415 |
| Inferred supported-dialect rewrite / validation | 4,289 | **50.005** | 30.388 | — | — | — | — |
| Gold-dialect rewrite oracle / validation | 4,289 | 50.462 | 30.575 | **23.774** | **45.079** | **0.258** | **0.0874** |
| Copy / IID test | 5,033 | 39.260 | 25.853 | 10.750 | 62.102 | 0.334 | 0.0296 |
| Gold-dialect rewrite oracle / IID test | 5,033 | **50.916** | **33.423** | **22.912** | **44.316** | **0.250** | **0.0757** |
| Copy / source OOD | 7,592 | 38.652 | 25.319 | 8.602 | 70.984 | 0.409 | 0.1616 |
| Gold-dialect rewrite oracle / source OOD | 7,592 | **44.607** | **31.767** | **15.191** | **63.849** | **0.378** | **0.1927** |
| Copy / RAJ zero-shot | 1,387 | 48.642 | 48.642 | 4.912 | 59.002 | 0.357 | 0.2884 |
| Gold-dialect rewrite oracle / RAJ zero-shot | 1,387 | 48.642 | 48.642 | 4.912 | 59.002 | 0.357 | 0.2884 |

The legacy rewrite equals copy on RAJ because RAJ has no learned rewrite
inventory. All legacy test rewrite values above used gold dialect and are
diagnostic oracle values. The fair inferred-dialect baseline is evaluated on
locked tests only after protocol freeze.

### Identification

| Model / track | Accuracy ↑ | Macro-F1 (13) ↑ | Regional macro-F1 ↑ | ECE ↓ | Worst present F1 ↑ |
|---|---:|---:|---:|---:|---:|
| SVM / validation | 0.8550 | 0.7838 | 0.7734 | 0.0113 | 0.4138 |
| SVM / IID test | 0.8540 | 0.7783 | 0.7669 | 0.0111 | 0.4118 |
| SVM / source OOD | 0.4724 | 0.2229 | 0.3738 | 0.1891 | 0.0343 |
| SVM / external transcript | 0.9705 | 0.5943 | 0.9658 | 0.0146 | 0.8739 |
| SGD, 5-seed mean / validation | 0.8510 | 0.7768 | 0.7664 | 0.0566 | 0.3850 |
| SGD, 5-seed mean / IID test | 0.8498 | 0.7759 | 0.7653 | 0.0534 | 0.4377 |
| SGD, 5-seed mean / source OOD | 0.4686 | 0.2168 | 0.3630 | 0.1624 | 0.0257 |
| SGD, 5-seed mean / external | 0.9672 | 0.5917 | 0.9614 | 0.0415 | 0.8483 |

The SVM's source-OOD collapse (regional macro-F1 0.374 versus 0.773 on
validation) strongly motivates source-robust modeling. Conversely, its high
IID score and excellent calibration mean the neural models must beat a strong,
cheap baseline—not merely the majority classifier.

The external macro-F1 (13) is low because absent labels are included in that
13-way scalar, while regional macro-F1 is computed over labels present in the
external track. Both must be reported with the coverage definition.

## Systems results on NVIDIA GB10

| Model | Total params | Active params/token | Throughput tok/s ↑ | Peak memory GiB ↓ | Slowdown vs M0 |
|---|---:|---:|---:|---:|---:|
| M0 dense | 83.79M | 83.79M | **15,337** | **2.83** | 1.00× |
| M1 Switch | 381.11M | 83.84M | 5,211 | 5.69 | 2.94× |
| M2 standard MoE | 168.77M | 83.84M | 8,272 | 3.84 | 1.85× |
| M3 Boichitro | 168.79M | 83.86M | 8,224 | 3.84 | 1.86× |

Active parameters are tightly matched, but wall-clock compute is not. M2/M3
run at only about 54% of dense throughput, and M1 at 34%. M3 adds just 0.58%
throughput overhead and 0.0068 GiB over M2, which is favorable for the proposed
mechanisms. Nevertheless, any “compute matched” wording must distinguish active
parameter matching from measured time/energy matching.

## Smoke-test-only output

A tiny M3 end-to-end smoke run evaluated deterministic prefixes of 8
normalization and 16 identification examples per track. It scores 2.111 IID
chrF++, 0 source-OOD/romanized chrF++, and zero ID accuracy. Its manifest
explicitly states `smoke_only_not_scientific_evidence`; these values demonstrate
pipeline execution only and must never appear as model-quality results.

## Software, tables, and figures

- Regression suite: **107 passed**, with one non-failing PyTorch GPU-capability
  warning.
- Journal figure suite: **44 PNG/PDF pairs**, exceeding the requested minimum.
- Figure validation: **398/398 checks passed** for format, hashes, resolution,
  captions, and source data.
- Evidence classes: **40 development-only** and **4 prior-test descriptive**;
  none is mislabeled as locked evidence.
- Paper tables: **10 reproducible table families**.
- Manuscript: 3,154 words, structurally sound but still a short journal-neutral
  draft with withheld results, abbreviated references, and required metadata.

The figure engineering is strong, but 44 technically valid figures do not
replace locked evidence. A final paper should curate a small main set and move
most diagnostics to supplementary material.

## Claim-by-claim evaluation

| Proposed claim | Current verdict | Reason |
|---|---|---|
| Provenance-first, leakage-controlled benchmark | **Provisionally supported** | Automatic gates pass; native review and redistribution clearance remain |
| Frozen tokenizer chosen without test access | **Supported** | Three-seed proxy and immutable hashes exist |
| Stable dense-to-MoE upcycling | **Supported developmentally** | Paired-bank pilot passes guards; one pilot seed |
| M1 improves over dense M0 | **Mixed** | Mean metrics improve slightly; tail performance, replay, and calibration worsen |
| M3 outperforms M2 on source-OOD normalization | **Untested** | One M2 development seed exists; no M3 run or locked manifest exists |
| Fixed source-blind system is competitive on validation | **Supported developmentally** | Selector reaches 52.667 chrF++; ID blend reaches 0.8009 F1 with paired positive exploratory intervals |
| Source-invariant routing works | **Untested** | Locked routing specialization analysis missing |
| Robust under perturbation/romanization | **Untested** | Registered robustness evaluation missing |
| Positive development fusion diagnostics | **Supported, exploratory** | Dependence-preserving intervals exclude zero, but selection reused these rows and locked confirmation is missing |
| Human-acceptable normalization | **Untested** | No blinded native ratings |

## Q1 scorecard

| Dimension | Score | Assessment |
|---|---:|---|
| Novelty and research framing | 8/10 | Important low-resource/source-confounding question |
| Data and protocol engineering | 8/10 | Strong controls; native and redistribution gates remain |
| Reproducibility and artifact discipline | 9/10 | Hashes, registries, manifests, tests, source data |
| Systems evidence | 7/10 | Good active-parameter accounting; no energy/time matching |
| Current model competitiveness | 6/10 | Raw neural normalization trails, but fair fixed fusion now exceeds the source-blind rewrite validation baseline |
| Experimental completeness | 3/10 | Seven of twelve main development runs are finalized; downstream locked/ablation studies remain absent |
| Statistical evidence | 4/10 | Paired development fusion evidence exists; locked architectural inference is absent |
| Human linguistic evidence | 0/10 | Dataset review and system ratings absent |
| Manuscript/submission package | 4/10 | Draft exists; core results, metadata, target adaptation absent |

**Overall blocking-gate rating: 5.5/10.** This is not an average of cosmetic
assets; it reflects that the primary hypothesis has no result.

## Mandatory work before submission

1. Finish all 12 M0–M3 main runs and the registered optimizer,
   bidirectional-ID, external-baseline, factorial, confirmatory, and optimization
   studies.
2. Freeze the protocol before any main neural test access, retaining the prior
   classical/smoke disclosure.
3. Run all 12 locked main evaluations, at least 30 locked ablation evaluations,
   external baselines, robustness, routing, and registered statistics.
4. Require the primary M3−M2 source-OOD macro-chrF++ 95% hierarchical paired
   interval to exclude zero and its registered randomization test to survive
   familywise correction.
5. Demonstrate that M3 and the fixed M3 fused view are competitive with the
   fair inferred-dialect rewrite and credible external baselines—not only M2—
   while retaining the gold-dialect rewrite solely as an oracle; inspect
   MYM/NOA/CHI error modes.
6. Report IID, source-OOD, RAJ zero-shot, romanized, worst-dialect, calibration,
   throughput, memory, and retention trade-offs together.
7. Complete all 230 native dataset-review rows and blinded multi-rater system
   evaluation, including adequacy, fluency, hallucination, agreement,
   qualifications, consent, compensation, and ethics/IRB determination.
8. Select an actual target journal, verify its current quartile and policies,
   complete author metadata/declarations, expand the manuscript, format full
   references, and curate figures.

## Final decision

**Do not submit now.** The project has a credible Q1-level research design, but
the available evidence is an engineering/development package, not a completed
paper. Q1 competitiveness becomes plausible only if the locked M3 result is
positive, statistically robust, competitive with strong baselines, and
confirmed by native-speaker evaluation. If M3 remains near the present M0/M1
range, the normalization system will need substantive modeling or decoding
improvement before journal submission.

## Exact evidence sources

- Dataset: `reports/FINAL_DATASET_REPORT.md` and
  `tables/paper/dataset_counts.csv`
- Tokenizer: `reports/tokenizer/TOKENIZER_FREEZE_REPORT.json`,
  `reports/tokenizer/tokenizer_intrinsic_screen.csv`, and
  `reports/tokenizer/tokenizer_proxy_summary.csv`
- Foundation/pilots: `runs/*/1701/training_report.json` and
  `tables/paper/*.csv`
- Main validation: `runs/task/boichitro_q1_v1/*/*/stage_*/best_selection.json`
- Consolidated main rows: `reports/model/main_validation_results_current.csv`
  with run-count/hash audit in
  `reports/model/development_results_snapshot.json`
- Dialect/class summaries:
  `reports/model/main_validation_by_dialect_all_current.csv` and
  `reports/model/main_identification_by_class_all_current.csv`
- Classical baselines: `reports/model/classical_*_results.csv` and `metrics/`
- Source-blind repair: `reports/model/source_blind_normalization_baseline_audit.json`,
  `reports/model/cross_task_input_firewall.json`,
  `reports/model/normalization_fusion_selection_v2.json`,
  `reports/model/id_fusion_selection.json`, and
  `reports/model/development_fusion_uncertainty.json`
- Fixed no-retuning transfer:
  `reports/model/development_fusion_architecture_transfer.json`
- Systems: `reports/model/gb10_model_benchmark.json`
- Disclosure: `reports/PRIOR_TEST_ACCESS_DISCLOSURE.md`
- Readiness: `reports/Q1_JOURNAL_READINESS_AUDIT.md`
- Figure validation: `figures/q1/VALIDATION_REPORT.md`
