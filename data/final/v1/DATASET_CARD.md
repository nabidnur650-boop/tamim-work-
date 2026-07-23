# Boichitro Data v1.0.0

## Scope

This is the frozen data release for the Bangla dialect SLM/MoE experiments.
The label inventory has **13 classes: 12 regional varieties plus Standard
Bangla (`STD`)**. It must not be described as 13 regional dialects.

The primary supervised task is dialect-to-Standard-Bangla normalization. The
secondary task is 13-label dialect identification. Bengali-script parallel,
romanized robustness, source-held-out OOD, synthetic robustness, and tokenizer
data are stored separately so a result cannot silently mix these conditions.

## Release state

- Data-engineering gate: `PASS_INTERNAL_DATA_ENGINEERING`
- Public/paper release gate: `CONDITIONAL_NATIVE_REVIEW_REQUIRED`
- Authentic normalization rows: 54,598
- Train-only synthetic robustness rows: 3,325
- Identification rows: 122,353
- Romanized source-held-out rows: 1,342
- Unique tokenizer-training texts: 100,236

The algorithmic build is frozen and reproducible. Linguistic publication
claims remain conditional on completing `reports/HUMAN_NATIVE_REVIEW_SAMPLE.csv`.

## Text normalization levels

1. `raw`: original source cell, unchanged.
2. `nfc`: Unicode NFC with BOM/zero-width storage artifacts removed.
3. `clean`: control-character and whitespace cleanup.
4. `model`: conservative quote, dash, punctuation-spacing, and transcript-marker cleanup.
5. `compact`: NFKC letters/numbers only, used **only** for leakage and duplicate detection.

The compact form is never used as the model input. Punctuation and dialectal
spelling are not broadly standardized away.

## Normalization composition

| dialect | split | is_synthetic | rows |
|---|---|---|---|
| BAR | test | 0 | 491 |
| BAR | test_ood | 0 | 608 |
| BAR | train | 0 | 2,727 |
| BAR | train | 1 | 408 |
| BAR | validation | 0 | 333 |
| CHI | test | 0 | 1,879 |
| CHI | test_ood | 0 | 2,096 |
| CHI | train | 0 | 13,655 |
| CHI | train | 1 | 909 |
| CHI | validation | 0 | 1,706 |
| KHU | test | 0 | 101 |
| KHU | train | 0 | 711 |
| KHU | train | 1 | 106 |
| KHU | validation | 0 | 89 |
| MYM | test | 0 | 375 |
| MYM | test_ood | 0 | 1,384 |
| MYM | train | 0 | 1,868 |
| MYM | train | 1 | 280 |
| MYM | validation | 0 | 250 |
| NAR | test | 0 | 101 |
| NAR | train | 0 | 928 |
| NAR | train | 1 | 139 |
| NAR | validation | 0 | 101 |
| NOA | test | 0 | 373 |
| NOA | test_ood | 0 | 1,400 |
| NOA | train | 0 | 1,869 |
| NOA | train | 1 | 280 |
| NOA | validation | 0 | 249 |
| RAJ | test_ood | 0 | 1,387 |
| RAN | test | 0 | 127 |
| RAN | train | 0 | 1,100 |
| RAN | train | 1 | 164 |
| RAN | validation | 0 | 159 |
| SYL | test | 0 | 1,586 |
| SYL | test_ood | 0 | 2,104 |
| SYL | train | 0 | 13,439 |
| SYL | train | 1 | 1,039 |
| SYL | validation | 0 | 1,402 |

## Identification composition

| dialect | split | rows |
|---|---|---|
| BAR | test | 587 |
| BAR | test_external | 65 |
| BAR | test_ood | 495 |
| BAR | train | 4,018 |
| BAR | validation | 413 |
| CHI | test | 2,205 |
| CHI | test_external | 162 |
| CHI | test_ood | 1,654 |
| CHI | train | 16,886 |
| CHI | validation | 1,991 |
| KHU | test | 95 |
| KHU | train | 668 |
| KHU | validation | 86 |
| KIS | test | 793 |
| KIS | test_external | 159 |
| KIS | train | 7,953 |
| KIS | validation | 797 |
| MYM | test | 448 |
| MYM | test_ood | 701 |
| MYM | train | 2,171 |
| MYM | validation | 302 |
| NAR | test | 850 |
| NAR | test_external | 162 |
| NAR | train | 8,009 |
| NAR | validation | 865 |
| NOA | test | 387 |
| NOA | test_ood | 786 |
| NOA | train | 1,783 |
| NOA | validation | 260 |
| NSD | test | 538 |
| NSD | test_external | 108 |
| NSD | train | 5,300 |
| NSD | validation | 576 |
| RAJ | test | 88 |
| RAJ | test_ood | 717 |
| RAJ | train | 705 |
| RAJ | validation | 91 |
| RAN | test | 698 |
| RAN | test_external | 108 |
| RAN | train | 6,481 |
| RAN | validation | 680 |
| STD | test | 2,452 |
| STD | test_ood | 1,172 |
| STD | train | 18,768 |
| STD | validation | 2,251 |
| SYL | test | 1,617 |
| SYL | test_external | 283 |
| SYL | test_ood | 1,278 |
| SYL | train | 15,134 |
| SYL | validation | 1,407 |
| TAN | test | 506 |
| TAN | test_external | 104 |
| TAN | train | 5,021 |
| TAN | validation | 519 |

## External-source roles

- Kothon v4: authentic CHI/SYL train-development candidate after decontamination.
- Sylheti translation v3: only novel rows; lexical items are train-only.
- ChattogramSent v2: only the novel remainder; not claimed as independent OOD.
- ONUBAD v2: locked source-OOD Bengali-script sentence benchmark.
- BhasaBodh v1: romanized companion to accepted ONUBAD rows, not double-counted.
- BD-Dialect v2: fully locked phrase/lexical OOD challenge, including RAJ.
- RAJ has no admitted normalization train/validation pairs. Its 1,387
  normalization rows form the separate `zero_shot_raj` track and are excluded
  from the supported-dialect `source_ood` macro average.
- Hugging Face regional ASR: exact-name mapped transcript identification data.
- BanglaDial v2: lower-confidence merged-provenance identification pool.
- Chittagonian vulgar lexicon: tokenizer/safety auxiliary only.

Habiganj and Sandwip are not collapsed into Sylhet or Chittagong. They remain
audited exclusions because geographic proximity is not a defensible label map.

## Synthetic policy

Synthetic examples are deterministic perturbations of accepted authentic
**training parents only**, capped at 10% overall and 15% per dialect. They keep
parent IDs, receive 0.5 example loss weight, make no authenticity claim, and
are ineligible for dialect identification, validation, or test sets. The old
51,101-row derived ZIP remains quarantined and contributes zero final rows.

## Licensing and attribution

Mendeley inputs used here are pinned CC BY 4.0 releases; Hugging Face inputs are
pinned Apache-2.0 releases. See `data/manifests/licenses.yaml` for per-source
DOIs, commits, caveats, and release obligations. Modifications and derived
augmentation must be disclosed in any redistributed version.

## Known limitations

- BanglaDial is a merged corpus whose original component provenance is incomplete.
- The Hugging Face ASR card self-reports Apache-2.0; upstream competition terms
  should be rechecked before redistributing its text extract.
- Written dialect labels are regional proxies, not speaker-identity ground truth.
- Native-speaker review is deliberately not fabricated by this automated build.
- Source-held-out coverage is strongest for BAR/CHI/SYL and the five BD-Dialect varieties.
