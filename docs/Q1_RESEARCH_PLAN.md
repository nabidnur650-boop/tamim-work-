# Bangla Dialect SLM-MoE: Audit and Q1-Grade Research Plan

Date: 2026-07-19
Working model name: Boichitro-MoE
Scope of this document: initial audit and research plan; the original ZIP and notebook files remain unchanged.

Companion execution artifacts created after the member-level archive audit:

- BANGLA_DIALECT_MOE_EXPERIMENT_BLUEPRINT.md — frozen task, architecture, training, ablation, statistics, and figure protocol.
- bangla_dialect_moe/reports/LOCAL_ARCHIVE_AUDIT.md — reproducible archive findings.
- bangla_dialect_moe/configs/experiment_registry.yaml — machine-readable run registry.
- bangla_dialect_moe/tools/audit_local_archives.py — archive auditor.
- bangla_dialect_moe/tools/build_preliminary_manifest.py — protected-split, exact-component manifest builder.
- bangla_dialect_moe/reports/PRELIMINARY_DATA_GATE.md — current executable gate status.

The companion blueprint supersedes this document where the deeper audit established eight locally supported normalization dialects and exact active-compute dimensions.

## 1. Executive decision

The existing work is a useful engineering baseline, but its reported external results are not publication-valid. Do not start the final MoE training from the present merged files.

The correct research program is:

1. Rebuild a leakage-free, provenance-complete corpus from the two archives.
2. Repair and re-evaluate the custom tokenizer under controlled tokenizer baselines.
3. Pretrain a data-adequate dense Bangla foundation using the repaired tokenizer.
4. Upcycle the dense model into a sparse MoE.
5. Test a focused architectural contribution: dialect-guided but source-invariant routing.
6. Evaluate in-domain, source-out-of-domain, robustness, translation, generation, routing, and efficiency.
7. Report matched-compute dense and MoE comparisons with multiple seeds and uncertainty.

The central paper should not claim novelty merely from combining RMSNorm, GQA, SwiGLU, Muon, MTP, and MoE. Those are established components. The publishable novelty should be the treatment of dialect learning under severe dialect–dataset-source confounding:

> Can a small sparse model learn linguistically meaningful Bangla-dialect experts rather than dataset/template experts, while improving worst-dialect and source-OOD performance at matched active compute?

The proposed answer is a shared/fine-grained MoE with a clean-train lexicon routing prior, a dialect-supervised routing objective, and a source-adversarial routing objective. Its value must be established through controlled ablations.

## 2. Materials audited

### 2.1 Local files

| File | Size | Role |
|---|---:|---|
| archive (1).zip | 2.2 MB | Three synthetic/balancing CSV files |
| archive(1).zip | 5.5 MB | BanglaDial, Vashantor, Sylheti, regional corpus, and ChatgaiyyaAlap |
| tokenizer.ipynb | 445 KB | Data registry, cleaning, token mining, deterministic WordPiece, freeze |
| bleach-dense-model.ipynb | 599 KB | 31M dense decoder, CLM training, classification, lexical baselines |

The two ZIPs are distinct. The user message repeated one path, but the directory contains both names.

### 2.2 Dataset inventory

The synthetic archive contains:

| File | Rows | Main labels |
|---|---:|---|
| bangla_dialect_balancing_synthetic_v0.csv | 23,101 | BAR, KHU, KIS, NSD, RAJ, RAN, STD, SYL, TAN |
| khulna_rajshahi_7000_each (1).csv | 14,000 | KHU, RAJ |
| mymensingh_noakhali_7000_each (1).csv | 14,000 | MYM, NOA |

The source archive contains:

- BanglaDial: 63,303 rows, 12 labels.
- Vashantor: five regional variants, each with official train/validation/test splits and parallel standard Bangla, Banglish, regional Bangla, regional Banglish, and English fields.
- Sylheti parallel data: 1,200 pairs.
- BanglaRegionalTextCorpus: 4,653 rows over Rangpur, Narail, Barishal, and Khulna.
- Regional_cleaned_dataset: the same 4,653 regional texts in cleaned form.
- ChatgaiyyaAlap: 4,011 Chittagonian–standard pairs plus a 1,546-row dictionary.
- JSON copies of Vashantor that duplicate its CSV representation.

## 3. What the existing work already accomplished

### 3.1 Existing tokenizer pipeline

The notebook creates a 13-label taxonomy:

BAR, CHI, KHU, KIS, MYM, NAR, NOA, NSD, RAJ, RAN, STD, SYL, TAN.

Its main stages are:

1. Registry and provenance schema.
2. Unicode/text cleaning and quality scoring.
3. Exact and loose duplicate handling.
4. Dialect-specific lexical token mining.
5. Deterministic WordPiece-like vocabulary construction.
6. Frozen tokenizer, token-ID maps, and routing metadata.

Saved notebook results:

- Raw registered rows: 141,171.
- Exact-clean train rows after Portion 2: 123,833.
- Recommended rows after loose synthetic thinning: 121,503.
- Unique lexical tokens: 48,987.
- Selected vocabulary: 32,000.
- Strict protected symbols: 3,696, including 3,679 lexical items.
- Validation UNK rate: 0.
- Mean fertility: 1.1306.
- Maximum dialect mean fertility: 1.3447.
- Dialect tags and strict protected tokens are atomic.

This is worth preserving as a tokenizer baseline and as reusable audit code.

### 3.2 Existing dense model

BLEACH-Dense-32M is a valid small decoder-only architecture:

- 12 layers.
- Hidden size 384.
- 6 query heads and 2 KV heads.
- GQA, RoPE, RMSNorm, QK normalization, SwiGLU.
- Tied input/output embeddings.
- Sequence length 512.
- 31,173,504 parameters.
- Auxiliary future-token objective.

Its training already uses two optimizer families:

- Muon for 2D hidden matrices.
- AdamW for embeddings, norms, biases, and other non-Muon parameters.

It also uses source/quality weighting and dialect-balanced sampling.

Saved training results:

- Packed train blocks: 2,221.
- Non-padding tokens in one packed train pass: about 1,121,830.
- Training: 48 epochs, 1,104 optimizer steps.
- Best validation PPL: 46.7405 at epoch 24.
- Final validation PPL: 56.8816.
- Final train PPL: 8.2477.
- The deterioration after epoch 24 is clear overfitting.

Saved classification results:

| Model | Internal 13-way macro-F1 | External-present macro-F1 |
|---|---:|---:|
| BLEACH dense classifier | 0.5930 | 0.1253 |
| Frozen BLEACH embedding + logistic regression | 0.6896 | 0.1511 |
| Initial char TF-IDF SVM | 0.7100 | 0.1399 |
| Final lexical ensemble | 0.7239 | 0.1339 |
| Best external-focused lexical model | not the internal winner | 0.1517 |

The lexical baseline is a strong baseline that must remain in the final paper. It also warns that the current benchmark is dominated by orthographic/source cues.

## 4. Publication-blocking findings

### 4.1 Critical: Vashantor loads the wrong column

The Vashantor files contain:

- bangla_speech: standard Bangla.
- region_bangla_speech: the actual regional sentence.

The notebook calls a generic text-column inference function. Its priority list contains bangla_speech, so it selects standard Bangla and labels that text as BAR, CHI, MYM, NOA, or SYL.

This is confirmed by the saved notebook output, which says:

- regional_col=bangla_speech
- target_col=None

This creates contradictory examples: near-identical standard sentences receive five different regional labels.

Measured overlap among the selected standard-Bangla sets is extreme:

- Train: roughly 1,869–1,874 shared sentences out of 1,875 for every pair of regions.
- Validation: up to all 250 shared.
- Test: roughly 368–373 shared out of 375.

The actual regional columns have almost no cross-region exact overlap.

Action:

- Replace inference with an explicit schema per dialect.
- Use region_bangla_speech as input.
- Use bangla_speech as the standard target.
- Group the five regional versions of the same semantic sentence with one pair/group ID.
- Recompute every tokenizer, split, CLM, classification, and translation artifact.

### 4.2 Critical: held-out Vashantor data re-enters training through synthetic CSV

The MYM/NOA balancing CSV contains rows whose source names explicitly include Vashantor Test and Vashantor Validation, but the CSV has no split_original field. The registry therefore assigns synthetic_train.

At least 1,598 rows have held-out ancestry:

- Mymensingh test and augmented test: 460.
- Mymensingh validation and augmented validation: 295.
- Noakhali test and augmented test: 534.
- Noakhali validation and augmented validation: 309.

The existing leakage test checks only split_original. It does not inspect ancestry encoded in source or source_original, so it passes incorrectly.

Action:

- Quarantine every synthetic row until it has an explicit parent-row list.
- Reject any row with a validation/test ancestor.
- Require source, parent_dataset, parent_split, parent_row_id, generator, prompt hash, seed, and generation version.
- Build split membership from the ancestry graph, not from a user-supplied text field.

### 4.3 Critical: a missing Barishal validation file

The ZIP filename contains two spaces:

Barishal  Validation Translation.csv

The notebook constructs a filename with one space and misses the 250 Barishal validation rows. This produces BAR=375 external rows while the other four labels have 625.

Action:

- Use a checked dataset manifest with exact archive member paths.
- Validate expected row counts for every dialect and split.
- Fail closed when a required file is absent; never continue with an imbalanced benchmark silently.

### 4.4 High: duplicate regional corpus is loaded twice

BanglaRegionalTextCorpus.xlsx and Regional_cleaned_dataset.xlsx contain the same 4,653 regional texts in the same row order. They have 4,600 unique normalized texts, and all row-position texts match after normalization.

Exact deduplication later removes most duplicate text/label pairs, but loading both obscures provenance and can leave cleaned near-variants.

Action:

- Retain the cleaned version once.
- Preserve a link to the raw record rather than treating it as another source.

### 4.5 High: synthetic cross-dialect conflicts and templates

Across the three synthetic CSVs:

- Rows: 51,101.
- Unique normalized texts: 45,709.
- Texts appearing under more than one dialect: 4,039.
- Rows involved in cross-dialect duplicate text: 9,430.

In bangla_dialect_balancing_synthetic_v0.csv alone:

- Exact duplicate rows by text: 4,234.
- Cross-dialect duplicate texts: 2,882.
- Rows involved: 7,116.

The current setting intentionally preserves cross-dialect duplicate text. That is unsafe for dialect classification because identical input cannot have multiple correct dialect labels unless it is explicitly annotated as shared/ambiguous.

Action:

- Assign a shared/ambiguous status after human review, or remove conflicting instances from supervised classification.
- Keep shared forms for CLM only, without a forced dialect routing target.
- Split template families as groups.
- Report real-only and real-plus-synthetic results separately.

### 4.6 High: BanglaDial is a merged corpus and may contaminate source holdouts

BanglaDial is itself assembled from multiple earlier corpora, including Vashantor. Its two-column format has lost component-level provenance. A Vashantor source-held-out evaluation can therefore still leak through BanglaDial.

Action:

- Exact-, near-, and semantic-deduplicate BanglaDial against all official Vashantor splits before splitting.
- For the strictest Vashantor OOD protocol, exclude any BanglaDial row connected to Vashantor.
- Do not call Vashantor a source-OOD test unless this has been verified.

### 4.7 High: model-data scale mismatch

The dense model sees only about 1.12M non-padding tokens per unique packed pass, then repeats them for 48 epochs. More than 61% of the 32k token IDs occur at most five times in the train data.

Consequences:

- 32k may be too large for this corpus and model size.
- 12.3M of 31.2M parameters are in the embedding table.
- Repeated-data memorization is likely.
- A sparse MoE would have even fewer unique training tokens per expert.

Action:

- Add a large, legally usable general Bangla pretraining corpus.
- Target roughly 1–2B clean unique tokens for the recommended model scale.
- Limit repeated dialect data to a pre-registered cap, normally no more than four effective passes before an ablation.
- Compare 16k, 24k, and 32k vocabularies at matched model compute.

### 4.8 High: current external results are an invalid diagnostic

Five-way random-chance accuracy is 0.20. The external-present models obtain about 0.21 accuracy and 0.15 macro-F1. The original Vashantor work reports much stronger region identification, so the near-chance result should have triggered a schema audit.

The external CLM PPL is also much lower than internal validation PPL, despite classification collapse. That paradox is explained by mislabeled repeated standard Bangla.

Action:

- Retire all external scores from substantive claims.
- Keep them only in an audit appendix showing how schema leakage can produce misleading evaluation.

### 4.9 Medium: tokenizer quality is evaluated mostly intrinsically

Zero UNK is partly guaranteed by character fallbacks. Low fertility does not alone demonstrate better modeling. The custom vocabulary protects thousands of full tokens and consumes many embedding parameters.

Action:

- Compare against standard algorithms under fixed data, split, vocabulary, and model budget.
- Require downstream and language-model evidence, not only fertility.

### 4.10 Medium: monolithic notebooks limit reproducibility

The notebooks contain very large cells, environment-specific Kaggle paths, repeated source definitions, and training plus paper reporting in one execution chain.

Action:

- Preserve the notebooks as legacy evidence.
- Move final work into tested Python modules and configuration files.
- Use notebooks only for audit and visualization.

## 5. Research questions and pre-registered hypotheses

### RQ1: Protocol

How much of current dialect-identification performance survives semantic-group deduplication and source-held-out evaluation?

H1: Internal random/row splits substantially overestimate macro-F1 relative to source-OOD and independent human test sets.

### RQ2: Tokenizer

Does a dialect-balanced tokenizer improve worst-dialect modeling and robustness at matched vocabulary and model compute?

H2: The selected dialect-aware tokenizer improves worst-dialect bits-per-character and macro-F1 without increasing token-cost inequality.

### RQ3: Sparse architecture

Does a shared/fine-grained MoE outperform an active-compute-matched dense SLM for Bangla dialect modeling?

H3: A shared-plus-top-2 routed MoE improves worst-dialect and average OOD metrics at similar active FLOPs.

### RQ4: Routing

Can routing be made dialect-sensitive but dataset-source-insensitive?

H4: Dialect-guided/source-adversarial routing increases conditional expert–dialect association while reducing expert–source association, and this predicts better source-OOD performance.

### RQ5: Optimization

Does Muon plus AdamW improve token efficiency for this small MoE over AdamW alone?

H5: The hybrid optimizer reaches a fixed validation loss with fewer training tokens, without worse stability or final OOD performance.

## 6. Intended contributions

Subject to results, the paper can make four defensible contributions:

1. A leakage-audited Bangla dialect benchmark protocol with full provenance and source-OOD evaluation.
2. A controlled study of dialect-aware tokenization across 12 regional labels plus standard Bangla.
3. Boichitro-MoE: a compact shared/fine-grained MoE with dialect-guided, source-invariant routing.
4. A layer-wise routing analysis showing when experts learn lexical, dialectal, shared-language, or dataset-source behavior.

Do not use “first” or “state of the art” until the final literature and benchmark sweep is complete.

## 7. Data rebuild

### 7.1 Canonical row schema

Every row must contain:

- row_id: immutable hash.
- text_raw and text_clean.
- dialect_label and taxonomy_version.
- source_dataset and source_version.
- archive_member and source_row_id.
- license and redistribution_status.
- text_role: regional, standard target, monolingual, lexicon, synthetic.
- pair_id: parallel semantic pair.
- semantic_group_id: all dialect/standard versions of the same content.
- exact_group_id, near_duplicate_group_id, template_group_id.
- split_original and split_final.
- is_synthetic.
- parent_row_ids and parent_splits.
- generator_model, prompt_hash, seed, and generation_version.
- quality scores and human-audit status.

No row enters training if any parent or connected semantic group belongs to validation/test.

### 7.2 Explicit source adapters

Implement and unit-test one adapter per source.

Vashantor:

- Regional input: dialect-specific bangla_speech column.
- Standard target: bangla_speech.
- Preserve official train, validation, and test.
- Create one semantic group across the five regions and standard form.
- Include the double-space Barishal validation filename through the manifest.

ChatgaiyyaAlap:

- Regional input: চট্টগ্রাম.
- Standard target: বাংলা.
- Dictionary remains lexicon metadata; it is not duplicated as sentences.

Sylheti:

- Regional input: Local Bangla Dialect(Sylheti) Text.
- Standard target: Standard Bangla Text.

BanglaRegionalTextCorpus:

- Use the cleaned sentence file once.
- Preserve the raw row as metadata.

BanglaDial:

- Keep its label mapping.
- Mark component provenance as unknown.
- Run aggressive cross-source contamination checks.

### 7.3 Cleaning

Cleaning must be conservative and dialect-preserving:

- Unicode NFC.
- Remove controls, zero-width artifacts, broken replacement characters, and HTML residue.
- Normalize whitespace.
- Retain meaningful punctuation, repeated emphasis, dialect spelling, and code-mixing.
- Keep raw and clean versions.
- Do not standardize dialect spelling into standard Bangla.

Add tests for Bengali combining marks, nukta/virama behavior, zero-width joiners, Bengali digits, punctuation, Latin-script Banglish, and mixed-script text.

### 7.4 Deduplication before splitting

Use a connected-component approach:

1. Exact raw hash.
2. Exact normalized hash.
3. Punctuation/digit-insensitive hash for template detection, not automatic deletion.
4. Character 5-gram MinHash/LSH.
5. Token Jaccard.
6. Multilingual sentence-embedding similarity for high-risk cross-source pairs.
7. Explicit pair, semantic, and generation-parent links.

All connected examples must remain in one split.

For classification, identical text with multiple labels is:

- removed from unambiguous classification;
- marked shared/ambiguous after review; or
- evaluated with set-valued labels in a separate analysis.

### 7.5 Split protocols

Freeze all splits before tokenizer or model training.

Protocol A — clean IID:

- Stratified by dialect and source.
- Grouped by exact, near-duplicate, semantic, and template IDs.
- Real validation/test only.

Protocol B — leave-one-source-out:

- Train on all eligible sources except one.
- Test on the unseen source for overlapping dialects.
- Repeat across feasible sources.

Protocol C — Vashantor official:

- Preserve official train/validation/test.
- Remove Vashantor-connected rows from other training sources.
- Evaluate 5-way identification and dialect-to-standard translation.

Protocol D — synthetic/template OOD:

- Train synthetic templates and generators are disjoint from test template families.
- Test remains human-authored.

Protocol E — independent human test:

- Recommended: 200 items per label for 13 labels, 2,600 total.
- Three native or region-competent annotators per item.
- Collect from sources absent from all training corpora.
- Record consent/licensing and redact personal information.

Primary label policy:

- Report regional-only macro-F1 separately from 13-way macro-F1 because STD is a register/control label.
- Use the full 13-way task only if all labels pass quality and minimum-source gates.
- Otherwise pre-register the verified 10-label primary task and retain 13-way as secondary.

### 7.6 General Bangla pretraining corpus

A 31M–160M model cannot be credibly trained from scratch on roughly 1M unique dialect tokens.

Recommended target:

- 1.5B clean, deduplicated Bangla/subword tokens.
- Minimum viable target: 300M.
- Include standard Bangla, conversational text, literature where licensed, Wikipedia, educational text, and a controlled amount of Banglish/code-mixing.

Candidate sources include Bangla2B+, Sangraha/Indic corpora, Bengali Wikipedia, and licensed open corpora. Build a license ledger before downloading or redistributing anything.

Required filters:

- language/script identification with code-mix retention.
- document and paragraph deduplication.
- benchmark contamination search.
- PII and unsafe-content screening appropriate to release policy.
- quality classifier plus rule-based artifact filters.
- domain distribution report.

Never use benchmark validation/test text for tokenizer training, vocabulary mining, language-model pretraining, or hyperparameter selection.

### 7.7 Synthetic-data policy

Quarantine the current 51,101 rows. Re-admit only rows that pass:

- train-only ancestry.
- no exact/near conflict with held-out text.
- no conflicting dialect label.
- source/template family tracking.
- automatic fluency and diversity checks.
- native-speaker audit on a stratified sample.

Use synthetic data as an ablation, not as an invisible part of the main result:

- Real only.
- Real plus audited synthetic.
- Real plus audited synthetic with quality weighting.

No synthetic row is allowed in validation or test.

### 7.8 Human quality audit

For every dialect/source stratum, sample at least 100 real rows and 100 admitted synthetic rows when available.

Annotate:

- dialect authenticity.
- grammaticality/fluency.
- semantic coherence.
- likely standard-Bangla leakage.
- source/template artifact.
- harmful or identifying content.

Report:

- acceptance rate by source/dialect.
- Krippendorff alpha or Fleiss kappa.
- adjudication protocol.
- annotator regional competence and compensation.

## 8. Tokenizer v2: DiaTok-BN study

### 8.1 What to retain

Retain:

- conservative Unicode handling.
- deterministic builds and hashes.
- dialect tags.
- protected-token audit tables.
- token-to-dialect statistics.
- zero-UNK character fallback.
- frozen model-I/O manifest.

Repair:

- rebuild from clean train only.
- exclude mislabeled Vashantor and held-out-derived synthetic.
- reduce reliance on synthetic-only protected words.
- remove arbitrary candidate score thresholds not tied to downstream performance.

### 8.2 Tokenizer candidates

Train every candidate on exactly the same clean tokenizer corpus.

| ID | Algorithm | Vocab sizes |
|---|---|---|
| T0 | Existing deterministic WordPiece rebuilt cleanly | 16k, 24k, 32k |
| T1 | Standard WordPiece | 16k, 24k, 32k |
| T2 | SentencePiece Unigram | 16k, 24k, 32k |
| T3 | Byte-fallback BPE | 16k, 24k, 32k |
| T4 | Parity-aware BPE | 16k, 24k, 32k |
| T5 | DiaTok-BN: parity sampling plus real-supported dialect lexicon quotas | 16k, 24k, 32k |

Optional diagnostic:

- byte/patch model such as a small BLT-style baseline, if implementation time permits.

### 8.3 Proposed DiaTok-BN design

The proposed tokenizer is a testable method, not simply a manually enlarged word list:

1. Balance tokenizer training sampling by dialect and source.
2. Optimize worst-dialect compression alongside global compression.
3. Reserve a small, fixed vocabulary budget for shared high-frequency morphemes.
4. Reserve per-dialect quotas only for tokens supported by real train data.
5. Penalize synthetic-only and high-source-purity tokens.
6. Retain character/byte fallback.
7. Keep dialect tags atomic, but do not expose gold tags during identification evaluation.

Protected-token budgets should be much smaller than the current 3,679 lexical symbols unless downstream evidence supports them.

### 8.4 Intrinsic evaluation

Report by dialect, source, and script:

- UNK rate.
- tokens per whitespace word.
- characters per token.
- bytes per token.
- compression ratio.
- normalized sequence length.
- single-token retention rate.
- vocabulary coverage.
- token-cost Gini coefficient across dialects.
- max/min and standard deviation of fertility.
- percentage of vocabulary with train count 0, 1–5, 6–20, and above 20.
- morphological-boundary F1 on a small manually segmented set, if available.
- robustness under spelling variation, punctuation, digits, and Romanization.

Zero UNK alone is not an acceptance criterion.

### 8.5 Downstream-controlled tokenizer evaluation

Train a 20M–35M dense proxy for a fixed token/FLOP budget under the top tokenizer candidates.

Compare:

- bits per byte/character, because PPL is not comparable across tokenizers.
- clean IID and source-OOD dialect macro-F1.
- dialect-to-standard chrF++ on a small controlled run.
- tokens/second and memory.

Select the final tokenizer only after this controlled study.

## 9. Model design

### 9.1 Architecture ladder

Use a pilot ladder before the main run.

| Scale | Total parameters | Active parameters | Purpose |
|---|---:|---:|---|
| Pilot | about 70–90M | about 35–45M | Router/optimizer/tokenizer sweeps |
| Recommended main | about 155–165M | about 75–80M | Primary paper model |
| Ambitious confirmation | about 350–450M | about 120–160M | Scaling confirmation only |

Exact counts must be generated from code and included in the model card.

### 9.2 Recommended main configuration

- Decoder-only Transformer.
- 16 layers, hidden size 512.
- 8 attention heads, 2 KV heads.
- RoPE, RMSNorm, QK norm, SwiGLU.
- Context 1,024 for pretraining; evaluate 512 and 1,024.
- Tied input/output embeddings.
- Selected 16k/24k/32k tokenizer from the tokenizer study.
- Four dense FFN layers and 12 MoE FFN layers.
- Dense FFN intermediate width around 1,536.
- One always-active shared expert per MoE layer.
- Eight fine-grained routed experts per MoE layer.
- Shared and routed expert intermediate width around 768.
- Top-2 routed experts plus the shared expert.
- Softmax before top-k.
- Dropless routing on a single GPU; if distributed, measure capacity overflow and dropped tokens explicitly.
- Multi-token prediction with one auxiliary future head initially; two-head MTP is an ablation.

With 32k vocabulary and routed expert width around 768, this is approximately 160M total and 78M active parameters. Final dimensions should be adjusted after measured FLOP matching.

### 9.3 Why GQA is the default instead of MLA

DeepSeek’s MLA is valuable for large-model KV-cache efficiency. At this scale and 1,024-token context, GQA is simpler, already implemented, and easier to compare fairly. MLA can be a pilot ablation, but it should not be included merely to make the architecture look advanced.

### 9.4 DeepSeek-derived MoE components

Use:

- shared-expert isolation for common Bangla knowledge.
- fine-grained routed experts.
- auxiliary-loss-free dynamic bias for load balancing.
- top-k routing with normalized selected weights.
- multi-token prediction.

Measure rather than assume expert specialization.

### 9.5 Proposed routing contribution

Boichitro-MoE should route from model states, not from a gold dialect label.

For token t at MoE layer l:

router logits = learned hidden-state logits + load-balance bias + annealed causal dialect-evidence prior.

The dialect-evidence prior is produced from a prefix-only running mean after
the dense prefix and a learned auxiliary dialect predictor. It is trained only
on clean training labels and maps the resulting soft dialect evidence to expert
seeds. It must:

- never use validation/test frequency.
- be available from the input itself.
- be annealed from a stronger warm-start value to a small or zero value.
- have a no-prior ablation and a randomized evidence-to-expert negative control.

Add sequence-level robustness supervision with two training heads:

- Dialect head: predicts dialect and encourages dialect-relevant allocation.
- Source head on the final causal sequence representation through gradient
  reversal: tries to predict dataset source while the shared representation
  and upstream routing pathway learn to hide source identity.

This yields a dialect-sensitive/source-insensitive objective without giving the router a gold label at inference.

For ambiguous/shared rows:

- omit dialect routing supervision; or
- use a soft/set-valued target.

For general Bangla:

- use STD/general labels only where appropriate.
- keep the shared expert active.

### 9.6 Dialect tags

Tags are useful for controllable generation but can leak labels in identification.

Training policy:

- 50% tag dropout during dialect-adaptive pretraining.
- A separate unknown-dialect tag.
- No gold tag in dialect identification, OOD classification, or unconditional CLM evaluation.
- Gold tag allowed only in explicitly labeled controllable-generation and oracle-routing experiments.

Report:

- untagged inference.
- predicted-tag inference.
- gold-tag oracle upper bound.

### 9.7 Dense-to-MoE upcycling

Recommended training path:

1. Train a dense Bangla foundation with the final tokenizer.
2. Split/clone dense FFN weights into shared and routed expert segments.
3. Use small expert-specific perturbations or virtual-group initialization.
4. Initialize router biases for near-uniform load.
5. Continue general Bangla pretraining so experts diverge before dialect adaptation.
6. Apply dialect/source routing objectives only in the dialect-aware stage.

Compare against:

- continued dense training.
- MoE trained from scratch at pilot scale.
- standard cloned-expert upcycling.
- abrupt bank release, annealed cross-bank release, and permanent
  complementary-bank top-2 routing under an otherwise matched validation-only
  pilot. Carry only a strategy that passes the frozen no-regression guard into
  the full continuation and task stages.

## 10. Training program

### Stage A — dense foundation

Goal: learn general Bangla before asking sparse experts to specialize.

- 1.0–1.5B clean general-Bangla tokens for the recommended run.
- One main pass through deduplicated data where possible.
- 1,024-token packed sequences.
- Document-aware packing.
- No benchmark text.
- Evaluate by tokens seen, not epochs.

### Stage B — MoE upcycling and general continuation

- Upcycle the selected dense checkpoint.
- Continue on 200–400M general Bangla tokens.
- Enable loss-free balancing.
- Preserve the validation-selected complementary-bank topology when exact
  partitioned FFN reconstruction is required; do not introduce an unregistered
  unrestricted-routing switch at the end of training.
- Do not enable dialect/source auxiliary heads yet.
- Confirm experts diverge without collapse.

### Stage C — dialect adaptation

Recommended budget: 15–25M tokens.

Mixture target:

- 50–60% general Bangla replay.
- 20–30% clean authentic dialect text, capped near four effective passes.
- 10–15% parallel regional-to-standard examples.
- 5–10% Romanized/code-mixed/noise-robustness examples.

If the authentic corpus cannot fill its share without exceeding the repeat cap, reduce the stage or collect more data. Do not silently repeat it dozens of times.

Enable:

- dialect routing head.
- source-adversarial routing head.
- source×dialect group-robust weighting.
- tag dropout.

### Stage D — multitask supervised adaptation

Tasks:

- dialect identification.
- dialect-to-standard translation.
- standard-to-dialect controlled generation where pairs exist.
- Romanized-to-Bengali normalization where licensed data exists.

Use task tokens and a mixed-task sampler. Keep a general-CLM replay fraction to reduce catastrophic forgetting.

### Stage E — optional distillation

If the final use case requires a smaller deployable model:

- distill MoE logits and hidden/routing knowledge into a dense 30M–80M student.
- report this as deployment work, not as the primary architectural comparison.

## 11. Two-optimizer training

### 11.1 Parameter ownership

Muon:

- attention projection matrices.
- dense FFN matrices.
- shared-expert matrices.
- routed-expert matrices.

AdamW:

- token embeddings and tied output parameters.
- normalization gains.
- biases.
- router matrices and balance biases.
- auxiliary classification heads.
- MTP heads.

Add a unit test that asserts:

- every trainable parameter belongs to exactly one optimizer group.
- no parameter is missing.
- no parameter appears twice.

### 11.2 Initial sweep ranges

Pilot, do not hard-code before sweeping:

- Muon LR: 0.01, 0.02, 0.03.
- AdamW LR: 2e-4, 3e-4, 5e-4.
- Router AdamW LR: 1e-4, 2e-4, 3e-4.
- Weight decay: 0.05, 0.10.
- Warmup: 2%, 5%.
- AdamW betas: (0.9, 0.95).
- Muon momentum: 0.95 with Nesterov.
- Gradient clipping: 1.0, with per-group diagnostics.

Use a maintained, tested Muon implementation rather than relying on an unverified notebook copy.

### 11.3 Scheduling

- One shared global step.
- Independent peak LRs but synchronized warmup/cosine schedules.
- Decay to 10% of peak.
- Log update-to-weight ratio by parameter family.
- Monitor QK logits, router logits, gradient norms, and non-finite values.

### 11.4 Loss

Primary pretraining loss:

L = group-robust next-token loss
  + MTP weight × future-token loss
  + dialect-route weight × dialect loss
  + source-adversarial weight × source loss through gradient reversal
  + router-z weight × router stabilization.

Initial pilot ranges:

- MTP weight: 0, 0.05, 0.10.
- Dialect-route weight: 0.02, 0.05, 0.10.
- Source-adversarial weight: 0, 0.01, 0.03.
- Router z-loss: 1e-4, 1e-3.

Load balancing in the proposed model uses dynamic expert biases, not a gradient-producing balance loss. A conventional auxiliary-balance model is required as a control.

### 11.5 Data weighting

Use groups defined by dialect × source × real/synthetic.

- Main model: regularized GroupDRO or capped exponentiated group weights.
- Prevent a tiny/noisy group from dominating with weight floors/ceilings.
- Synthetic quality weights are calibrated on human-audited samples.
- Report the effective number of tokens drawn from every group.

## 12. Baselines

### 12.1 Non-neural and encoder baselines

- Majority and stratified random.
- Character n-gram linear SVM.
- Character and word TF-IDF logistic regression.
- Current weighted lexical ensemble.
- BanglaBERT fine-tuning.
- XLM-R or a current multilingual encoder.
- A source-only classifier to quantify dataset artifacts.

### 12.2 Translation baselines

- Original Vashantor mT5/BanglaT5-style setup.
- BanglaT5.
- mT5.
- mBART-50 or NLLB-200 where licensing permits.
- A current small multilingual decoder adapted with LoRA.

### 12.3 Generative architecture baselines

- Existing BLEACH-Dense-32M rebuilt on clean data.
- Active-compute-matched dense model.
- Total-parameter-matched dense model as a high-compute upper bound.
- Switch-style top-1 MoE.
- Generic top-2 MoE without a shared expert.
- DeepSeek-style shared/fine-grained MoE.
- Proposed Boichitro-MoE.

### 12.4 Fairness rules

For the main dense/MoE comparison:

- same tokenizer.
- same train data and order per seed.
- same tokens seen.
- same context and effective token batch.
- measured, approximately matched active FLOPs.
- same downstream data and selection rule.
- three seeds for every main neural model.

The total-parameter-matched dense model is not a compute-matched baseline and must be labeled accordingly.

## 13. Core experiment matrix

| ID | Model | Shared expert | Routing | Source robustness | Optimizer | Full seeds |
|---|---|---|---|---|---|---:|
| C0 | Char TF-IDF SVM | n/a | n/a | no | standard | 5 |
| C1 | BanglaBERT | n/a | n/a | ERM | AdamW | 3 |
| C2 | Active-matched dense | n/a | dense | GroupDRO control | Muon+AdamW | 3 |
| C3 | Switch top-1 MoE | no | generic | no | Muon+AdamW | 3 |
| C4 | DeepSeek-style top-2 MoE | yes | loss-free | no | Muon+AdamW | 3 |
| C5 | C4 + causal dialect-evidence prior | yes | dialect-guided | no | Muon+AdamW | 3 |
| C6 | Proposed full model | yes | dialect-guided | source adversary + GroupDRO | Muon+AdamW | 3 |
| C7 | Total-matched dense | n/a | dense | same as C6 where applicable | Muon+AdamW | 3 if budget allows |

Use one-seed pilot runs to select stable hyperparameters. Do not use pilot results as final evidence.

## 14. Required ablations

### 14.1 Data/protocol

- Old loader/split versus repaired loader/split, labeled as audit only.
- Row split versus connected-group split.
- Real-only versus real plus audited synthetic.
- ERM versus source×dialect GroupDRO.
- With and without general Bangla replay.
- With and without BanglaDial rows connected to Vashantor.

### 14.2 Tokenizer

- Existing clean-rebuilt tokenizer versus standard Unigram/BPE/WordPiece.
- 16k versus 24k versus 32k.
- No protected dialect lexicon.
- Shared-token quota only.
- Full DiaTok-BN.
- Tag present, predicted, dropped, and oracle.

### 14.3 MoE architecture

- No shared expert versus one shared expert.
- 4, 8, and 16 routed experts at matched active FFN compute.
- Top-1 versus top-2.
- Coarse experts versus fine-grained segmentation.
- Auxiliary balance loss versus auxiliary-loss-free bias.
- Dropless versus fixed-capacity routing, if distributed.
- Scratch versus dense-to-MoE upcycling.

### 14.4 Proposed routing

- No causal dialect-evidence prior.
- Lexical prior only during warmup.
- Lexical prior retained at low weight.
- Dialect routing head off/on.
- Source adversary off/on.
- GroupDRO off/on.
- Gold dialect oracle upper bound.
- Randomized dialect lexicon negative control.

The randomized-lexicon control is important: it checks whether gains come from meaningful dialect evidence rather than extra router parameters or regularization.

### 14.5 Optimization

- AdamW only.
- Muon plus AdamW.
- MTP off versus one future token versus two.
- Source/quality weighting off/on.
- Early stopping versus fixed token budget.

## 15. Evaluation

### 15.1 Language modeling

Report:

- NLL and PPL only within the same tokenizer.
- bits per byte and bits per character across tokenizers.
- token accuracy.
- results per dialect, source, real/synthetic status, and frequency bin.
- worst-dialect result.
- rare-token loss.
- calibration of next-token probabilities.

### 15.2 Dialect identification

Primary metrics:

- macro-F1.
- balanced accuracy.
- MCC.
- regional-only macro-F1.
- worst-dialect F1.
- ECE and Brier score.

Protocols:

- clean IID.
- leave-one-source-out.
- Vashantor official 5-way.
- independent human test.
- Romanized/code-mixed.
- spelling and punctuation perturbation.

Always include:

- per-dialect precision/recall/F1.
- normalized confusion matrix.
- source-only and text-length-only diagnostic baselines.

### 15.3 Dialect-to-standard translation

Automatic:

- sacreBLEU with full signature.
- chrF++.
- TER.
- COMET or an appropriate multilingual learned metric, with a limitation note for Bangla.
- semantic similarity with a strong Bangla encoder.

Human:

- adequacy.
- standard-Bangla fluency.
- preservation of meaning and named entities.
- residual dialect leakage.

Use blinded pairwise comparison and at least three annotators per sampled output.

### 15.4 Controllable generation

Evaluate:

- requested-dialect accuracy using an independently trained classifier.
- semantic consistency across dialect generations from the same prompt.
- distinct-n and self-BLEU.
- repetition and memorization.
- native-speaker authenticity and fluency.
- tag controllability versus untagged generation.

Do not evaluate generation using only a classifier trained on the same sources; that would reproduce source shortcuts.

### 15.5 Robustness

Create controlled transformations:

- phonetic spelling variants.
- Bengali punctuation removal.
- digit and named-entity changes.
- keyboard errors.
- Romanization and mixed Bengali/Latin script.
- short versus long utterances.
- code-mixed English.

Plot performance against perturbation severity.

### 15.6 Efficiency

Report measured:

- total and active parameters.
- forward/backward FLOPs per token.
- tokens/second.
- peak VRAM.
- training GPU-hours.
- inference latency and throughput at batch 1 and batched.
- KV-cache memory.
- energy estimate where measurement is available.

For MoE:

- expert load CV and Gini.
- maximum/minimum load.
- overflow/drop rate.
- router entropy.
- number of unique experts activated per sequence.

## 16. Routing and interpretability analysis

The routing analysis is a core paper contribution, not an appendix decoration.

Measure by layer:

- mutual information between expert and dialect.
- mutual information between expert and source.
- conditional expert–source association given dialect.
- expert overlap between dialects.
- shared-expert contribution.
- routing stability under meaning-preserving spelling changes.
- routing change under dialect conversion of the same semantic sentence.
- routing differences for standard versus regional parallel pairs.

Causal tests:

- force or suppress the most dialect-associated expert.
- replace the dialect-evidence mapping with a shuffled mapping.
- steer middle layers toward shared experts.
- compare output loss and downstream behavior.

Interpretation rule:

- High expert–dialect association is not enough.
- It is useful only if expert–source association is controlled and OOD performance improves.

## 17. Figures

### Main paper figures

1. Data provenance and split graph: raw sources, dedup components, train/validation/test, and synthetic ancestry.
2. Data audit heatmap: dialect × source counts, real/synthetic ratios, and flagged leakage.
3. Tokenizer parity: fertility/STRR/compression by dialect for all finalist tokenizers.
4. Architecture diagram: dense layers, shared expert, top-2 routed experts, causal dialect-evidence prior, dialect head, source-adversarial head.
5. Training curves versus tokens seen: dense, standard MoE, and proposed MoE; include validation loss and worst-dialect loss.
6. Accuracy-efficiency Pareto plot: source-OOD macro-F1 or BPC versus active FLOPs, total parameters, and latency.
7. IID versus OOD performance: paired points per model with confidence intervals.
8. Layer × expert routing heatmap faceted by dialect.
9. Expert–dialect and expert–source information across layers.
10. Translation quality by dialect with automatic and human scores.
11. Robustness curves under spelling/Romanization severity.
12. Ablation forest plot showing effect size and 95% confidence intervals.

### Appendix figures

- Old wrong-column Vashantor overlap heatmap.
- Exact/near-duplicate component-size distribution.
- Vocabulary frequency histogram and unused/rare embedding fraction.
- Per-dialect tokenizer length distributions.
- Full confusion matrices for every protocol.
- Expert load and entropy over training.
- Expert activation Sankey or alluvial plot for selected parallel sentences.
- Per-source calibration diagrams.
- Synthetic quality and template diversity plots.
- Scaling curves for pilot/main sizes.

Avoid radar charts when a grouped dot/bar plot gives a clearer comparison.

## 18. Tables

1. Dataset inventory, version, license, role, and final row counts.
2. Audit findings and number of affected rows.
3. Tokenizer intrinsic and proxy-downstream results.
4. Model configurations, total/active parameters, and measured FLOPs.
5. Main clean-IID and source-OOD identification results.
6. Vashantor and other translation results.
7. Full ablation results.
8. Efficiency and training cost.
9. Human evaluation and agreement.
10. Per-dialect results with bootstrap confidence intervals.

## 19. Statistical protocol

- Three seeds for main neural systems.
- Five seeds for inexpensive classifiers.
- Report mean, standard deviation, and 95% confidence intervals.
- Use paired bootstrap over test examples/groups.
- For classification pairwise errors, use McNemar where appropriate.
- For translation, use paired bootstrap/resampling with sacreBLEU and chrF++.
- For human pairwise judgments, use a mixed-effects model with item and annotator random effects.
- Apply Holm correction to a pre-registered family of primary comparisons.
- Report effect sizes, not only p-values.

Primary comparison:

C6 proposed full model versus C4 standard DeepSeek-style MoE on source-OOD regional-only macro-F1.

Secondary comparisons:

- C6 versus active-matched dense.
- tokenizer T5 versus best standard tokenizer.
- Muon+AdamW versus AdamW.

## 20. Acceptance gates

### Data gate

- Zero train rows connected to validation/test through exact, near, semantic, template, or generation-parent links.
- Correct Vashantor regional columns.
- Expected row counts for every official split.
- All licenses documented.
- Human audit completed.

### Tokenizer gate

- Zero or negligible UNK with fallback.
- No dialect has extreme token-cost disadvantage.
- Vocabulary rare-token fraction materially lower than the current 61%.
- Proxy LM/downstream result is non-inferior to the best standard tokenizer and improves a pre-registered worst-dialect metric.

### MoE stability gate

- No non-finite loss.
- No persistent expert collapse.
- Zero token drops in the main single-GPU setup.
- Expert load CV/Gini within the pre-registered acceptable range.
- Router does not simply predict source.

### Main-result gate

The architectural claim is made only if:

- C6 beats C4 on the primary source-OOD metric with a positive paired confidence interval; and
- it does not lose more than one absolute macro-F1 point on clean IID; and
- efficiency is measured at comparable active compute.

If not, report the negative result honestly and center the paper on the audit/benchmark only if that contribution is independently strong enough.

## 21. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| Dialect labels reflect source rather than language | Inflated IID, failed OOD | source-held-out splits, source adversary, source-only baseline |
| Too little authentic dialect data | Expert memorization | general pretraining, upcycling, repeat cap, new collection |
| Synthetic templates dominate | Spurious routing | quarantine, ancestry, template grouping, real-only ablation |
| 32k vocabulary wastes parameters | weak SLM | 16k/24k/32k controlled study |
| MoE experts collapse | no capacity gain | loss-free bias, dropless routing, load monitoring |
| “Novelty by combination” criticism | weak paper | focus on dialect–source disentangled routing and causal analysis |
| Gold dialect tags leak labels | invalid classification | tag dropout, untagged primary evaluation |
| Two optimizers are mis-partitioned | silent training bug | exact parameter ownership test |
| PPL compared across tokenizers | misleading result | bits/byte and bits/character |
| Vashantor contamination through BanglaDial | invalid OOD | cross-source connected dedup and source-exclusive protocol |
| Human evaluation lacks regional competence | unreliable claims | region-matched annotators and adjudication |
| Compute budget grows through ablations | incomplete study | proxy sweeps, shared checkpoints, pre-registered pruning |

## 22. Recommended project structure

Create a new project directory; do not extend the two monolithic notebooks indefinitely.

    bangla_dialect_moe/
      README.md
      pyproject.toml
      configs/
        data/
        tokenizer/
        model/
        train/
        eval/
      src/
        data/
          adapters/
          registry.py
          normalize.py
          dedup.py
          split.py
          contamination.py
        tokenizer/
          train.py
          evaluate.py
          dialect_lexicon.py
        model/
          dense.py
          moe.py
          router.py
          upcycle.py
        training/
          optimizers.py
          losses.py
          sampler.py
          trainer.py
        evaluation/
          lm.py
          classification.py
          translation.py
          generation.py
          routing.py
      scripts/
        build_registry.py
        freeze_splits.py
        train_tokenizer.py
        train_dense.py
        upcycle_moe.py
        train_moe.py
        evaluate_all.py
        build_paper_figures.py
      notebooks/
        00_data_audit.ipynb
        01_tokenizer_analysis.ipynb
        02_training_diagnostics.ipynb
        03_routing_analysis.ipynb
        04_paper_figures.ipynb
      tests/
      artifacts/
      reports/
      paper/

Training must run from scripts/configs. Notebooks should read immutable reports rather than contain the only copy of model code.

## 23. Reproducibility checklist

- Immutable raw-file SHA-256 hashes.
- Dataset manifest and license ledger.
- Frozen split IDs.
- Full package lock.
- Git commit in every run.
- Config and random seed in every artifact.
- Dataset/tokenizer/model hashes.
- Hardware, CUDA, PyTorch, and kernel versions.
- Deterministic evaluation.
- Checkpoint resume test.
- Parameter-count and active-FLOP tests.
- Optimizer ownership test.
- No-overlap split test.
- Saved per-example predictions.
- Saved routing traces for a fixed analysis subset.
- Model card, tokenizer card, and data statement.

## 24. Compute tiers

### Minimum viable study

- 300M general Bangla tokens.
- Pilot model only.
- One main dense and one proposed MoE, three seeds for dialect adaptation.
- Core classification and Vashantor translation.

Estimated order: 100–200 96GB-GPU hours after code stabilization.

### Recommended Q1-grade study

- 1.0–1.5B general tokens.
- Recommended 160M-total/78M-active main model.
- Core matrix C2–C7, using shared dense checkpoints and proxy pruning.
- Three full seeds for finalists.
- Independent human test and human translation evaluation.

Estimated order: 300–600 96GB-GPU hours. Benchmark actual tokens/second before committing.

### Ambitious study

- Add a 350–450M scale confirmation.
- Add BLT/byte-patch diagnostic.
- Larger independent corpus and more translation pairs.

Estimated order: above 1,000 96GB-GPU hours.

These are planning ranges, not promises. Produce measured estimates from 1,000-step microbenchmarks.

## 25. Timeline

### Weeks 1–2: protocol repair

- Implement explicit adapters.
- Build provenance graph.
- Detect all contamination.
- Freeze clean splits.
- Complete initial human audit.

### Weeks 3–4: tokenizer study

- Train candidate tokenizers.
- Run intrinsic metrics.
- Train proxy LMs.
- Freeze tokenizer v2.

### Weeks 5–6: dense foundation and pilots

- Build general corpus.
- Run architecture/optimizer pilots.
- Train selected dense foundation.

### Weeks 7–8: MoE

- Upcycle.
- Stabilize loss-free routing.
- Run proposed routing ablations.

### Weeks 9–10: main training and downstream tasks

- Three-seed finalist runs.
- Identification, translation, generation, robustness.

### Weeks 11–12: human evaluation and analysis

- Human translation/generation review.
- Routing causal analysis.
- Statistical tests.

### Weeks 13–14: paper and release

- Freeze tables/figures.
- Reproduction run.
- Cards, data statement, limitations, ethics, and artifact packaging.

## 26. Immediate execution order

The next implementation work should occur in this exact order:

1. Create the new project skeleton.
2. Write a machine-readable manifest for every archive member.
3. Fix Vashantor columns and the Barishal validation filename.
4. Build semantic group IDs across parallel variants.
5. Quarantine the synthetic archive and reconstruct ancestry.
6. Cross-deduplicate BanglaDial and all external benchmarks.
7. Freeze clean IID, source-OOD, and official Vashantor splits.
8. Generate a red/green data-audit report.
9. Only after a green report, rebuild tokenizer candidates.
10. Only after tokenizer selection, train dense/MoE pilots.

## 27. What should be reused, rewritten, or retired

### Reuse

- Taxonomy mapping after review.
- Unicode cleaning tests.
- quality/provenance schema concepts.
- token/dialect statistics.
- tokenizer serialization and atomic tag handling.
- BLEACH dense architecture modules.
- lexical classifiers and error-audit framework.
- Muon/AdamW parameter-group concept.
- per-dialect PPL and plotting logic.

### Rewrite

- all source loaders.
- split and leakage logic.
- synthetic provenance.
- tokenizer candidate selection.
- packing/data samplers around immutable split IDs.
- MoE architecture and router.
- training harness as scripts.

### Retire from claims

- all current external classification/PPL results.
- the current 141,171-row merged registry as final data.
- the current 48-epoch training conclusion.
- any claim that zero UNK proves tokenizer superiority.
- any claim that 0.724 internal macro-F1 is the final dialect benchmark.

## 28. Literature anchors

Architecture and MoE:

- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Upcycling Large Language Models into Mixture of Experts](https://arxiv.org/abs/2410.07524)
- [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
- [Multilingual Routing in Mixture-of-Experts](https://arxiv.org/abs/2510.04694)

Small-model and optimization:

- [MobileLLM](https://arxiv.org/abs/2402.14905)
- [SmolLM2](https://arxiv.org/abs/2502.02737)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)

Tokenizer:

- [Tokenizer Choice for LLM Training: Negligible or Crucial?](https://aclanthology.org/2024.findings-naacl.247/)
- [Byte Latent Transformer](https://arxiv.org/abs/2412.09871)
- [TokLens](https://aclanthology.org/2026.acl-srw.18/)
- [Parity-Aware Byte-Pair Encoding](https://aclanthology.org/2026.acl-long.342/)
- [MUTANT: A Recipe for Multilingual Tokenizer Design](https://aclanthology.org/2026.acl-long.2146/)

Bangla and dialect resources:

- [Vashantor](https://arxiv.org/abs/2311.11142)
- [BanglaDial](https://pmc.ncbi.nlm.nih.gov/articles/PMC12597015/)
- [ChatgaiyyaAlap](https://pmc.ncbi.nlm.nih.gov/articles/PMC11925091/)
- [BanglaBERT](https://aclanthology.org/2022.findings-naacl.98/)
- [BhasaBodh](https://aclanthology.org/2025.banglalp-1.9/)
- [Bangla dialect translation and sentiment benchmark](https://aclanthology.org/2025.banglalp-1.26/)

## 29. Bottom line

The strongest paper is not “we added MoE and several modern tricks to the old notebook.” The strongest paper is:

> We expose and repair dialect/source leakage, build a controlled tokenizer and benchmark, and show whether a dialect-guided but source-invariant sparse SLM learns transferable linguistic experts under strict OOD evaluation.

That framing is technically coherent, falsifiable, and capable of supporting a serious journal-quality submission. The data repair is the first mandatory experiment.
