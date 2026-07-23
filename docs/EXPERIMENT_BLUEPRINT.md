# Boichitro-MoE: Q1 Experiment Execution Blueprint

Protocol version: 1.0
Status: execution-ready after the data gate
Local evidence: bangla_dialect_moe/reports/LOCAL_ARCHIVE_AUDIT.md
Parent document: BANGLA_DIALECT_MOE_Q1_RESEARCH_PLAN.md

## 1. Frozen research decision

### Working title

Boichitro-MoE: Source-Invariant Sparse Small Language Modeling for Bangla
Dialect Normalization

### Primary task

Source-blind dialect-to-Standard-Bangla text normalization for eight locally
supported regional varieties:

- BAR: Barishal/Barisal
- CHI: Chittagong/Chatgaiyya
- KHU: Khulna
- MYM: Mymensingh
- NAR: Narail
- NOA: Noakhali
- RAN: Rangpur
- SYL: Sylhet

The model receives dialect text without a gold dialect tag and generates
Standard Bangla. This is the primary task because it tests whether a generative
SLM and its sparse experts learn transferable dialect transformations. A pure
classification paper would not justify a decoder MoE of this complexity.

### Auxiliary specialization

Dialect identification over twelve regional labels plus Standard Bangla:

BAR, CHI, KHU, KIS, MYM, NAR, NOA, NSD, RAJ, RAN, SYL, TAN, and STD.

The 13-class result is released only if every class passes provenance, quality,
minimum-source, and independent-test gates. Otherwise:

- the verified regional subset is the primary identification result;
- 13-class identification is reported as a diagnostic;
- regional-only macro-F1 is always reported separately from STD.

### Supporting tasks

- Causal language modeling by dialect and source.
- Romanized-dialect to Standard Bangla normalization where real aligned data
  exist.
- Controlled Standard-Bangla-to-dialect generation as a secondary analysis.
- A classification-specialized bidirectional branch as a separate model, not
  as evidence for the causal normalization checkpoint.

### Central comparison

At matched active parameters, tokens, data, and measured active FLOPs:

Boichitro-MoE versus a standard shared-expert, top-2, loss-free MoE.

### Primary endpoint

Macro-averaged chrF++ on the locked source-independent normalization track for
the five trained dialects with an admissible independent source (BAR, CHI, MYM,
NOA, and SYL). IID chrF++ across all eight trained normalization dialects is the
broad-coverage secondary endpoint. RAJ is reported separately as a zero-shot
challenge because it has no normalization train/validation pairs; it is never
silently averaged into the supported-dialect endpoint.

The architectural claim is supported only if the hierarchical paired 95%
confidence interval for the difference is entirely above zero.

### Main scientific claim

The proposed router should encode dialect-relevant variation while suppressing
dataset-source shortcuts, improving source-OOD normalization at approximately
the same active compute as a dense SLM and a standard MoE.

### Claims that are not allowed in advance

- No guaranteed state-of-the-art claim.
- No claim that MoE is better unless the confirmatory comparison succeeds.
- No claim that the tokenizer is novel unless its controlled study succeeds.
- No use of current notebook external scores as evidence.
- No main result chosen from test-set inspection.
- No synthetic-data benefit claim without a real-only comparison.

## 2. What the local ZIP files actually provide

The reproducible audit fixes the local experimental foundation.

| Material | Rows | Scientific role |
|---|---:|---|
| Vashantor CSV pairs | 12,500 | Five-dialect normalization, Romanization, official evaluation |
| ChatgaiyyaAlap sentence pairs | 4,011 | CHI normalization and cross-source testing |
| Sylheti aligned pairs | 1,200 | SYL normalization and cross-source testing |
| BanglaRegionalTextCorpus | 4,653 | BAR, KHU, NAR, RAN normalization |
| Total aligned local pairs | 22,364 | Primary supervised task |
| BanglaDial | 63,303 | Conditional classification pool after reconstruction |
| Derived/augmented archive | 51,101 | Quarantined ablation pool |

Important facts:

- There are 22,345 unique dialect-tagged aligned pairs after normalized exact
  deduplication.
- The aligned data cover eight, not five, normalization dialects.
- BanglaDial includes Vashantor train, validation, and test text.
- The derived archive has 1,598 rows explicitly descended from Vashantor
  validation/test sources.
- The derived archive contains 4,039 normalized cross-label conflicts
  involving 9,430 rows.
- BanglaDial has 599 cross-label normalized texts and 3,885 rows containing an
  angle-bracket placeholder.
- Vashantor CSV must be used instead of JSON because four JSON test files
  contain split/incomplete records.
- Vashantor has six exact normalized regional-input overlaps between its train
  and validation partitions.
- BanglaRegionalTextCorpus and Regional_cleaned_dataset repeat the same
  normalized regional text set; only the former retains Standard Bangla
  targets.
- Neither ZIP includes license or README material.

These are protocol constraints, not optional cleaning preferences.

### Implemented preliminary data gate

The first executable build now produces a deterministic 136,768-row manifest:

- 85,667 raw rows and 51,101 quarantined derived rows;
- 109,174 role-aware exact components, maximum component size 23;
- 22,200 quality/duplicate-filtered normalization-task pairs;
- 17,190 preliminary normalization-train pairs across all eight dialects;
- zero derived rows with training eligibility;
- zero train-eligible rows in a protected exact component;
- 145 rows blocked because one exact regional source maps to multiple targets;
- 15,248 BanglaDial rows with an exact Vashantor source/target match;
- 14 passing regression tests.

This gate remains RED_REMAINING_GATES. It is evidence that the exact pipeline
works, not authorization to start final tokenizer/model training. Near/template
deduplication, original component provenance, licenses, human audit, the OOD
test, and the general Bangla manifest remain mandatory.

## 3. Task contracts

### 3.1 Normalization contract

Training sequence:

    <bos> <task_norm> <dial_unknown> regional source <sep> standard target <eos>

Loss:

- The primary sequence loss is applied only to the Standard Bangla target and
  end token.
- Prompt and source tokens are masked from target cross-entropy.
- A small causal replay batch retains ordinary next-token loss.
- The gold dialect label is not placed in the primary input.

Inference:

    <bos> <task_norm> <dial_unknown> regional source <sep>

Required reporting:

- Unknown-dialect inference: primary.
- Predicted-dialect tag: secondary.
- Gold-dialect tag: oracle upper bound only.

### 3.2 Identification contract

Causal classifier input:

    <bos> <task_id> text <cls>

The final <cls> state sees the complete sequence under the causal mask and feeds
a classification head. No gold dialect token is present.

Classification-specialized branch:

- Start from the same selected backbone.
- Enable bidirectional attention only in the ID branch.
- Run a short masked-next-token adaptation using training data only.
- Add supervised contrastive learning with same-dialect, different-source
  positives.
- Report its additional adaptation compute.
- Never use this bidirectional branch for normalization results.

This branch tests whether LLM2Vec-style conversion is useful for a compact
Bangla decoder; it is not silently mixed into the generative comparison.

### 3.3 Language-modeling contract

- Use untagged dialect input for the primary diagnostic.
- Report bits per character and bits per byte across tokenizers.
- Report perplexity only among models sharing the identical tokenizer.
- Partition every score by dialect, source, and real/synthetic status.

### 3.4 Controlled generation contract

Gold dialect tags are permitted only for explicitly requested
Standard-to-dialect generation. This task is secondary and cannot be used to
support source-blind identification or normalization claims.

## 4. Data protocol

### 4.1 Canonical source selection

| Archive member family | Use |
|---|---|
| Vashantor CSV | canonical |
| Vashantor JSON | exclude |
| BanglaDial CSV | canonical representation after row provenance |
| BanglaDial XLSX | duplicate; exclude |
| BanglaRegionalTextCorpus.xlsx | canonical four-column pair source |
| Regional_cleaned_dataset.xlsx | duplicate regional text; exclude |
| ChatgaiyyaAlap sentence CSV | canonical pair source |
| Chatgaiyya dictionary | lexicon metadata only |
| Sylheti pair CSV | canonical pair source |
| All three files in archive (1).zip | quarantine by default |

### 4.2 Required canonical row schema

Every admitted example receives:

- row_id: immutable SHA-256 identifier.
- text_raw and text_clean.
- target_raw and target_clean where aligned.
- dialect_label and taxonomy_version.
- source_dataset, source_version, archive_sha256, and archive_member.
- source_row_id.
- source_kind: human, curated, merged, synthetic, augmented, or unknown.
- split_original and split_final.
- pair_id and semantic_group_id.
- exact_group_id, near_group_id, and template_group_id.
- parent_row_ids and parent_split_ids.
- is_synthetic and generation metadata.
- license, redistribution status, and citation.
- quality flags and human-review status.

No missing provenance field may be replaced with a guessed value. Unknown is a
valid explicit value.

### 4.3 Cleaning

Apply only dialect-preserving operations:

- Unicode NFC.
- Whitespace and non-breaking-space normalization.
- Remove BOM, zero-width controls, HTML residue, and corrupt control
  characters.
- Preserve dialect spelling, repetition, punctuation, Bengali digits,
  Romanization, and code-mixing.
- Keep the raw field unchanged.

Hard-review conditions:

- empty after cleaning;
- angle-bracket placeholder;
- punctuation-only;
- source equals target;
- contradictory dialect labels;
- unexpectedly low Bengali/Latin letter content;
- extreme length;
- target contains unexplained dialect residue;
- regional input appears to be Standard Bangla.

Source-equals-target pairs are not automatically wrong. They form a
copy-required category and receive a separate quality label.

### 4.4 Connected-component deduplication

Create a graph with these edges:

1. exact raw text;
2. exact normalized text;
3. explicit parallel pair;
4. Vashantor split and row-index semantic alignment;
5. generation parent;
6. character 5-gram MinHash candidate above the frozen threshold;
7. token Jaccard candidate;
8. high sentence-embedding similarity after manual calibration;
9. matching Standard Bangla target plus high regional-source similarity;
10. shared synthetic template family.

All connected rows belong to one final split.

For Vashantor:

- the five regional variants and Standard Bangla form one semantic group per
  intended row;
- the six train-validation exact regional overlaps are assigned to validation
  or removed from training;
- all BanglaDial matches inherit the protected Vashantor split;
- no derived child of a protected row may enter training.

### 4.5 BanglaDial reconstruction

BanglaDial is not admitted wholesale.

Processing order:

1. Match every row to Vashantor, ONUBAD, Bhashamul/REGIPA, and known component
   releases where the originals can be acquired.
2. Assign component_source and original split where recoverable.
3. Remove or review 3,885 angle-placeholder rows.
4. Remove punctuation-only and corrupt records.
5. Resolve 599 cross-label text conflicts:
   - adjudicated single label;
   - explicitly ambiguous set label; or
   - exclusion from single-label classification.
6. Remove all rows connected to protected evaluation material.
7. Keep unknown-provenance rows out of source-OOD training claims.

BanglaDial may be used for clean IID training after this procedure. It is never
an independent external benchmark.

### 4.6 Synthetic archive policy

The default main run uses zero rows from archive (1).zip.

Re-admission gate:

- train-only parent ancestry;
- no protected semantic, exact, near, or template connection;
- one unambiguous dialect label;
- real-data support for dialectal markers;
- human authenticity and fluency audit;
- generator and template family recorded;
- quality calibration by dialect;
- repeat cap enforced.

The file bangla_dialect_balancing_synthetic_v0.csv is explicitly marked
candidate_not_native_reviewed and gives every row the same 0.62 quality value.
That value is not a measured quality score and cannot be used as a continuous
training weight.

Required synthetic experiments:

- real only;
- real plus audited synthetic;
- real plus audited synthetic with calibrated weights.

### 4.7 Frozen local splits

Protocol N-IID:

- Vashantor: preserve official roles after connected filtering.
- Unsplit pair datasets: stable hash-based 80/10/10 split by connected
  semantic group within dialect and source.
- Real validation and test only.
- No target or source semantic group crosses a split.

Protocol N-LOSO:

- BAR: Vashantor versus BanglaRegionalTextCorpus.
- CHI: Vashantor versus ChatgaiyyaAlap.
- SYL: Vashantor versus Sylheti1200.
- Train on one source family and test on the other; then reverse.
- MYM and NOA receive external-source folds when eligible data are acquired.
- KHU, NAR, and RAN cannot receive a local leave-one-source-out claim.

Protocol N-OOD:

- Locked authentic source absent from all training corpora.
- Eight-dialect macro average.
- Human-verified Standard Bangla reference.

Protocol ID-IID:

- Connected-group, source-stratified classification split.
- Ambiguous text excluded from single-label scoring.

Protocol ID-LOSO:

- Only labels with at least two verified source families.
- Repeat for every feasible held-out source.

Protocol ID-OOD:

- Locked independent human test.
- Regional-only and 13-class scores.

### 4.8 New human test required for the strongest paper

Boichitro-Norm-OOD:

- target: 200 independently sourced regional inputs per normalization dialect;
- total target: 1,600;
- two region-competent translators create or verify Standard Bangla;
- a third adjudicator resolves disagreements;
- inputs must not be translations of existing training prompts;
- sources, consent, licensing, PII redaction, and domain are recorded.

Boichitro-ID-OOD:

- target: 200 inputs per regional/STD label;
- total target: 2,600;
- source families absent from training;
- region-competent annotation and adjudication;
- report inter-annotator agreement and ambiguous cases.

Run a bootstrap power simulation on pilot predictions before finalizing sample
sizes. Minimum target sizes may increase; they may not decrease solely for
convenience.

### 4.9 External data that are actually needed

Local ZIP data remain the dialect-task foundation. External data serve three
specific gaps.

Essential gap A: general Bangla pretraining

- minimum credible clean budget: 300M tokens;
- recommended budget: 1.0B to 1.5B tokens;
- candidate families: licensed Bangla web corpora, Bengali Wikipedia,
  educational/public-domain text, BDNC where terms permit, and a controlled
  conversational/code-mixed portion;
- all material needs license, deduplication, PII, quality, and benchmark
  contamination reports.

Essential gap B: authentic independent OOD evaluation

- DIALTSA-BN: 600 authentic YouTube-derived items over BAR, CHI, SYL, and NOA;
- a new eight-dialect human set remains the preferred primary test;
- no external test enters tokenizer fitting, pretraining, or model selection.

Useful secondary benchmarks after license and overlap checks:

- BhasaBodh for CHI/SYL and Romanized normalization;
- BanglaCHQ-Prantik for medical-domain CHI/SYL robustness;
- BD-Dialect v2 for aligned NOA, SYL, CHI, RAJ, and MYM;
- ONUBAD originals to recover the component provenance and targets lost in
  BanglaDial;
- Kothon if its source independence and license pass audit.

Saptak must not be treated as independent because its own inventory combines
several earlier corpora, including Vashantor and ONUBAD. It can be a
provenance-aware training index, not a clean OOD benchmark.

## 5. Tokenizer experiment

### 5.1 Existing evidence

The existing tokenizer notebook produced a deterministic 32k WordPiece-like
tokenizer with zero UNK and mean fertility about 1.13. However:

- it was built before the final leakage repair;
- 19,593 of 32,000 tokens occur at most five times;
- zero UNK does not establish downstream superiority;
- its vocabulary is large for an approximately 80M-active SLM.

It remains baseline TOK-OLD, not the selected tokenizer.

### 5.2 Candidate grid

Train on the identical clean, train-only tokenizer corpus.

| Family | 16k | 24k | 32k |
|---|---|---|---|
| clean rebuilt existing WordPiece | yes | yes | yes |
| SentencePiece Unigram | yes | yes | yes |
| byte-fallback BPE | yes | yes | yes |
| DiaTok-BN parity-balanced candidate | yes | yes | yes |

DiaTok-BN components:

- dialect/source-balanced sampling;
- a small shared morpheme quota;
- per-dialect token quota supported by real train data;
- penalty for high source purity;
- penalty for synthetic-only evidence;
- byte/character fallback;
- atomic task, dialect, mask, class, separator, and control tokens.

### 5.3 Selection procedure

Phase T0, intrinsic screen:

- build all 12 candidates;
- compute train/dev fertility, bytes per token, character coverage, vocabulary
  frequency bins, token-cost Gini, and robustness;
- eliminate candidates Pareto-dominated on global compression,
  worst-dialect compression, rare-vocabulary fraction, and throughput.

Phase T1, one-seed proxy:

- select four candidates including at least one standard tokenizer and one
  DiaTok candidate;
- train the same 28M dense proxy for 25M tokens;
- measure development bits per character, worst-dialect bits per character,
  N-IID chrF++, ID macro-F1, throughput, and memory.

Phase T2, confirmatory proxy:

- top two candidates;
- seeds 1701, 2903, and 4307;
- 75M tokens per run;
- identical training data order within seed.

Freeze rule:

1. Candidate must be within 1% of the best global development BPC.
2. Candidate must use no more than 105% of the best median sequence length.
3. Among passing candidates, choose lowest worst-dialect BPC.
4. If the 95% paired interval includes a meaningful loss, choose the simpler
   standard tokenizer.

No test metric participates.

## 6. Model family

### 6.1 Exact active-compute design

Dimensions:

- decoder-only Transformer;
- 16 layers;
- hidden size 512;
- 8 query heads and 2 KV heads;
- head size 64;
- RoPE, RMSNorm, QK norm, SwiGLU, and tied embeddings;
- context 1,024;
- selected 16k, 24k, or 32k tokenizer.

Dense FFN width is 2,304.

Each MoE layer contains:

- one always-active shared expert, width 768;
- eight routed experts, each width 768;
- top-2 routed experts;
- selected routed weights normalized after top-k.

Therefore the active MoE FFN width is:

    shared 768 + routed 2 × 768 = 2,304

This matches the dense FFN width exactly before kernel overhead. Final fairness
uses measured FLOPs, not width alone.

Approximate scale by tokenizer:

- dense: roughly 75M to 84M total and active parameters;
- MoE: roughly 160M to 169M total and 75M to 84M active parameters.

Exact counts and FLOPs are generated from code and frozen in Table 4.

### 6.2 Layer map

Layers 1–4:

- dense FFN;
- general lexical/context processing.

Layers 5–8:

- shared plus routed MoE;
- optional train-only causal dialect-evidence prior during warm-start.

Layers 9–12:

- shared plus generic hidden-state routing;
- no direct dialect-evidence prior in the proposed default.

Layers 13–16:

- shared plus routed MoE;
- task-conditioned routing from the known task token;
- sequence-level dialect and source robustness supervision.

This early/middle/late schedule is a hypothesis motivated by multilingual MoE
routing studies. It requires comparison against identical routing at all
layers; it is not assumed correct.

### 6.3 Router

For token t and layer l:

    router_logit(l,t) =
        hidden_projection(l,t)
        + alpha(l) × causal_dialect_evidence(l,t)
        + task_projection(l)
        + dynamic_load_bias(l)

Rules:

- task_projection is nonzero only in layers 13–16;
- causal dialect evidence is nonzero only in layers 5–8 in the default;
- evidence comes from a prefix-only running-mean state and an auxiliary
  dialect predictor learned on training data only;
- alpha is warmed down to zero or the pilot-selected small value;
- no gold dialect label is part of the inference router;
- softmax precedes top-k;
- single-GPU training is dropless;
- distributed training logs capacity and dropped tokens.

Sequence router summary:

- average selected probabilities and expert usage over non-padding tokens;
- dialect head predicts dialect;
- source head receives the final causal sequence representation through
  gradient reversal, so its reversed gradient regularizes the shared router
  pathway without exposing source metadata to routing;
- ambiguous rows omit dialect loss or use an adjudicated soft/set target.

### 6.4 Core novelty versus supporting engineering

Core paper method:

- task-conditioned late routing;
- clean-train causal dialect-evidence warm-start;
- dialect-relevant, source-adversarial router representation;
- layer-wise causal routing analysis.

Supporting engineering, not claimed as novel:

- shared/fine-grained experts;
- auxiliary-loss-free dynamic balance bias;
- dense-to-MoE upcycling;
- MTP;
- Muon plus AdamW;
- GQA/RoPE/RMSNorm/SwiGLU.

This separation prevents a kitchen-sink novelty claim.

### 6.5 Function-preserving upcycling pilot

The dense FFN width 2,304 is divided into three 768-neuron virtual groups:

- shared expert initializes from group S;
- routed experts 0–3 initialize from group A with independent small noise;
- routed experts 4–7 initialize from group B with independent small noise.

The exact reconstruction requires complementary routing:

- select one routed expert from bank A and one from bank B;
- shared plus both routed banks approximately reconstruct the dense FFN;
- balance within each bank.

The validation-only recovery study uses the same seed, data order, 20M-token
budget, 200M-token scheduler horizon, and optimizer for five conditions:

- abrupt release after 4M tokens;
- unrestricted routing from token zero;
- random initialization;
- linear cross-bank penalty release from 4M to 20M tokens;
- permanent complementary-bank top-2 routing.

The selection guard is fixed before the two recovery candidates run: at most
0.5% transient BPC regression, no final BPC regression, then minimum final BPC.
Abrupt release, unrestricted transfer, scratch initialization, and fully
released annealing fail. Permanent complementary-bank routing is the only
eligible condition and is therefore fixed for the 200M-token M2 continuation
and inherited by M2/M3 task adaptation. This is structured group-limited
top-2 routing: one routed expert is selected within each complementary bank,
plus the always-active shared expert.

### 6.6 Classification-specialized branch

Boichitro-ID-BI:

- convert attention masks to bidirectional;
- add <mask> and run 5M to 10M train-only masked-next-token adaptation;
- use <cls> pooling;
- loss is classification cross-entropy plus supervised contrastive loss;
- same-dialect positives must preferentially come from different sources;
- source adversary remains active;
- fine-tune all layers for the main result and compare last-four-layer/expert
  tuning as an efficiency ablation.

The causal classifier remains the fair shared-backbone comparison.

## 7. Optimization

### 7.1 Parameter ownership

Muon:

- query, key, value, and output projection matrices;
- dense FFN matrices;
- shared-expert matrices;
- routed-expert matrices;

AdamW:

- token/tied output embedding;
- normalization scales;
- every bias;
- router projection and dynamic load bias;
- task and dialect embeddings;
- classification, contrastive, adversarial, and MTP heads.

Required unit tests:

- every trainable parameter belongs to exactly one optimizer;
- no overlap;
- no omission;
- parameter ownership is identical after checkpoint reload;
- both schedulers use the same global step.

### 7.2 Training step

1. Zero both optimizers.
2. Forward under bfloat16 autocast where supported.
3. Compute all active task losses.
4. Backpropagate once.
5. Unscale and check non-finite gradients.
6. Log gradient/update norm by parameter family.
7. Apply the frozen clipping policy.
8. Step Muon and AdamW once each.
9. Step the synchronized schedules once.
10. Update the non-gradient load-balance bias.

### 7.3 Pilot search

Muon peak learning rate:

- 0.01, 0.02, 0.03.

AdamW non-router peak learning rate:

- 2e-4, 3e-4, 5e-4.

Router/head peak learning rate:

- 1e-4, 2e-4, 3e-4.

Other:

- weight decay 0.05 or 0.10;
- warmup 2% or 5%;
- AdamW betas 0.9 and 0.95;
- Muon momentum 0.95 with Nesterov;
- clipping 1.0 versus adaptive group clipping;
- cosine decay to 10% peak.

Pilot search uses development loss and stability. The AdamW-only control uses a
validation-only short pilot over 5e-5, 1e-4, 2e-4, and 3e-4; it is not forced to
use Muon-tuned rates.

Mature-checkpoint continuation uses a separate validation-only guard. A first
10M-token dense continuation with a restarted Muon 0.02 / AdamW 3e-4 schedule
regressed held-out BPC from 1.2303 to 1.3624, so that run was stopped and
retained as a rejected diagnostic before any locked neural test access. Four
10M-token candidates (Muon 0.001, 0.002, 0.005, 0.010 with proportional AdamW
rates) share the full 200M scheduler horizon and no warmup restart. The minimum
final validation BPC is selected only among candidates with no final BPC
regression and at most 1% intermediate regression. M0, M1, M2, and the matched
upcycling-recovery pilots all consume that frozen selection report.

The separate upcycling report is also validation-only. It selects permanent
complementary-bank routing at an exact 20M-token endpoint BPC of 1.2287 versus
the dense-equivalent start of 1.2303. Fully released annealing reaches 1.2641
at its exact unrestricted endpoint and is rejected. No locked neural test was
accessed for either decision.

### 7.4 Losses

Normalization/multitask batch:

    L =
        L_target
        + lambda_cls × L_dialect
        + lambda_adv × L_source_GRL
        + lambda_mtp × L_MTP
        + lambda_router × L_router_stability

Initial main-scale candidates:

- lambda_cls: 0.10 or 0.20;
- lambda_adv maximum: 0.01 or 0.03, linearly ramped;
- lambda_mtp: 0.05 or 0.10;
- router z-loss: 1e-4 or 1e-3.

Load balance in the proposed model uses dynamic non-gradient expert biases.
A conventional auxiliary-balance loss is retained as a control.

Classification-specialized loss:

    L_ID =
        L_class
        + 0.05 or 0.10 × L_supervised_contrastive
        + lambda_adv × L_source_GRL

All final weights are selected using validation data and frozen before tests.

### 7.5 Group-robust sampling

Groups:

    dialect × source × real_or_synthetic

Rules:

- every observed train group remains auditable; very small groups are protected
  from unstable extreme weighting by the frozen uniform mixture and 10×
  GroupDRO ratio cap rather than being silently oversampled;
- regularized GroupDRO weights have pre-registered floors and ceilings;
- report effective sampled tokens per group;
- compare ordinary ERM and balanced sampling;
- do not let synthetic or tiny noisy groups dominate.

## 8. Training stages

### Gate D0: data

Pass only if:

- source adapters reproduce expected counts;
- all protected connected-component leakage is zero;
- licenses/citations are recorded;
- raw/clean/provenance hashes are frozen;
- human audit of every source-dialect stratum is complete;
- derived ZIP remains excluded from the default manifest.

### Stage T: tokenizer

- Execute Sections 5.2–5.3.
- Freeze tokenizer files and SHA-256.
- Regenerate embedding/output dimensions.

### Stage F: dense foundation

Recommended:

- 1.0B to 1.5B clean general Bangla tokens;
- context 1,024;
- document-aware packing;
- global token batch target 262,144, adjusted only after hardware benchmark;
- evaluation every 25M tokens;
- checkpoint every 50M tokens;
- 2% warmup;
- fixed token budget.

Minimum:

- 300M tokens;
- clearly label as minimum-compute study.

The current dense notebook checkpoint is a pilot baseline, not the foundation:
it trained on about 1.12M non-padding tokens per effective pass and strongly
overfit by the final epoch.

### Stage U: MoE upcycling and general continuation

- Upcycle layers 5–16 from the selected dense checkpoint.
- Continue for 200M to 300M general Bangla tokens.
- Global token batch target 131,072.
- Enable dynamic loss-free balance.
- Do not enable dialect or source heads during the initial divergence phase.
- Activate the proposed routing objectives only after stable expert use.
- Retain the validation-selected complementary-bank topology throughout Stage
  U and downstream M2/M3 adaptation; do not silently switch to unrestricted
  top-2 at an evaluation boundary.
- Evaluate every 10M tokens.
- Use the validation-selected mature-checkpoint learning rate with no warmup
  restart; keep the router/head rate separately fixed for newly initialized
  routing parameters.

Stability gate:

- no non-finite loss;
- no persistent single-expert collapse;
- zero dropped tokens in single-GPU runs;
- expert load CV and Gini within frozen pilot bounds;
- router entropy does not collapse;
- development BPC recovers the pre-upcycle dense checkpoint within the
  pre-registered continuation window.

### Stage A: dialect-adaptive continuation

Budget:

- 12M tokens initially;
- global token batch target 65,536;
- shorten rather than repeat authentic examples beyond five effective passes.

Mixture:

- 65% general Bangla replay;
- 20% authentic dialect source text;
- 10% aligned normalization sequences;
- 5% real Romanized/code-mixed robustness examples.

Enable:

- dialect head;
- source adversary;
- task-independent dialect continuation;
- group-robust sampling;
- 50% dialect-tag dropout on tasks where tags are legal.

### Stage S: supervised specialization

Maximum:

- eight effective passes through real aligned training pairs;
- maximum approximately 6M packed task tokens;
- global token batch target 16,384 to 32,768;
- validate every quarter effective pass;
- patience five validations after a minimum of two passes.

Task mixture:

- 55% source-blind normalization;
- 20% identification;
- 10% dialect CLM;
- 10% general Standard Bangla replay;
- 5% Romanized/noise robustness where real data exist.

Checkpoint selection:

- choose highest macro development chrF++;
- candidate must keep general replay BPC degradation at or below 5%;
- ties within 0.1 chrF++ select the lower-latency checkpoint;
- no test inspection.

### Stage ID: classification specialization

- Start from the frozen Stage A checkpoint.
- Train causal <cls> classifier for all fair backbone comparisons.
- Separately run Boichitro-ID-BI adaptation.
- Select by regional-only development macro-F1.
- ECE is the tie-breaker within 0.2 macro-F1.
- Maximum 20 epochs or fixed equivalent steps, with patience four.

## 9. Confirmatory model matrix

| ID | Model | Total/active target | Router | Robustness | Seeds |
|---|---|---|---|---|---:|
| M0 | active-matched dense | about 80M/80M | none | GroupDRO control | 3 |
| M1 | Switch top-1 MoE | matched active FLOPs | generic | none | 3 |
| M2 | shared complementary-bank top-2 MoE | about 165M/80M | hidden + loss-free balance | none | 3 |
| M3 | Boichitro-MoE | about 165M/80M | complementary-bank staged dialect/task router | source adversary + GroupDRO | 3 |
| M4 | total-parameter-matched dense | about 165M/165M | none | same data robustness | 3 if budget |

Main seeds:

- 1701
- 2903
- 4307

Paired design:

- M0–M3 start from the same frozen dense foundation family;
- within a seed, data order and example sampling keys are matched;
- router/expert initialization differs as required;
- adaptation seeds from one foundation do not count as independent foundation
  pretraining seeds;
- the paper states this limitation exactly.

If claiming full pretraining superiority rather than continuation superiority,
train three independent foundation seeds. Otherwise the claim is restricted to
upcycling and specialization from a shared foundation.

## 10. External and task baselines

Normalization:

- copy-source baseline;
- dictionary/character rewrite baseline;
- BanglaT5 or mT5 on Vashantor;
- mBART-50;
- NLLB-200 distilled 600M;
- M0 dense;
- M2 standard MoE;
- M3 proposed.

Identification:

- majority and stratified random;
- character TF-IDF linear SVM;
- word plus character TF-IDF logistic regression;
- BanglaBERT;
- XLM-R base;
- M0 causal <cls>;
- M2 causal <cls>;
- M3 causal <cls>;
- Boichitro-ID-BI.

Diagnostic:

- source-only classifier using metadata;
- text-length/script/punctuation-only classifier;
- wrong-column Vashantor result shown only as an audit demonstration.

External pretrained models are reported in a separate block from
from-scratch/custom-tokenizer compute-matched models.

## 11. Ablation program

### 11.1 Core routing factorial at pilot scale

Run all sixteen combinations of four binary factors on the pilot model:

- A: causal dialect-evidence prior off/on;
- B: dialect router head off/on;
- C: source adversary off/on;
- D: task-conditioned late routing off/on.

Common conditions:

- one pre-registered pilot seed;
- same checkpoint;
- same 5M dialect-adaptation tokens;
- same data order;
- no test access.

Analyze main effects and two-way interactions. This prevents an arbitrary
leave-one-out interpretation when components interact.

### 11.2 Confirmatory main-scale ablations

Three seeds:

- M3 minus causal dialect-evidence prior;
- M3 minus dialect head;
- M3 minus source adversary;
- M3 minus task-conditioned late routing;
- M3 with randomized dialect-evidence mapping negative control;
- M3 without GroupDRO.

### 11.3 Architecture ablations

Pilot first; confirm only pre-registered finalists:

- no shared expert;
- 4, 8, or 16 routed experts at matched active FFN compute;
- top-1 versus top-2;
- coarse versus fine-grained experts;
- auxiliary balance loss versus loss-free bias;
- all-layer versus early/middle/late structured routing;
- abrupt, unbanked, annealed, and permanent paired-bank upcycling;
- scratch versus upcycled.

### 11.4 Optimization ablations

- AdamW only versus Muon plus AdamW;
- MTP off versus one future token versus two;
- fixed token budget versus early stopping;
- ordinary clipping versus selected group clipping.

Report:

- tokens to a frozen validation-loss threshold;
- wall-clock time to threshold;
- final fixed-budget metric;
- loss spikes/non-finite runs;
- update-to-weight ratio by parameter family.

### 11.5 Data ablations

- real only;
- real plus audited synthetic;
- quality-weighted audited synthetic;
- with versus without general replay;
- row split versus connected-group split, audit only;
- BanglaDial excluded versus reconstructed;
- source-balanced versus ordinary sampling.

### 11.6 Tokenizer ablations

- selected tokenizer versus clean rebuilt existing tokenizer;
- selected vocabulary size versus adjacent sizes;
- no dialect quota;
- no source-purity penalty;
- no fallback;
- old notebook tokenizer as historical baseline.

### 11.7 Classification specialization ablations

- causal <cls>;
- bidirectional attention only;
- bidirectional plus masked-next-token adaptation;
- plus supervised contrastive learning;
- full fine-tuning versus last four layers and routed experts;
- source adversary off/on.

### 11.8 Promotion rule

Exploratory one-seed pilots do not become paper evidence. An exploratory effect
is promoted only when:

- direction is stable across at least two checkpoints;
- absolute development gain reaches the pre-registered smallest effect of
  interest;
- three-seed confirmation is run;
- the test remains unopened.

## 12. Decoding

Development-only grid:

- greedy;
- beam size 2, 4, or 6;
- length penalty 0.8, 1.0, or 1.2;
- no-repeat constraint off/on only if repetition errors justify the predeclared
  diagnostic.

Freeze one decoding configuration per model family before test evaluation.
Primary model comparisons also report greedy decoding so beam search cannot
hide model differences.

Normalization must preserve:

- named entities;
- numbers;
- negation;
- tense/aspect;
- speaker intent;
- code-mixed terms without unjustified translation.

## 13. Metrics

### 13.1 Primary normalization

- macro chrF++: primary;
- sacreBLEU with full signature;
- TER;
- character and word error rate;
- exact match for copy-required cases;
- named-entity/number preservation;
- semantic similarity as secondary;
- learned metric only after validating Bangla correlation on human judgments.

Report:

- per dialect;
- macro average;
- worst dialect;
- source and domain;
- Bengali-script versus Romanized;
- sentence-length and rarity bins.

### 13.2 Identification

- regional-only macro-F1;
- 13-class macro-F1;
- balanced accuracy;
- MCC;
- per-class precision/recall/F1;
- worst-dialect F1;
- ECE and Brier;
- normalized confusion matrix.

### 13.3 Language modeling

- BPC and bits per byte;
- same-tokenizer NLL/PPL;
- rare-token loss;
- worst-dialect BPC;
- calibration;
- general replay degradation.

### 13.4 Robustness

Perturbations:

- phonetic spelling variants;
- punctuation removal;
- digit/name substitution;
- keyboard noise;
- Bengali/Latin Romanization;
- code-mixing;
- short and long input;
- source-style stripping.

Plot performance versus severity. Perturbation generation rules and seeds are
frozen before test.

### 13.5 Efficiency

- exact total and active parameters;
- measured forward/backward FLOPs per token;
- tokens/second;
- peak VRAM;
- training GPU-hours;
- batch-1 and batched latency;
- energy estimate when measurable;
- KV-cache bytes.

MoE-specific:

- per-layer expert load;
- CV and Gini;
- minimum/maximum load;
- entropy;
- dropped-token rate;
- experts per sequence;
- communication overhead if distributed.

## 14. Statistical protocol

Main neural systems:

- three seeds;
- mean, standard deviation, and seed-level interval;
- hierarchical paired bootstrap over seed and semantic group, stratified by
  dialect;
- paired semantic-group randomization within dialect and seed for p-values;
- 10,000 bootstrap and 10,000 randomization resamples.

Primary comparison:

- M3 versus M2 on locked OOD macro chrF++;
- two-sided alpha 0.05;
- no multiplicity correction because there is one primary hypothesis.

Secondary confirmatory family:

- M3 versus M0 normalization;
- M3 versus M2 regional ID macro-F1;
- selected tokenizer versus best standard tokenizer;
- hybrid optimizer versus AdamW;
- Holm correction.

Classification:

- hierarchical paired bootstrap intervals and paired randomization tests for
  macro-F1;
- McNemar for paired correctness where meaningful;
- calibration interval by stratified bootstrap.

Human evaluation:

- paired item-cluster bootstrap intervals over blinded ratings;
- paired item-level randomization tests with Holm correction;
- quadratic-weighted inter-rater kappa for ordinal dimensions and unweighted
  kappa for hallucination flags;
- report ordinal-score and hallucination-rate effects with direction stated.

All tests report effect sizes and intervals, not only p-values.

## 15. Human evaluation

System comparison:

- M3 versus M2;
- optional M3 versus strongest external translation baseline.

Sample:

- at least 100 test items per normalization dialect;
- 800 items for the primary pairwise comparison;
- three qualified annotators per item;
- randomized, blinded, counterbalanced output order.

Questions:

- Which output better preserves meaning?
- Which output is better Standard Bangla?
- Which better preserves entities, numbers, and negation?
- Is either output unacceptable?

Ratings:

- pairwise A/B/tie;
- adequacy 1–5;
- Standard-Bangla fluency 1–5;
- residual dialect leakage;
- hallucination/error flag.

Report:

- annotator regional competence;
- compensation;
- consent and ethics;
- agreement;
- adjudication;
- exclusions;
- mixed-effects results.

## 16. Error analysis

Sample 300 normalization errors, stratified by dialect, source, and system.

Taxonomy:

- semantic omission;
- semantic addition/hallucination;
- wrong lexical normalization;
- under-normalization;
- over-normalization;
- morphology/tense/aspect;
- word order;
- negation;
- entity/number corruption;
- code-mixing;
- spelling/punctuation only;
- reference ambiguity;
- source-label or data error.

Two annotators label each case; a third adjudicates.

For identification:

- inspect high-confidence wrong cases;
- compare source-only and dialect models;
- test ambiguous/shared labels;
- examine length, script, template, and rare-token bins.

For routing:

- compare expert paths for parallel Standard/regional semantic groups;
- intervene on the most dialect-associated expert;
- suppress source-associated experts;
- run shuffled-prior negative control;
- measure output and loss change.

## 17. Routing analysis

Per layer:

- mutual information between expert and dialect;
- mutual information between expert and source;
- conditional expert-source association given dialect;
- dialect-pair expert overlap;
- task-pair expert overlap;
- shared-expert contribution;
- routing entropy;
- routing stability under meaning-preserving spelling noise.

Success condition:

- expert-dialect association increases where intended;
- conditional expert-source association decreases;
- source-OOD quality improves;
- the association is causally relevant under intervention.

High mutual information alone is not evidence of useful specialization.

## 18. Main figures

Figure 1, data provenance and leakage graph:

- source archives;
- canonical versus duplicate members;
- semantic components;
- protected splits;
- synthetic parent paths.

Output: figures/fig01_data_provenance.pdf

Figure 2, architecture:

- dense layers;
- early, middle, and late MoE blocks;
- shared and routed experts;
- task conditioning;
- dialect head and source GRL head;
- two-optimizer ownership.

Output: figures/fig02_boichitro_architecture.pdf

Figure 3, tokenizer parity:

- per-dialect fertility and BPC;
- rare-vocabulary fraction;
- token-cost Gini;
- finalist intervals.

Output: figures/fig03_tokenizer_parity.pdf

Figure 4, primary forest plot:

- M3 minus M2 chrF++ effect;
- per-dialect and macro;
- paired 95% intervals;
- OOD and IID panels.

Output: figures/fig04_primary_effects.pdf

Figure 5, model results:

- per-dialect normalization chrF++;
- corresponding classification F1;
- source/domain facets.

Output: figures/fig05_per_dialect_results.pdf

Figure 6, routing:

- layer × expert heatmaps;
- expert-dialect versus conditional expert-source information curves;
- shared-expert use.

Output: figures/fig06_routing_specialization.pdf

Figure 7, efficiency Pareto:

- chrF++ versus measured active FLOPs;
- marker size for total parameters;
- color for latency;
- dense, standard MoE, proposed, and external baselines.

Output: figures/fig07_efficiency_pareto.pdf

Figure 8, ablations:

- effect-size forest plot;
- three-seed main ablations;
- pilot factorial main effects in a separate panel.

Output: figures/fig08_ablation_effects.pdf

Figure 9, robustness:

- spelling, Romanization, and code-mixing severity curves;
- macro and worst-dialect quality.

Output: figures/fig09_robustness.pdf

Figure 10, human/error analysis:

- blinded pairwise win/tie/loss;
- major error category rates.

Output: figures/fig10_human_errors.pdf

Plot rules:

- colorblind-safe palette;
- semantic-group bootstrap intervals;
- no deceptive axis truncation;
- show individual seed points;
- same model colors everywhere;
- publish plotting data as CSV/Parquet.

## 19. Main tables

Table 1:

- dataset, version, license, source type, dialects, raw rows, retained rows,
  split, and excluded reasons.

Table 2:

- audit findings and affected row counts.

Table 3:

- tokenizer intrinsic, proxy, throughput, and parity metrics.

Table 4:

- exact architecture, total/active parameters, FLOPs, memory, and optimizer.

Table 5:

- main OOD normalization results.

Table 6:

- clean IID, LOSO, and OOD identification.

Table 7:

- confirmatory ablations.

Table 8:

- efficiency and training cost.

Table 9:

- human evaluation and agreement.

Appendix:

- full per-dialect/per-source results;
- all confusion matrices;
- all routing loads;
- full tokenizer candidates;
- pilot factorial;
- failed/unstable runs;
- hyperparameter search space and selected values.

## 20. Run selection and test firewall

Development score for normalization:

- macro chrF++;
- general replay BPC is a hard degradation guardrail, not a hidden weighted
  component.

Development score for identification:

- regional-only macro-F1;
- ECE tie-breaker.

Before opening tests, freeze:

- data manifest and hashes;
- tokenizer and hash;
- model code commit;
- all hyperparameters;
- seed list;
- checkpoints;
- decoding;
- metric versions;
- bootstrap script;
- figure templates;
- primary/secondary hypotheses.

Test policy:

- one scripted evaluation per frozen seed/checkpoint;
- a crashed evaluation may be rerun only from the same immutable artifacts;
- no model change after seeing test outputs;
- any later exploratory result is labeled post hoc.

## 21. Acceptance gates

Data gate:

- zero protected train connections;
- all local counts reproduced;
- licenses recorded;
- human source audit completed;
- derived archive absent from main manifest.

Tokenizer gate:

- no severe dialect token-cost disparity;
- rare vocabulary materially below the old 61.2% at count at most five;
- proxy is non-inferior globally;
- worst-dialect development BPC improves or is statistically tied at smaller
  vocabulary.

MoE gate:

- stable loss;
- no persistent collapse;
- no hidden token dropping;
- active FLOPs matched;
- upcycled quality recovered;
- router cannot be explained primarily by source.

Primary result gate:

- M3 − M2 macro OOD chrF++ paired interval above zero;
- no more than 1.0 chrF++ absolute IID loss;
- human adequacy win probability above 0.5 with interval;
- measured active compute comparable.

Classification gate:

- independent source-OOD macro-F1 improves;
- calibration does not materially degrade;
- source-only shortcuts are lower;
- ambiguous labels reported.

If a gate fails, report the negative result and shift the paper claim toward
the audited benchmark and routing diagnosis. Q1 quality means a defensible
result, not forcing a positive one.

## 22. Compute control

Before any long run:

- benchmark 1,000 stable steps;
- measure tokens/second, peak memory, and kernel utilization;
- project hours for every registered run;
- reserve 15% compute for failed-run recovery;
- prune only by pre-registered development rules.

Recommended order:

1. CPU data audit and adapters.
2. Tokenizer intrinsic screen.
3. 28M proxy tokenizer runs.
4. 40M-active architecture/optimizer pilots.
5. One dense foundation.
6. Standard and proposed MoE continuation.
7. Core three-seed specialization.
8. Confirmatory ablations.
9. Locked test.
10. Human and routing analyses.

Recommended Q1 tier:

- 1.0B to 1.5B dense foundation tokens;
- 200M to 300M MoE continuation tokens;
- M0–M3 three-seed specialization;
- external and human OOD evaluation;
- estimated 300–600 96GB-GPU hours only as a planning range.

Measured microbenchmarks replace this estimate.

## 23. Artifact contract

Required immutable artifacts:

- archives.sha256;
- source manifest;
- canonical rows with provenance;
- excluded_rows with reason codes;
- split manifest;
- contamination report;
- tokenizer corpus manifest;
- tokenizer files and card;
- config per run;
- environment lock;
- code commit;
- logs and checkpoints;
- per-example predictions;
- per-example metric contributions;
- fixed routing-trace subset;
- bootstrap samples/seeds;
- plotting data;
- model card, data statement, ethics, and limitations.

Naming:

    runs/<run_id>/<seed>/
    predictions/<protocol>/<run_id>_<seed>.parquet
    routing/<run_id>_<seed>_<analysis_set>.parquet
    metrics/<protocol>/<run_id>_<seed>.json
    figures/data/<figure_id>.parquet

No final table is manually copied from notebook output.

## 24. Execution sequence

### Phase 1: local foundation

1. Keep both original ZIP files immutable.
2. Run audit_local_archives.py and freeze its JSON.
3. Implement explicit source adapters.
4. Emit canonical row and pair manifests.
5. Reconstruct BanglaDial component sources.
6. Build connected components.
7. Quarantine all protected/ambiguous/derived rows.
8. Freeze N-IID, N-LOSO, ID-IID, and ID-LOSO.
9. Resolve dataset licenses.

### Phase 2: additional evidence

10. Acquire and audit general Bangla pretraining data.
11. Acquire only original external dialect releases, never an untracked merge.
12. Cross-deduplicate all candidate benchmarks.
13. Collect the eight-dialect OOD human set.
14. Freeze the test firewall.

### Phase 3: modeling

15. Run tokenizer screen and freeze.
16. Train dense foundation.
17. Run upcycling/optimizer pilots.
18. Train M0–M3.
19. Specialize normalization and ID branches.
20. Run confirmatory ablations.

### Phase 4: evidence and paper

21. Evaluate locked tests.
22. Complete human evaluation.
23. Run routing interventions.
24. Generate every figure/table from saved artifacts.
25. Perform an independent reproduction run.
26. Release code, hashes, cards, exclusions, and limitations.

## 25. Paper structure

1. Introduction: source robustness, dialect normalization, and compact sparse
   modeling.
2. Related work: Bangla dialect resources, SLMs, MoE routing, tokenizer
   fairness, and source artifacts.
3. Local data audit and benchmark protocol.
4. Boichitro-MoE.
5. Training and task specialization.
6. Experimental setup and baselines.
7. Main normalization and identification results.
8. Routing, robustness, and efficiency.
9. Human/error analysis.
10. Limitations, ethics, dialect/language identity, and release.

## 26. Final decision

The full experiment is centered on the user-provided ZIP data:

- 22,364 local aligned rows support eight-dialect normalization;
- the reconstructed local pool supports conditional 13-class identification;
- the 51,101 derived rows are an explicit ablation, not silent training data;
- only general Bangla pretraining and independent OOD evaluation require
  external collection.

The proposed SLM is not merely a dense notebook with MoE added. It is a
compute-matched, task-specialized model whose router is tested for dialect
knowledge, source invariance, causal relevance, robustness, and efficiency
under a locked statistical protocol.
