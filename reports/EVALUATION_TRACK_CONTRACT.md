# Frozen evaluation-track contract

## Normalization

- `iid`: all eight trained normalization dialects (BAR, CHI, KHU, MYM, NAR,
  NOA, RAN, SYL); primary broad-coverage endpoint.
- `source_ood`: independent-source rows for the five trained dialects with an
  admissible external source (BAR, CHI, MYM, NOA, SYL); primary transfer
  endpoint. Macro chrF++ is averaged over these five dialects only.
- `zero_shot_raj`: the 1,387 RAJ rows from BD-Dialect. RAJ has no normalization
  training or validation pair, so this is a separately labelled zero-shot
  challenge and is never folded into the supported-dialect source-OOD mean.
- `romanized_ood`: the independently frozen CHI/SYL romanized challenge.

KHU, NAR, and RAN do not have a defensible independent-source normalization
set in the admitted data. Their IID results remain reported; no eight-dialect
source-OOD claim is made.

Every deployable normalization system receives source text only. Gold dialect,
source identity, evaluation track, and reference text are forbidden inference
features. The fair word-rewrite control first infers among the eight dialects
represented in its training-only lexicon. The legacy gold-dialect rewrite is
reported only as an oracle and is never a confirmatory comparator. Raw neural
and fixed validation-selected fusion outputs are stored as separate views.
Normalization-selector cross-validation holds out whole semantic groups, so
cross-dialect realizations, shared targets, and every repeated model output for
related content stay in one fold. Development intervals condition on the
selected family and threshold and are exploratory; confirmatory inference is
reserved for the post-freeze locked evaluation. Bootstrap multiplicities and
randomization swaps are assigned once per global semantic group and reused
across seeds, architectures, and cross-dialect realizations.

## Identification

- `iid`: held-out component-disjoint examples over the supported 13-label
  taxonomy.
- `source_ood`: only labels present in the independent-source test pool;
  absent labels are not imputed as zero-support classes.
- `external`: independently sourced regional transcript challenge.

All selection uses validation data. Main neural test evaluation is permitted
only after the immutable `locked_test_v1` protocol freeze.

The neural/SVM probability blend is a separate validation-selected system
view. Raw neural identification remains primary for architecture and
source-OOD inference because no source-OOD row is used to select the blend.
The development blend interval conditions on the validation-selected weight
and is not reported as a confirmatory p-value.
