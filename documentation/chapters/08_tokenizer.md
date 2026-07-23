# Tokenizer study and immutable freeze

Tokenizer selection compares WordPiece, Unigram, and byte-BPE families at multiple vocabulary sizes and corpus-balance conditions. Intrinsic measures include tokens per character and byte, fertility, unknown rate, round-trip behavior, dialect dispersion, worst-to-best ratios, and Gini-style fairness summaries.

Candidates passing the intrinsic screen enter a fixed-budget proxy language-model study with seeds 1701, 2903, and 4307. Bits per character is the cross-tokenizer quality metric because token-normalized perplexity is not comparable across vocabularies. Throughput and stability provide additional practical evidence.

The selected tokenizer is `wordpiece_natural_32k`, with an actual vocabulary of 32,000. Its frozen files, metadata, and recorded hash are copied into this documentation package. Test data were not used for selection.
