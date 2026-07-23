# General Bangla pretraining corpus and dense foundation

The general-language source is a revision-pinned Bengali subset of FineWeb-2. Source verification records immutable revisions and file hashes. Filtering applies Unicode checks, Bengali-script ratios, length constraints, and direct/compact benchmark decontamination.

The recorded corpus report examined 585,619 documents and accepted 567,746. The fixed foundation budget contains 300,004,991 tokens, with a disjoint 3,000,629-token validation allocation. Packed block order and shard provenance are immutable.

The dense foundation uses sixteen decoder layers, width 512, grouped-query attention with eight query heads and two key/value heads, RoPE, QK normalization, RMSNorm, multi-token prediction, and the frozen custom tokenizer. The fixed-budget foundation run is single-seed developmental evidence.
