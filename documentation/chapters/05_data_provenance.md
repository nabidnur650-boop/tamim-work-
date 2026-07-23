# Data provenance, taxonomy, and canonical schema

The frozen taxonomy has thirteen labels: twelve regional varieties plus Standard Bangla. Labels are BAR, CHI, KHU, KIS, MYM, NAR, NOA, NSD, RAJ, RAN, SYL, TAN, and STD. Not every source supports every label, and the documentation preserves those missing cells rather than imputing coverage.

Canonical records include immutable row identifiers, task, source/provider, source version, source row, license record, dialect code, regional input, Standard-Bangla target when available, normalized text views, semantic component, original split provenance, frozen split, evaluation track, quality tier, synthetic flag, and ancestry.

Source adapters separate local Vashantor, regional corpora, Chatgaiyya material, Sylheti translation, pinned public datasets, external transcripts, romanized pairs, and general-language pretraining text. The data artifact ledger in this package records row counts, byte sizes, and published SHA-256 values for every frozen dataset output.
