# External source decisions

Frozen: 2026-07-19. The machine-readable acquisition record is
`reports/external_source_acquisition.json`; every admitted file is pinned to a
Mendeley version or Hugging Face commit and verified by SHA-256.

## Admitted sources

| Source | Frozen release | Final role | License decision |
|---|---|---|---|
| Kothon | [10.17632/2fv6vf9v2z.4](https://data.mendeley.com/datasets/2fv6vf9v2z/4) | CHI/SYL authentic train-development after global dedup | CC BY 4.0 |
| Sylheti translation | [10.17632/5rmskrvh6g.3](https://data.mendeley.com/datasets/5rmskrvh6g/3) | Novel remainder; word rows train-only | CC BY 4.0 |
| ChattogramSent | [10.17632/k6hts2ktxw.2](https://data.mendeley.com/datasets/k6hts2ktxw/2) | Novel remainder; never claimed independent OOD | CC BY 4.0 |
| ONUBAD | [10.17632/6ft99kf89b.2](https://data.mendeley.com/datasets/6ft99kf89b/2) | Entire-source locked BAR/CHI/SYL sentence OOD | CC BY 4.0 |
| BD-Dialect | [10.17632/k769s4vk5z.2](https://data.mendeley.com/datasets/k769s4vk5z/2) | Entire-source locked five-variety phrase/lexical OOD | CC BY 4.0 |
| BhasaBodh | [10.17632/2jb4k7bb8x.1](https://data.mendeley.com/datasets/2jb4k7bb8x/1) | Romanized companion to accepted ONUBAD rows | CC BY 4.0 |
| BanglaDial | [10.17632/sx6ybcps2n.2](https://data.mendeley.com/datasets/sx6ybcps2n/2) | Lower-confidence merged-provenance ID pool | CC BY 4.0 |
| Regional ASR text | [sha1779/bengali_regional_dataset_refine](https://huggingface.co/datasets/sha1779/bengali_regional_dataset_refine/tree/6e3221bdf7f9c8c426276f1b619d7de597963f42) | Exact-name mapped authentic identification transcripts | Apache-2.0 card tag; upstream terms caveat |
| Chittagonian safety lexicon | [kit-nlp release](https://huggingface.co/datasets/kit-nlp/Vulgar_Lexicon_of_Chittagonian_Dialect_of_Bangla_or_Bengali/tree/ce4be3814b8eca1a4c0fa85f983625565f887088) | Tokenizer/safety auxiliary only | Apache-2.0 |

The local Vashantor, ChatgaiyyaAlap, Sylheti v2, and Bangla Regional Text
Corpus releases were also resolved to their versioned CC BY 4.0 repository
records. The local Sylheti v2 and Chatgaiyya CSV bytes exactly match the
repository SHA-256 values.

## Lineage decisions

- BhasaBodh's 980 Bengali-script CHI/SYL pairs reproduce ONUBAD sentence
  content and add romanization. They are one shared component, not two
  independent benchmarks.
- ChattogramSent contains 4,014 exact local ChatgaiyyaAlap pairs. Global source
  priority removes those ancestors; only the novel remainder can survive.
- Sylheti v3 contains 1,212 exact pairs from the local v2 file. The older local
  rows win and are not counted twice.
- BanglaDial v2 improves the released merged corpus, but component-level source
  provenance remains incomplete; it is therefore a Tier-B identification
  source, not the locked OOD benchmark.
- ONUBAD and BD-Dialect rows are removed from OOD if either side is exact/near
  development content. Development data is never deleted to make an OOD set
  look larger.

## Hugging Face mapping rule

The ASR release contains ten district names. Eight map by exact string identity
to the frozen taxonomy: Barishal, Chittagong, Kishoreganj, Narail, Narsingdi,
Rangpur, Sylhet, and Tangail. Habiganj (940 rows) and Sandwip (1,049 rows) are
audited exclusions. Neither is collapsed into a nearby label.

## Rejected Hugging Face candidates

| Repository | Reason |
|---|---|
| TanjimKIT duplicate vulgar lexicon | Exact data-blob duplicate of the canonical `kit-nlp` release |
| TanjimKIT CBDCB | Gated and narrowly cyberbullying-specific; unnecessary for the frozen core tasks |
| Regional-TinyStories 2M | 4.76 GB fully synthetic release with no card or license |
| sha1779/Bengali_Regional_dataset | No card/license and superseded by the pinned refined release |

No rejected dataset contributes a final task row.
