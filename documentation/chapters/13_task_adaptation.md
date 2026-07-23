# Task adaptation, replay retention, and identification

Task adaptation uses three seeds. Stage A consumes a fixed 12M-token mixture of general replay, dialect language modeling, normalization, and romanized material. Stage S consumes 6M tokens and selects normalization checkpoints only when general-language replay degradation remains at or below the preregistered 5% guard.

The selected `ret35_balanced` schedule allocates 30% normalization and 35% replay. It achieved 41.186 validation macro chrF++ in its pilot with 0.972% replay-NLL degradation. A higher-scoring default schedule was rejected because it incurred 15.91% degradation. This is a clear example of a protocol constraint overriding a superficially better task score.

Causal identification trains with early stopping and post-hoc temperature calibration. The separate bidirectional branch applies masked-next-token prediction and contrastive supervision before identification specialization. Two of its three seed manifests are complete at this snapshot.
