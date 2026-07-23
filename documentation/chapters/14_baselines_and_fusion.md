# Baselines and source-blind development fusion

Normalization baselines include identity copying and a training-only word-rewrite system. The fair rewrite infers a supported dialect from source text. A legacy gold-dialect rewrite is retained strictly as an oracle diagnostic and is not a deployable comparator.

Identification controls include character TF–IDF SVM and SGD systems. These are strong IID baselines but collapse on source-OOD material, motivating source-robust modeling. External model baselines are pinned in configuration but have no completed locked manifests.

A fixed source-blind candidate selector and neural/SVM probability blend were selected on development data. They use source text, candidate outputs, and dialect probabilities inferred from source text; references, gold dialects, source IDs, and evaluation-track labels are forbidden inference features. The fixed fusion transfers without retuning across all M2/M3 development runs, but remains exploratory until locked confirmation.
