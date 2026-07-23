# Frozen dataset composition and data gates

The final build contains 57,923 normalization rows, of which 54,598 are authentic and 3,325 are traceable train-only perturbations. The identification view contains 122,353 conflict-cleaned rows. The real romanized source-held-out track contains 1,342 rows. The tokenizer text view contains 100,236 unique texts.

The dataset report records 137,386 exclusion decisions. Large categories include same-label compact duplicates, cross-label identification conflicts, lower-priority exact normalization pairs, protected-evaluation relatives, source taxonomy exclusions, and quality failures. The full exclusion ledger is evidence of conservative filtering, not additional training data.

Automated gates pass for license resolution within the internal task scope, taxonomy, quarantine of the legacy derived archive, source-OOD protection, synthetic train-only use, compact-overlap checks, and cross-label conflict removal. The native-review gate fails because 0/230 sampled rows have been completed. Training is authorized internally; public redistribution and “linguistically validated” claims are not.
