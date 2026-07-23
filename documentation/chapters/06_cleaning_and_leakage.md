# Cleaning, deduplication, and leakage controls

Cleaning normalizes Unicode and whitespace while preserving linguistically meaningful content. Fatal text-quality failures are excluded with explicit reasons. The pipeline applies exact pair checks, compact text checks, cross-label conflict removal, source-priority rules, and protected-evaluation ancestry controls.

Near-duplicate protection combines 64-bit SimHash with character 4-gram Jaccard filtering. Candidate pairs within a maximum Hamming distance are checked against a 0.90 Jaccard threshold. Semantic relations are converted into connected components before split assignment, preventing related rows from straddling development and protected evaluation.

The direction of removal is asymmetric by design: training relatives are removed against protected IID evaluation; source-OOD material is protected against all development inputs. A separate cross-task firewall checks that normalization evaluation inputs do not appear among identification-training inputs. Synthetic perturbations are train-only and never constitute evaluation evidence.
