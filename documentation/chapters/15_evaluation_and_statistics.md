# Evaluation tracks, metrics, and statistical protocol

Normalization reports chrF++, BLEU, TER, character error rate, exact match, worst-dialect performance, and replay retention. Identification reports accuracy, balanced accuracy, thirteen-class and regional macro-F1, MCC, ECE-15, Brier score, and worst-present-dialect F1.

Tracks separate validation, group-IID test, source-held-out test, external transcript, and romanized challenge material. RAJ normalization is a zero-shot challenge and is not silently averaged into the trained-dialect endpoint.

Confirmatory inference is designed around paired per-example predictions. Semantic groups define the resampling unit. Hierarchical paired bootstrap intervals, semantic-group paired randomization, and Holm correction within registered endpoint families control dependence and multiplicity. Those locked confirmatory outputs have not been generated.
