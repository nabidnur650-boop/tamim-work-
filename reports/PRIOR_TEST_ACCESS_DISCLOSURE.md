# Prior test-access disclosure

The final neural evaluation protocol is mechanically frozen before any
`locked_test_v1` or `locked_external_test_v1` model manifest can be produced.
However, this workspace is **not claimed to have a pristine, never-read test
set**:

- fixed classical copy/rewrite/SVM baselines were scored on the designated
  IID/OOD tracks before the neural protocol freeze;
- the legacy rewrite used gold dialect metadata and is therefore retained only
  as an oracle; the corrected source-blind rewrite and fused neural systems are
  evaluated by the locked script after freeze;
- a tiny end-to-end pipeline smoke test read a small deterministic prefix under
  the separate `smoke_locked_test` namespace;
- an automated aggregate-only firewall compared normalized evaluation inputs
  with the frozen identification-training input set, emitted no protected text,
  and found zero exact, compact, or registered SimHash/Jaccard near-duplicate
  overlaps; it was not used for model or hyperparameter selection;
- no M0–M3, M3B, or fine-tuned external neural checkpoint has a main locked
  metric at this point;
- prior classical/smoke outputs are not used for neural hyperparameter,
  checkpoint, ablation, or decoding selection.

`tools/freeze_protocol.py` records this disclosure, refuses to freeze if main
locked neural manifests already exist, fingerprints all selected development
artifacts, and makes the main evaluators reject post-freeze code, config, or
checkpoint changes.

For publication, this limitation must remain in the experimental protocol. A
claim of a fully untouched test set requires a newly collected and independently
held native-reviewed evaluation set; deleting the existing outputs would not
undo prior access.
