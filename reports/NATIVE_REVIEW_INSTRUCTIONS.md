# Native review instructions for the frozen data audit

Qualified native speakers should complete `HUMAN_NATIVE_REVIEW_SAMPLE.csv`
without changing the row IDs, source text, target text, dialect, or split.
Reviewers must enter their own identifier and independently score each assigned
row:

- `dialect_authenticity_1_to_5`: 1 = not credible for the stated variety;
  5 = clearly authentic/natural for that variety.
- `target_adequacy_1_to_5`: 1 = meaning is lost or changed; 5 = all meaning is
  preserved in the target.
- `target_fluency_1_to_5`: 1 = unacceptable Standard Bangla; 5 = fully natural.
- `label_correct_yes_no`: whether the stated dialect label is defensible.
- `unsafe_or_pii_yes_no`: whether the row contains unsafe content or personal
  information that prevents release.

Use `review_notes` for uncertainty, correction proposals, identity-sensitive
label concerns, or suspected contamination. Do not copy ratings from another
reviewer. The preregistered thresholds are in `configs/native_review.yaml`.

After all 230 rows are complete, run:

    PYTHONPATH=src python tools/analyze_native_review.py

The analyzer never edits the frozen dataset. Failure requires a reviewed v2 and,
if affected rows entered training/evaluation, rerunning the corresponding model
protocol; deleting or editing v1 in place is forbidden.
