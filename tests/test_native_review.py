from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "analyze_native_review", PROJECT / "tools/analyze_native_review.py"
)
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class NativeReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load(
            (PROJECT / "configs/native_review.yaml").read_text(encoding="utf-8")
        )
        self.frame = pd.DataFrame(
            {
                "row_id": ["a", "b", "c", "d"],
                "source_id": ["s1", "s2", "s1", "s2"],
                "dialect": ["BAR", "BAR", "CHI", "CHI"],
                "reviewer_id": ["r1", "r2", "r3", "r1"],
                "dialect_authenticity_1_to_5": [5, 5, 5, 5],
                "target_adequacy_1_to_5": [5, 5, 5, 5],
                "target_fluency_1_to_5": [5, 5, 5, 5],
                "label_correct_yes_no": ["yes"] * 4,
                "unsafe_or_pii_yes_no": ["no"] * 4,
            }
        )

    def test_complete_high_quality_review_passes(self) -> None:
        report, by_dialect = review.analyze(self.frame, self.config)
        self.assertEqual(report["status"], "PASS_NATIVE_REVIEW")
        self.assertEqual(set(by_dialect["dialect"]), {"BAR", "CHI"})

    def test_incomplete_review_stays_pending(self) -> None:
        self.frame.loc[0, "reviewer_id"] = ""
        report, by_dialect = review.analyze(self.frame, self.config)
        self.assertEqual(report["status"], "PENDING_INCOMPLETE")
        self.assertTrue(by_dialect.empty)


if __name__ == "__main__":
    unittest.main()
