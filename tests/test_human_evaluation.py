from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "analyze_human_evaluation", PROJECT / "tools/analyze_human_evaluation.py"
)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class HumanEvaluationAnalysisTests(unittest.TestCase):
    def test_randomization_respects_registered_effect_direction(self) -> None:
        positive = np.ones(24)
        _, p_higher, p_two_positive = analysis.paired_sign_randomization(
            positive,
            replicates=4096,
            rng=np.random.default_rng(17),
            higher_is_better=True,
        )
        negative = -positive
        _, p_lower, p_two_negative = analysis.paired_sign_randomization(
            negative,
            replicates=4096,
            rng=np.random.default_rng(17),
            higher_is_better=False,
        )
        self.assertLess(p_higher, 0.01)
        self.assertLess(p_lower, 0.01)
        self.assertEqual(p_two_positive, p_two_negative)

    def test_randomization_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            analysis.paired_sign_randomization(
                np.asarray([]),
                replicates=10,
                rng=np.random.default_rng(1),
                higher_is_better=True,
            )


if __name__ == "__main__":
    unittest.main()
