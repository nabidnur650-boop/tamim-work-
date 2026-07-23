from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from boichitro.robustness import perturb_text, stable_perturbation_seed  # noqa: E402


class RobustnessTests(unittest.TestCase):
    def test_perturbations_are_deterministic_and_grapheme_aware(self) -> None:
        text = "আমি ভাত খাব"
        for family in ("grapheme_deletion", "adjacent_swap", "whitespace_noise"):
            seed = stable_perturbation_seed("row", family, 0.2, 123)
            first = perturb_text(text, family=family, severity=0.2, seed=seed)
            second = perturb_text(text, family=family, severity=0.2, seed=seed)
            self.assertEqual(first, second)
            self.assertNotEqual(first, text)

    def test_zero_severity_is_identity(self) -> None:
        self.assertEqual(
            perturb_text("বাংলা", family="grapheme_deletion", severity=0.0, seed=1),
            "বাংলা",
        )


if __name__ == "__main__":
    unittest.main()
