from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from boichitro.metrics import classification_metrics, normalization_metrics  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_perfect_classification(self) -> None:
        labels = [0, 1, 2, 0]
        probabilities = np.eye(13)[labels] * 0.99 + 0.01 / 13
        predictions = probabilities.argmax(axis=1)
        metrics, _, _ = classification_metrics(labels, predictions, probabilities)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["regional_macro_f1"], 1.0)

    def test_perfect_normalization(self) -> None:
        frame = pd.DataFrame(
            {
                "prediction": ["আমি যাব", "তুমি আসবে"],
                "reference": ["আমি যাব", "তুমি আসবে"],
                "dialect": ["BAR", "CHI"],
            }
        )
        metrics, _ = normalization_metrics(frame)
        self.assertAlmostEqual(metrics["macro_chrfpp"], 100.0)
        self.assertEqual(metrics["exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
