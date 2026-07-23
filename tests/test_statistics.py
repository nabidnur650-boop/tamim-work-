from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "run_statistics", PROJECT / "tools/run_statistics.py"
)
assert SPEC is not None and SPEC.loader is not None
statistics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(statistics)


class StatisticsTests(unittest.TestCase):
    def test_precomputed_chrf_statistics_are_exact(self) -> None:
        treatment = pd.DataFrame(
            {
                "row_id": ["a", "b", "c", "d"],
                "semantic_group_id": ["g1", "g2", "g3", "g4"],
                "dialect": ["BAR", "BAR", "SYL", "SYL"],
                "reference": ["আমি যাব", "আমি খাব", "সে যাবে", "তুমি খাবে"],
                "prediction": ["আমি যাব", "আমি খাব", "সে যায়", "তুমি খাবে"],
            }
        )
        control = treatment.copy()
        control["prediction"] = ["আমি যাই", "আমি খাই", "সে যায়", "তুমি খাও"]
        aligned = statistics.align_pair(treatment, control, "normalization")
        prepared = statistics.prepare_group_statistics(aligned, "normalization")
        cached = statistics.score_group_statistics(prepared)
        exact = (
            statistics.macro_chrf(aligned, "prediction_treatment"),
            statistics.macro_chrf(aligned, "prediction_control"),
        )
        self.assertEqual(cached, exact)

    def test_confusion_sufficient_statistics_are_exact(self) -> None:
        treatment = pd.DataFrame(
            {
                "row_id": ["a", "b", "c", "d"],
                "semantic_group_id": ["g1", "g2", "g3", "g4"],
                "dialect": ["BAR", "BAR", "CHI", "CHI"],
                "label_id": [0, 0, 1, 1],
                "prediction_id": [0, 1, 1, 1],
            }
        )
        control = treatment.copy()
        control["prediction_id"] = [0, 0, 0, 1]
        aligned = statistics.align_pair(treatment, control, "identification")
        prepared = statistics.prepare_group_statistics(aligned, "identification")
        cached = statistics.score_group_statistics(prepared)
        exact = (
            statistics.regional_f1(aligned, "prediction_treatment"),
            statistics.regional_f1(aligned, "prediction_control"),
        )
        self.assertEqual(cached, exact)

    def test_paired_randomization_is_exactly_null_for_identical_predictions(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": ["a", "b", "c", "d"],
                "semantic_group_id": ["g1", "g2", "g3", "g4"],
                "dialect": ["BAR", "BAR", "CHI", "CHI"],
                "label_id": [0, 0, 1, 1],
                "prediction_treatment": [0, 1, 1, 0],
                "prediction_control": [0, 1, 1, 0],
            }
        )
        draws, observed = statistics.paired_randomization(
            {1701: frame}, task="identification", replicates=64, seed=19
        )
        self.assertEqual(observed, 0.0)
        self.assertTrue(draws["null_delta"].eq(0.0).all())

    def test_repeated_runs_must_share_identical_semantic_groups(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": ["a", "b", "c", "d"],
                "semantic_group_id": ["g1", "g2", "g3", "g4"],
                "dialect": ["BAR", "BAR", "CHI", "CHI"],
                "label_id": [0, 0, 1, 1],
                "prediction_treatment": [0, 1, 1, 0],
                "prediction_control": [0, 0, 1, 1],
            }
        )
        prepared = {
            1701: statistics.prepare_group_statistics(frame, "identification"),
            2903: statistics.prepare_group_statistics(
                frame.loc[frame["row_id"].ne("d")], "identification"
            ),
        }
        with self.assertRaisesRegex(ValueError, "different semantic groups"):
            statistics.shared_group_contract(prepared)

    def test_cross_dialect_group_draws_are_synchronized(self) -> None:
        shared = {
            "BAR": np.asarray(["g1", "g2"], dtype=object),
            "CHI": np.asarray(["g1", "g3"], dtype=object),
        }
        sampled = statistics.synchronized_bootstrap_groups(
            shared, np.random.default_rng(71)
        )
        self.assertEqual(
            int(np.count_nonzero(sampled["BAR"] == "g1")),
            int(np.count_nonzero(sampled["CHI"] == "g1")),
        )
        swaps = statistics.synchronized_randomization_swaps(
            shared, np.random.default_rng(73)
        )
        self.assertEqual(bool(swaps["BAR"][0]), bool(swaps["CHI"][0]))


if __name__ == "__main__":
    unittest.main()
