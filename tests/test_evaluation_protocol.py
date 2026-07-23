from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


class EvaluationProtocolTests(unittest.TestCase):
    def test_supported_ood_and_zero_shot_raj_are_disjoint_and_exhaustive(self) -> None:
        config = yaml.safe_load(
            (PROJECT / "configs/locked_evaluation.yaml").read_text(encoding="utf-8")
        )
        tracks = config["normalization"]["tracks"]
        frame = pd.read_parquet(PROJECT / tracks["source_ood"]["frame"])
        supported = frame.loc[
            ~frame["dialect"].isin(tracks["source_ood"]["exclude_dialects"])
        ]
        zero_shot = frame.loc[
            frame["dialect"].isin(tracks["zero_shot_raj"]["include_dialects"])
        ]
        self.assertEqual(
            set(supported["dialect"]), {"BAR", "CHI", "MYM", "NOA", "SYL"}
        )
        self.assertEqual(set(zero_shot["dialect"]), {"RAJ"})
        self.assertTrue(set(supported["row_id"]).isdisjoint(set(zero_shot["row_id"])))
        self.assertEqual(len(supported) + len(zero_shot), len(frame))

    def test_iid_covers_exactly_eight_trained_normalization_dialects(self) -> None:
        frame = pd.read_parquet(PROJECT / "data/final/v1/normalization_test_iid.parquet")
        self.assertEqual(
            set(frame["dialect"]),
            {"BAR", "CHI", "KHU", "MYM", "NAR", "NOA", "RAN", "SYL"},
        )


if __name__ == "__main__":
    unittest.main()
