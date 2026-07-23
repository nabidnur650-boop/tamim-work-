from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_development_results",
    PROJECT / "tools/summarize_development_results.py",
)
assert SPEC and SPEC.loader
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class DevelopmentSummaryTests(unittest.TestCase):
    def test_selected_checkpoint_details_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            root = project / "runs/task/boichitro_q1_v1/M0__base/1701"
            (root / "stage_s").mkdir(parents=True)
            (root / "stage_id").mkdir(parents=True)
            (root / "stage_s/best_selection.json").write_text(
                json.dumps(
                    {
                        "status": "SELECTED_ON_VALIDATION",
                        "global_step": 39,
                        "validation": {
                            "macro_chrfpp": 40.5,
                            "worst_dialect_chrfpp": 30.0,
                            "replay_nll": 3.5,
                            "replay_relative_degradation": 0.012,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "stage_id/best_selection.json").write_text(
                json.dumps(
                    {
                        "status": "SELECTED_ON_VALIDATION",
                        "epoch": 2,
                        "validation": {
                            "accuracy": 0.8,
                            "balanced_accuracy": 0.7,
                            "macro_f1_13": 0.72,
                            "regional_macro_f1": 0.71,
                            "mcc": 0.75,
                            "ece_15": 0.08,
                            "brier": 0.25,
                            "worst_present_dialect_f1": 0.4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [{"dialect": "BAR", "rows": 3, "chrfpp": 40.5}]
            ).to_csv(
                root / "stage_s/validation_by_dialect_epoch_39.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "label_id": 0,
                        "dialect": "BAR",
                        "precision": 0.8,
                        "recall": 0.7,
                        "f1": 0.75,
                        "support": 3,
                    }
                ]
            ).to_csv(
                root / "stage_id/validation_by_class_epoch_02.csv", index=False
            )

            main, dialect, classes = summary.collect_development_rows(project)
            self.assertEqual(len(main), 1)
            self.assertEqual(main.iloc[0]["variant"], "M0")
            self.assertAlmostEqual(main.iloc[0]["replay_degradation_percent"], 1.2)
            self.assertEqual(dialect.iloc[0]["selected_step"], 39)
            self.assertEqual(classes.iloc[0]["selected_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
