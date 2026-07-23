from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "run_task_experiments", PROJECT / "tools/run_task_experiments.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class TaskRunnerProtocolTests(unittest.TestCase):
    def test_main_stage_s_matches_completed_validation_selection(self) -> None:
        import yaml

        config = yaml.safe_load(
            (PROJECT / "configs/task_experiments.yaml").read_text(encoding="utf-8")
        )
        contract = runner.validate_stage_s_schedule_contract(config)
        self.assertEqual(contract["selected_candidate_id"], "ret35_balanced")
        self.assertTrue(contract["selected_validation"]["replay_guard_pass"])
        self.assertFalse(contract["test_data_access"])

    def test_limited_validation_is_dialect_stratified_and_deterministic(self) -> None:
        frame = pd.DataFrame(
            [
                {"row_id": f"{dialect}-{index}", "dialect": dialect, "value": index}
                for dialect in ("BAR", "CHI", "SYL")
                for index in range(10)
            ]
        )
        first = runner.stratified_frame_subset(frame, 9)
        second = runner.stratified_frame_subset(frame, 9)
        self.assertEqual(first["row_id"].tolist(), second["row_id"].tolist())
        self.assertEqual(first["dialect"].value_counts().to_dict(), {"BAR": 3, "CHI": 3, "SYL": 3})

    def test_finalized_run_is_resumable_after_stage_a_tensor_retirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stage_s").mkdir()
            (root / "stage_id").mkdir()
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE_VALIDATION_ONLY",
                        "stages_requested": ["a", "s", "id"],
                    }
                ),
                encoding="utf-8",
            )
            for stage in ("stage_s", "stage_id"):
                (root / stage / "training_report.json").write_text(
                    json.dumps({"status": "COMPLETE"}), encoding="utf-8"
                )
                (root / stage / "best_checkpoint.pt").touch()
                (root / stage / "best_selection.json").touch()
            (root / "stage_id/temperature_calibration.json").touch()

            self.assertTrue(runner.task_run_complete(root, ["a", "s", "id"]))
            self.assertFalse(runner.task_run_complete(root, ["a"]))

    def test_stage_a_pin_is_protocol_and_variant_specific(self) -> None:
        config = {
            "protocol_id": "main_v1",
            "artifact_retention": {
                "stage_a_pins": [
                    {
                        "protocol_id": "main_v1",
                        "variant": "M3",
                        "reason": "downstream specialist",
                    }
                ]
            },
        }
        self.assertEqual(
            runner.stage_a_retention_pin(config, variant="M3"),
            "downstream specialist",
        )
        self.assertIsNone(runner.stage_a_retention_pin(config, variant="M2"))
        config["protocol_id"] = "ablation_v1"
        self.assertIsNone(runner.stage_a_retention_pin(config, variant="M3"))


if __name__ == "__main__":
    unittest.main()
