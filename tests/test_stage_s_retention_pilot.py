from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "run_stage_s_retention_pilot",
    PROJECT / "tools/run_stage_s_retention_pilot.py",
)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class StageSRetentionPilotTests(unittest.TestCase):
    def test_registered_candidates_are_fixed_valid_mixtures(self) -> None:
        config = yaml.safe_load(
            (PROJECT / "configs/stage_s_retention_pilot.yaml").read_text(
                encoding="utf-8"
            )
        )
        pilot.validate_pilot_config(config)
        self.assertEqual(len(config["candidates"]), 4)
        self.assertEqual(config["development_seed"], 1701)
        self.assertEqual(config["token_budget"], 6_000_000)
        self.assertEqual(config["replay_degradation_limit"], 0.05)
        for candidate in config["candidates"]:
            self.assertAlmostEqual(sum(candidate["mixture"].values()), 1.0)
            self.assertGreaterEqual(candidate["mixture"]["general_replay"], 0.25)

    def test_selection_excludes_higher_scoring_guard_failure(self) -> None:
        curve = [
            {
                "optimizer_step": 1,
                "macro_chrfpp": 50.0,
                "worst_dialect_chrfpp": 40.0,
                "replay_relative_degradation": 0.051,
                "replay_guard_pass": False,
            },
            {
                "optimizer_step": 2,
                "macro_chrfpp": 45.0,
                "worst_dialect_chrfpp": 35.0,
                "replay_relative_degradation": 0.049,
                "replay_guard_pass": True,
            },
        ]
        self.assertEqual(pilot.selected_row(curve)["optimizer_step"], 2)

    def test_default_failure_snapshot_is_separate_from_main_protocol(self) -> None:
        rejected = yaml.safe_load(
            (
                PROJECT
                / "configs/task_experiments_rejected_stage_s_default.yaml"
            ).read_text(encoding="utf-8")
        )
        main = yaml.safe_load(
            (PROJECT / "configs/task_experiments.yaml").read_text(encoding="utf-8")
        )
        self.assertNotEqual(rejected["protocol_id"], main["protocol_id"])
        self.assertEqual(rejected["stage_s"]["mixture"]["general_replay"], 0.10)
        self.assertEqual(rejected["stage_s"]["trainer"]["muon_lr"], 0.006)


if __name__ == "__main__":
    unittest.main()
