from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from boichitro.data import EncodedDataset  # noqa: E402
from boichitro.modeling import BoichitroConfig, BoichitroForMultiTask  # noqa: E402
from boichitro.training import (  # noqa: E402
    GroupDROState,
    StageTrainerConfig,
    selection_decision,
    train_stage,
)


class TrainingTests(unittest.TestCase):
    def test_primary_tolerance_uses_non_drifting_calibration_tie_break(self) -> None:
        selected = -float("inf")
        selected_tie = float("inf")
        ceiling = -float("inf")
        decisions = []
        for value, ece in ((0.7000, 0.10), (0.7010, 0.05), (0.7025, 0.20), (0.7040, 0.30)):
            choose, ceiling, _ = selection_decision(
                value,
                ece,
                selected_value=selected,
                selected_tie=selected_tie,
                primary_ceiling=ceiling,
                mode="max",
                tolerance=0.002,
                tie_mode="min",
            )
            decisions.append(choose)
            if choose:
                selected = value
                selected_tie = ece
        self.assertEqual(decisions, [True, True, False, True])
        self.assertEqual(selected, 0.7040)
        self.assertEqual(ceiling, 0.7040)

    def test_groupdro_upweights_high_loss_group(self) -> None:
        state = GroupDROState(
            2,
            eta=0.1,
            uniform_mix=0.0,
            max_ratio=100.0,
            device=torch.device("cpu"),
        )
        state.update(torch.tensor([0, 0, 1, 1]), torch.tensor([1.0, 1.0, 3.0, 3.0]))
        self.assertGreater(float(state.weights[1]), float(state.weights[0]))
        weights = state.example_weights(torch.tensor([0, 1, -100]))
        self.assertEqual(float(weights[-1]), 1.0)

    def test_fixed_budget_step_validation_selects_best_checkpoint(self) -> None:
        base = {
            "input_ids": [1, 2, 3, 4],
            "labels": [1, 2, 3, 4],
            "task_id": 0,
            "dialect_label": -100,
            "classification_label": -100,
            "source_label": -100,
            "group_id": -100,
            "example_weight": 1.0,
            "task": "clm",
            "dialect": "STD",
            "source_id": "unit",
            "semantic_group_id": "unit",
        }
        dataset = EncodedDataset(
            [{**base, "row_id": f"row-{index}"} for index in range(8)]
        )
        model = BoichitroForMultiTask(
            BoichitroConfig(
                vocab_size=16,
                architecture="dense",
                max_seq_len=8,
                n_layers=1,
                d_model=16,
                n_heads=4,
                n_kv_heads=2,
                dense_ffn_dim=32,
                dense_prefix_layers=1,
                use_mtp=False,
                use_classification_head=False,
                use_dialect_aux_head=False,
            )
        )
        markers: list[int] = []

        def validate(_model, marker: int) -> dict[str, float]:
            markers.append(marker)
            return {"objective": float(marker)}

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            report = train_stage(
                model,
                dataset,
                pad_token_id=0,
                group_count=0,
                config=StageTrainerConfig(
                    seed=11,
                    epochs=1,
                    micro_batch_size=2,
                    gradient_accumulation_steps=1,
                    use_muon=False,
                    validation_checkpoints=4,
                    log_every_steps=10,
                    num_workers=0,
                    save_optimizer_state=False,
                ),
                run_dir=run_dir,
                validation_fn=validate,
                device=torch.device("cpu"),
            )
            checkpoint = torch.load(
                run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
            )
            selection = json.loads(
                (run_dir / "best_selection.json").read_text(encoding="utf-8")
            )
            best_inode = (run_dir / "best_checkpoint.pt").stat().st_ino
            last_inode = (run_dir / "last_checkpoint.pt").stat().st_ino
        self.assertEqual(markers, [1, 2, 3, 4])
        self.assertEqual(checkpoint["extra"]["validation_id"], 4)
        self.assertEqual(checkpoint["extra"]["validation_unit"], "optimizer_step")
        self.assertEqual(checkpoint["tokens_seen"], 32)
        self.assertEqual(checkpoint["examples_seen"], 8)
        self.assertEqual(selection["status"], "SELECTED_ON_VALIDATION")
        self.assertEqual(selection["validation_id"], 4)
        self.assertEqual(report["best_metric"], 4.0)
        self.assertEqual(best_inode, last_inode)

    def test_selected_only_stage_removes_redundant_last_checkpoint(self) -> None:
        base = {
            "row_id": "row",
            "input_ids": [1, 2, 3],
            "labels": [1, 2, 3],
            "task_id": 0,
            "dialect_label": -100,
            "classification_label": -100,
            "source_label": -100,
            "group_id": -100,
            "example_weight": 1.0,
            "task": "clm",
            "dialect": "STD",
            "source_id": "unit",
            "semantic_group_id": "unit",
        }
        model = BoichitroForMultiTask(
            BoichitroConfig(
                vocab_size=8,
                max_seq_len=4,
                n_layers=1,
                d_model=8,
                n_heads=2,
                n_kv_heads=1,
                dense_ffn_dim=16,
                dense_prefix_layers=1,
                use_mtp=False,
                use_classification_head=False,
                use_dialect_aux_head=False,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            train_stage(
                model,
                EncodedDataset([base]),
                pad_token_id=0,
                group_count=0,
                config=StageTrainerConfig(
                    seed=7,
                    epochs=1,
                    micro_batch_size=1,
                    gradient_accumulation_steps=1,
                    use_muon=False,
                    validation_checkpoints=1,
                    num_workers=0,
                    save_optimizer_state=False,
                    retain_last_checkpoint=False,
                ),
                run_dir=run_dir,
                validation_fn=lambda _model, _step: {"objective": 1.0},
                device=torch.device("cpu"),
            )
            self.assertTrue((run_dir / "best_checkpoint.pt").exists())
            self.assertFalse((run_dir / "last_checkpoint.pt").exists())


if __name__ == "__main__":
    unittest.main()
