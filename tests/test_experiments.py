from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from boichitro.experiments import task_model_from_checkpoint  # noqa: E402
from boichitro.modeling import (  # noqa: E402
    BoichitroConfig,
    BoichitroForMultiTask,
    upcycle_dense_to_moe,
)


def base_config(architecture: str) -> BoichitroConfig:
    return BoichitroConfig(
        vocab_size=257,
        architecture=architecture,
        max_seq_len=256,
        n_layers=4,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        dense_ffn_dim=192,
        dense_prefix_layers=4 if architecture == "dense" else 1,
        n_routed_experts=4,
        expert_dim=64,
        top_k=2,
        shared_expert=True,
        n_sources=1,
    )


class ExperimentModelTests(unittest.TestCase):
    def checkpoint(self, model: BoichitroForMultiTask, root: Path) -> Path:
        path = root / "checkpoint.pt"
        torch.save(
            {
                "model_config": model.config.to_dict(),
                "model_state_dict": model.state_dict(),
                "tokens_seen": 123,
            },
            path,
        )
        return path

    def test_m3_adds_only_registered_task_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense = BoichitroForMultiTask(base_config("dense"))
            moe_config = base_config("standard_moe")
            moe = BoichitroForMultiTask(moe_config)
            upcycle_dense_to_moe(dense, moe, noise_std=0.0)
            path = self.checkpoint(moe, root)
            proposed, report = task_model_from_checkpoint(
                path, variant="M3", n_sources=7
            )
            self.assertEqual(proposed.config.architecture, "boichitro_moe")
            self.assertEqual(proposed.config.n_sources, 7)
            self.assertTrue(proposed.config.use_task_conditioning)
            self.assertTrue(proposed.config.use_lexical_routing_prior)
            self.assertTrue(proposed.config.use_source_adversary)
            self.assertGreater(report["copied_tensors"], 0)
            self.assertTrue(any("source_head" in name for name in report["missing_tensors"]))

    def test_randomized_lexical_negative_control_is_registered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense = BoichitroForMultiTask(base_config("dense"))
            moe = BoichitroForMultiTask(base_config("standard_moe"))
            upcycle_dense_to_moe(dense, moe, noise_std=0.0)
            path = self.checkpoint(moe, root)
            proposed, report = task_model_from_checkpoint(
                path,
                variant="M3",
                n_sources=7,
                ablations={"randomized_lexical_prior"},
            )
            self.assertTrue(proposed.config.randomize_lexical_prior)
            self.assertIn("randomized_lexical_prior", report["ablations"])

    def test_auxiliary_head_ablation_keeps_lexical_router_factor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense = BoichitroForMultiTask(base_config("dense"))
            moe = BoichitroForMultiTask(base_config("standard_moe"))
            upcycle_dense_to_moe(dense, moe, noise_std=0.0)
            path = self.checkpoint(moe, root)
            proposed, _ = task_model_from_checkpoint(
                path, variant="M3", n_sources=7, ablations={"no_dialect_head"}
            )
            self.assertIsNone(proposed.dialect_head)
            self.assertIsNotNone(proposed.routing_dialect_head)
            self.assertTrue(proposed.config.use_lexical_routing_prior)

    def test_task_adaptation_preserves_selected_permanent_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense = BoichitroForMultiTask(base_config("dense"))
            config = base_config("standard_moe")
            config.banked_upcycle_fraction = 1.0
            config.banked_upcycle_release_fraction = 1.0
            moe = BoichitroForMultiTask(config)
            upcycle_dense_to_moe(dense, moe, noise_std=0.0)
            path = self.checkpoint(moe, root)
            proposed, report = task_model_from_checkpoint(
                path, variant="M3", n_sources=7
            )
            self.assertEqual(proposed.config.banked_upcycle_fraction, 1.0)
            self.assertEqual(proposed.config.banked_upcycle_release_fraction, 1.0)
            self.assertTrue(report["permanent_paired_bank_routing"])

    def test_upcycling_pilots_share_data_compute_and_optimizer_contract(self) -> None:
        configs = {
            name: yaml.safe_load((PROJECT / path).read_text(encoding="utf-8"))
            for name, path in {
                "main": "configs/continuation_m2_200m.yaml",
                "banked": "configs/pilot_m2_banked_20m.yaml",
                "unbanked": "configs/pilot_m2_unbanked_20m.yaml",
                "scratch": "configs/pilot_m2_scratch_20m.yaml",
                "annealed": "configs/pilot_m2_annealed_20m.yaml",
                "paired": "configs/pilot_m2_paired_20m.yaml",
            }.items()
        }
        self.assertTrue(configs["main"]["evaluate_at_start"])
        for candidate in (
            configs["banked"],
            configs["unbanked"],
            configs["scratch"],
            configs["annealed"],
            configs["paired"],
        ):
            for key in (
                "seed",
                "data_seed",
                "architecture",
                "packed_data",
                "micro_batch_size",
                "gradient_accumulation_steps",
                "global_token_batch",
                "warmup_fraction",
                "optimizer",
            ):
                self.assertEqual(candidate[key], configs["main"][key])
            for key in (
                "dense_prefix_layers",
                "n_routed_experts",
                "expert_dim",
                "top_k",
                "shared_expert",
                "use_mtp",
            ):
                self.assertEqual(
                    candidate["model_overrides"][key],
                    configs["main"]["model_overrides"][key],
                )
        for candidate in (
            configs["banked"],
            configs["unbanked"],
            configs["scratch"],
            configs["annealed"],
            configs["paired"],
        ):
            self.assertEqual(candidate["token_budget"], 20_000_000)
            self.assertEqual(candidate["scheduler_token_budget"], 200_000_000)
            self.assertEqual(candidate["eval_every_tokens"], 5_000_000)
            self.assertEqual(candidate["checkpoint_every_tokens"], 20_000_000)
            self.assertTrue(candidate["evaluate_at_start"])
        self.assertEqual(
            configs["banked"]["initial_checkpoint"],
            configs["main"]["initial_checkpoint"],
        )
        self.assertEqual(
            configs["unbanked"]["initial_checkpoint"],
            configs["main"]["initial_checkpoint"],
        )
        self.assertNotIn("initial_checkpoint", configs["scratch"])
        self.assertEqual(
            configs["annealed"]["initial_checkpoint"],
            configs["main"]["initial_checkpoint"],
        )
        self.assertEqual(
            configs["paired"]["initial_checkpoint"],
            configs["main"]["initial_checkpoint"],
        )
        self.assertEqual(
            configs["banked"]["model_overrides"]["banked_upcycle_fraction"],
            0.20,
        )
        self.assertEqual(
            configs["unbanked"]["model_overrides"]["banked_upcycle_fraction"],
            0.0,
        )

    def test_continuations_use_validation_selected_mature_checkpoint_schedule(self) -> None:
        paths = (
            "configs/continuation_m0_dense_200m.yaml",
            "configs/continuation_m1_switch_200m.yaml",
            "configs/continuation_m2_200m.yaml",
            "configs/pilot_m2_banked_20m.yaml",
            "configs/pilot_m2_unbanked_20m.yaml",
            "configs/pilot_m2_scratch_20m.yaml",
            "configs/pilot_m2_annealed_20m.yaml",
            "configs/pilot_m2_paired_20m.yaml",
        )
        for path in paths:
            config = yaml.safe_load((PROJECT / path).read_text(encoding="utf-8"))
            self.assertEqual(config["warmup_fraction"], 0.0)
            self.assertEqual(
                config["optimizer_selection_report"],
                "reports/model/continuation_lr_pilot_selection.json",
            )
        pilot = yaml.safe_load(
            (PROJECT / "configs/continuation_lr_pilot.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(pilot["token_budget"], 10_000_000)
        self.assertEqual(pilot["scheduler_token_budget"], 200_000_000)
        self.assertEqual(pilot["warmup_fraction"], 0.0)
        self.assertEqual(len(pilot["candidates"]), 4)
        self.assertTrue(
            all(
                float(candidate["muon_lr"]) <= 0.01
                for candidate in pilot["candidates"]
            )
        )
        main = yaml.safe_load(
            (PROJECT / "configs/continuation_m2_200m.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            main["upcycling_selection_report"],
            "reports/model/upcycling_strategy_selection.json",
        )


if __name__ == "__main__":
    unittest.main()
