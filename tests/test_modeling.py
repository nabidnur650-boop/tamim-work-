from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from boichitro.modeling import (  # noqa: E402
    BoichitroConfig,
    BoichitroForMultiTask,
    SparseMoE,
    supervised_contrastive_per_example,
    upcycle_dense_to_moe,
)
from boichitro.optim import build_optimizers, partition_parameters  # noqa: E402


def tiny_config(architecture: str = "dense", **updates: object) -> BoichitroConfig:
    values = dict(
        vocab_size=257,
        architecture=architecture,
        max_seq_len=32,
        n_layers=4,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        dense_ffn_dim=192,
        dense_prefix_layers=1,
        n_routed_experts=4,
        expert_dim=64,
        top_k=2,
        shared_expert=True,
        n_sources=3,
        use_source_adversary=architecture == "boichitro_moe",
        use_task_conditioning=architecture == "boichitro_moe",
        use_lexical_routing_prior=architecture == "boichitro_moe",
        dropout=0.0,
    )
    values.update(updates)
    return BoichitroConfig(**values)


class ModelingTests(unittest.TestCase):
    def test_dense_forward_backward(self) -> None:
        torch.manual_seed(1)
        model = BoichitroForMultiTask(tiny_config())
        ids = torch.randint(1, 257, (3, 16))
        result = model(ids, labels=ids, task_ids=torch.zeros(3, dtype=torch.long))
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_classification_only_skips_vocabulary_projection(self) -> None:
        model = BoichitroForMultiTask(tiny_config())
        ids = torch.randint(1, 257, (2, 8))
        output = model(
            ids,
            labels=torch.full_like(ids, -100),
            classification_labels=torch.tensor([0, 1]),
        )
        self.assertIsNone(output["logits"])
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_supervised_contrastive_branch_is_finite(self) -> None:
        model = BoichitroForMultiTask(
            tiny_config(contrastive_loss_weight=0.1, contrastive_temperature=0.07)
        )
        ids = torch.randint(1, 257, (4, 8))
        output = model(
            ids,
            labels=torch.full_like(ids, -100),
            dialect_labels=torch.tensor([0, 0, 1, 1]),
            source_labels=torch.tensor([0, 1, 0, 1]),
            classification_labels=torch.tensor([0, 0, 1, 1]),
        )
        self.assertTrue(torch.isfinite(output["losses"]["contrastive"]))
        self.assertGreater(float(output["losses"]["contrastive"].detach()), 0.0)
        output["loss"].backward()

    def test_contrastive_branch_rejects_same_source_positives(self) -> None:
        model = BoichitroForMultiTask(
            tiny_config(contrastive_loss_weight=0.1, contrastive_temperature=0.07)
        )
        ids = torch.randint(1, 257, (4, 8))
        output = model(
            ids,
            labels=torch.full_like(ids, -100),
            dialect_labels=torch.tensor([0, 0, 1, 1]),
            source_labels=torch.tensor([3, 3, 4, 4]),
            classification_labels=torch.tensor([0, 0, 1, 1]),
        )
        self.assertEqual(float(output["losses"]["contrastive"]), 0.0)

    def test_cross_source_contrastive_neutralizes_same_source_pairs(self) -> None:
        representations = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]]
        )
        labels = torch.tensor([0, 0, 0, 1])
        sources = torch.tensor([3, 3, 4, 8])
        original = supervised_contrastive_per_example(
            representations, labels, sources, 0.2
        )
        altered = representations.clone()
        altered[1] = torch.tensor([-0.8, -0.2])
        changed = supervised_contrastive_per_example(altered, labels, sources, 0.2)
        self.assertAlmostEqual(float(original[0]), float(changed[0]), places=6)

    def test_bidirectional_attention_reads_future_context(self) -> None:
        torch.manual_seed(12)
        causal = BoichitroForMultiTask(tiny_config(bidirectional_attention=False))
        bidirectional = copy.deepcopy(causal)
        bidirectional.config.bidirectional_attention = True
        for layer in bidirectional.layers:
            layer.attention.bidirectional = True
        first = torch.randint(1, 257, (1, 8))
        second = first.clone()
        second[0, -1] = (second[0, -1] % 256) + 1
        causal_a = causal(first)["hidden"][:, 0]
        causal_b = causal(second)["hidden"][:, 0]
        bidirectional_a = bidirectional(first)["hidden"][:, 0]
        bidirectional_b = bidirectional(second)["hidden"][:, 0]
        torch.testing.assert_close(causal_a, causal_b, rtol=0, atol=0)
        self.assertFalse(torch.equal(bidirectional_a, bidirectional_b))

    def test_dialect_evidence_routing_is_prefix_causal(self) -> None:
        torch.manual_seed(120)
        model = BoichitroForMultiTask(tiny_config("boichitro_moe")).eval()
        model.set_training_progress(0.0)
        first = torch.randint(1, 257, (1, 9))
        second = first.clone()
        second[0, -1] = (second[0, -1] % 256) + 1
        task_ids = torch.ones(1, dtype=torch.long)
        first_routes = model(
            first, task_ids=task_ids, capture_routing=True
        )["routing"]
        second_routes = model(
            second, task_ids=task_ids, capture_routing=True
        )["routing"]
        self.assertEqual(len(first_routes), len(second_routes))
        for first_layer, second_layer in zip(first_routes, second_routes):
            torch.testing.assert_close(
                first_layer.selected_experts[:, :-1],
                second_layer.selected_experts[:, :-1],
                rtol=0,
                atol=0,
            )

    def test_kv_cache_matches_full_causal_forward(self) -> None:
        torch.manual_seed(13)
        model = BoichitroForMultiTask(tiny_config()).eval()
        ids = torch.randint(1, 257, (2, 9))
        full = model(ids)["logits"]
        prefix = ids[:, :4]
        attention = torch.ones_like(prefix)
        cached = model(prefix, attention_mask=attention, use_cache=True)
        torch.testing.assert_close(
            cached["logits"][:, -1], full[:, 3], rtol=1e-5, atol=1e-5
        )
        past = cached["past_key_values"]
        for position in range(4, ids.size(1)):
            attention = torch.cat(
                (attention, torch.ones((ids.size(0), 1), dtype=torch.long)), dim=1
            )
            cached = model(
                ids[:, position : position + 1],
                attention_mask=attention,
                past_key_values=past,
                use_cache=True,
            )
            torch.testing.assert_close(
                cached["logits"][:, -1],
                full[:, position],
                rtol=1e-5,
                atol=1e-5,
            )
            past = cached["past_key_values"]

    def test_left_padded_cached_prompt_matches_unpadded_prompt(self) -> None:
        torch.manual_seed(14)
        model = BoichitroForMultiTask(tiny_config()).eval()
        short = torch.randint(1, 257, (1, 5))
        long = torch.randint(1, 257, (1, 8))
        batch = torch.cat((torch.cat((torch.zeros((1, 3), dtype=torch.long), short), 1), long))
        attention = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1], [1] * 8])
        cached = model(batch, attention_mask=attention, use_cache=True)["logits"][:, -1]
        expected = torch.cat((model(short)["logits"][:, -1], model(long)["logits"][:, -1]))
        torch.testing.assert_close(cached, expected, rtol=1e-5, atol=1e-5)

    def test_moe_routes_every_assignment_without_drops(self) -> None:
        torch.manual_seed(2)
        model = BoichitroForMultiTask(tiny_config("standard_moe"))
        ids = torch.randint(1, 257, (2, 12))
        result = model(ids, labels=ids, capture_routing=True)
        self.assertEqual(len(result["routing"]), 3)
        for routing in result["routing"]:
            self.assertEqual(int(routing.counts.sum()), 2 * 12 * 2)
            self.assertEqual(tuple(routing.selected_experts.shape), (2, 12, 2))

    def test_router_load_excludes_padding(self) -> None:
        torch.manual_seed(22)
        model = BoichitroForMultiTask(tiny_config("standard_moe"))
        ids = torch.randint(1, 257, (2, 12))
        attention = torch.tensor([[1] * 12, [1] * 5 + [0] * 7])
        result = model(ids, attention_mask=attention, capture_routing=True)
        for routing in result["routing"]:
            self.assertEqual(int(routing.counts.sum()), (12 + 5) * 2)

    def test_router_bias_counts_accumulate_then_clear(self) -> None:
        torch.manual_seed(23)
        model = BoichitroForMultiTask(tiny_config("standard_moe"))
        ids = torch.randint(1, 257, (2, 6))
        model(ids)
        model(ids)
        modules = [module for module in model.modules() if isinstance(module, SparseMoE)]
        self.assertTrue(all(int(module.last_counts.sum()) == 2 * 2 * 6 * 2 for module in modules))
        model.update_router_biases()
        self.assertTrue(all(int(module.last_counts.sum()) == 0 for module in modules))

    def test_banked_upcycle_reconstructs_dense_ffn(self) -> None:
        torch.manual_seed(3)
        dense = BoichitroForMultiTask(tiny_config("dense", use_mtp=False))
        moe = BoichitroForMultiTask(tiny_config("standard_moe", use_mtp=False))
        upcycle_dense_to_moe(dense, moe, noise_std=0.0)
        moe.set_training_progress(0.0)
        source_ffn = dense.layers[1].ffn
        destination = moe.layers[1].ffn
        self.assertIsInstance(destination, SparseMoE)
        value = torch.randn(2, 7, 64)
        expected = source_ffn(value)
        actual = destination(
            value,
            task_ids=None,
            dialect_probabilities=None,
            lexical_prior_scale=0.0,
            bank_constraint_strength=1.0,
            capture_assignments=False,
        ).hidden
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_annealed_bank_constraint_is_deterministic_and_gradual(self) -> None:
        config = tiny_config(
            "standard_moe",
            banked_upcycle_fraction=0.2,
            banked_upcycle_release_fraction=0.6,
            banked_pairing_penalty=0.25,
        )
        model = BoichitroForMultiTask(config)
        expected = ((0.0, 1.0), (0.2, 1.0), (0.4, 0.5), (0.6, 0.0))
        for progress, strength in expected:
            model.set_training_progress(progress)
            self.assertAlmostEqual(
                model.upcycle_bank_constraint_strength(), strength, places=7
            )

        router = SparseMoE(config, layer_index=1)
        probabilities = torch.tensor([[0.40, 0.35, 0.20, 0.05]])
        scores = probabilities.clone()
        strict, _ = router._topk(probabilities, scores, 1.0)
        early_release, _ = router._topk(probabilities, scores, 0.9)
        late_release, _ = router._topk(probabilities, scores, 0.5)
        free, _ = router._topk(probabilities, scores, 0.0)
        self.assertNotEqual(bool(strict[0, 0] < 2), bool(strict[0, 1] < 2))
        self.assertNotEqual(
            bool(early_release[0, 0] < 2), bool(early_release[0, 1] < 2)
        )
        self.assertEqual(
            bool(late_release[0, 0] < 2), bool(late_release[0, 1] < 2)
        )
        self.assertEqual(free.tolist(), [[0, 1]])

    def test_permanent_pairing_remains_active_at_final_progress(self) -> None:
        model = BoichitroForMultiTask(
            tiny_config(
                "standard_moe",
                banked_upcycle_fraction=1.0,
                banked_upcycle_release_fraction=1.0,
            )
        )
        model.set_training_progress(1.0)
        self.assertEqual(model.upcycle_bank_constraint_strength(), 1.0)

    def test_bank_constraint_is_disabled_for_switch(self) -> None:
        model = BoichitroForMultiTask(
            tiny_config(
                "switch",
                expert_dim=192,
                top_k=1,
                shared_expert=False,
                banked_upcycle_fraction=1.0,
                banked_upcycle_release_fraction=1.0,
            )
        )
        model.set_training_progress(0.0)
        self.assertEqual(model.upcycle_bank_constraint_strength(), 0.0)

    def test_switch_upcycle_reconstructs_dense_ffn(self) -> None:
        torch.manual_seed(31)
        dense = BoichitroForMultiTask(tiny_config("dense", use_mtp=False))
        switch = BoichitroForMultiTask(
            tiny_config(
                "switch",
                expert_dim=192,
                top_k=1,
                shared_expert=False,
                use_mtp=False,
            )
        )
        report = upcycle_dense_to_moe(
            dense, switch, noise_std=0.0, router_init_std=1e-3
        )
        source_ffn = dense.layers[1].ffn
        destination = switch.layers[1].ffn
        self.assertIsInstance(destination, SparseMoE)
        value = torch.randn(2, 7, 64)
        expected = source_ffn(value)
        actual = destination(
            value,
            task_ids=None,
            dialect_probabilities=None,
            lexical_prior_scale=0.0,
            bank_constraint_strength=0.0,
            capture_assignments=False,
        ).hidden
        self.assertEqual(report["scheme"], "full_width_identical_switch_expert_clones")
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_switch_straight_through_gate_breaks_symmetry_and_gets_lm_gradient(self) -> None:
        torch.manual_seed(32)
        dense = BoichitroForMultiTask(tiny_config("dense", use_mtp=False))
        switch = BoichitroForMultiTask(
            tiny_config(
                "switch",
                expert_dim=192,
                top_k=1,
                shared_expert=False,
                use_mtp=False,
            )
        )
        upcycle_dense_to_moe(
            dense, switch, noise_std=0.0, router_init_std=1e-3
        )
        destination = switch.layers[1].ffn
        self.assertIsInstance(destination, SparseMoE)
        value = torch.randn(4, 16, 64, requires_grad=True)
        routed = destination(
            value,
            task_ids=None,
            dialect_probabilities=None,
            lexical_prior_scale=0.0,
            bank_constraint_strength=0.0,
            capture_assignments=True,
        )
        self.assertGreater(int(routed.counts.gt(0).sum()), 1)
        routed.hidden.square().mean().backward()
        self.assertIsNotNone(destination.router.weight.grad)
        self.assertGreater(float(destination.router.weight.grad.abs().sum()), 0.0)

    def test_optimizer_ownership_and_step(self) -> None:
        torch.manual_seed(4)
        model = BoichitroForMultiTask(tiny_config("boichitro_moe"))
        muon, adamw, muon_names, adamw_names = partition_parameters(model)
        self.assertTrue(muon)
        self.assertTrue(adamw)
        self.assertFalse(set(muon_names) & set(adamw_names))
        bundle = build_optimizers(
            model,
            muon_lr=0.01,
            adamw_lr=3e-4,
            router_lr=2e-4,
            weight_decay=0.1,
        )
        ids = torch.randint(1, 257, (2, 10))
        labels = torch.tensor([0, 1])
        result = model(
            ids,
            labels=ids,
            task_ids=torch.zeros(2, dtype=torch.long),
            dialect_labels=labels,
            classification_labels=labels,
            source_labels=torch.tensor([0, 1]),
        )
        bundle.zero_grad()
        result["loss"].backward()
        bundle.step()
        model.update_router_biases()
        self.assertTrue(torch.isfinite(result["loss"]))


if __name__ == "__main__":
    unittest.main()
