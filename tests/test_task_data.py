from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from boichitro.data import (  # noqa: E402
    FixedMixtureDataset,
    FixedTokenMixtureDataset,
    MaskedNextTokenDataset,
    EncodedDataset,
    collate_examples,
    encode_identification,
    encode_normalization,
    exclude_sources,
    renormalize_proportions,
)
from boichitro.tokenization import CandidateSpec, build_tokenizer  # noqa: E402


class TaskDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        path = Path(self.temporary.name)
        texts = ["মুই ভাত খামু", "আমি ভাত খাব", "আঁই ঘরত যাই"]
        build_tokenizer(CandidateSpec("byte_bpe_balanced", 512, "balanced"), texts, path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.sources = {"s": 0}
        self.groups = {"BAR|s|authentic": 0}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_normalization_masks_prefix(self) -> None:
        example = encode_normalization(
            {
                "row_id": "r",
                "dialect": "BAR",
                "source_id": "s",
                "source_text_model": "মুই ভাত খামু",
                "target_text_model": "আমি ভাত খাব",
                "semantic_group_id": "g",
                "is_synthetic": False,
                "example_loss_weight": 1.0,
            },
            self.tokenizer,
            max_length=64,
            source_to_id=self.sources,
            group_to_id=self.groups,
        )
        separator = example.input_ids.index(self.tokenizer.token_to_id("<sep>"))
        self.assertTrue(all(label == -100 for label in example.labels[: separator + 1]))
        self.assertTrue(all(label >= 0 for label in example.labels[separator + 1 :]))

    def test_identification_cls_is_last_and_collates(self) -> None:
        example = encode_identification(
            {
                "row_id": "r",
                "dialect": "BAR",
                "source_id": "s",
                "text_model": "মুই ভাত খামু",
                "is_synthetic": False,
            },
            self.tokenizer,
            max_length=64,
            source_to_id=self.sources,
            group_to_id=self.groups,
        )
        self.assertEqual(example.input_ids[-1], self.tokenizer.token_to_id("<cls>"))
        batch = collate_examples([example.to_dict(), example.to_dict()], 0)
        self.assertEqual(tuple(batch["input_ids"].shape)[0], 2)
        self.assertEqual(batch["classification_labels"].tolist(), [0, 0])

    def test_fixed_mixture_is_reproducible(self) -> None:
        item = {
            "row_id": "x",
            "input_ids": [1],
            "labels": [-100],
            "task_id": 0,
            "dialect_label": -100,
            "classification_label": -100,
            "source_label": -100,
            "group_id": -100,
            "example_weight": 1.0,
            "task": "clm",
            "dialect": "STD",
            "source_id": "s",
            "semantic_group_id": "x",
        }
        components = {"a": EncodedDataset([item]), "b": EncodedDataset([item])}
        first = FixedMixtureDataset(components, {"a": 0.7, "b": 0.3}, epoch_examples=10, seed=1)
        second = FixedMixtureDataset(components, {"a": 0.7, "b": 0.3}, epoch_examples=10, seed=1)
        self.assertEqual(first.schedule, second.schedule)

    def test_fixed_token_mixture_tracks_token_proportions(self) -> None:
        base = {
            "row_id": "x",
            "labels": [-100],
            "task_id": 0,
            "dialect_label": -100,
            "classification_label": -100,
            "source_label": -100,
            "group_id": -100,
            "example_weight": 1.0,
            "task": "clm",
            "dialect": "STD",
            "source_id": "s",
            "semantic_group_id": "x",
        }
        short = EncodedDataset([{**base, "input_ids": [1, 2]}])
        long = EncodedDataset([{**base, "input_ids": list(range(10))}])
        mixture = FixedTokenMixtureDataset(
            {"short": short, "long": long},
            {"short": 0.5, "long": 0.5},
            token_budget=100,
            seed=17,
        )
        report = mixture.report()
        self.assertEqual(report["realized_tokens"]["short"], 50)
        self.assertEqual(report["realized_tokens"]["long"], 50)
        self.assertEqual(report["draws"]["short"], 25)
        self.assertEqual(report["draws"]["long"], 5)
        self.assertEqual(report["maximum_example_repeats"]["short"], 25)
        self.assertEqual(report["maximum_example_repeats"]["long"], 5)

    def test_mntp_masking_is_deterministic_and_disables_classification(self) -> None:
        example = encode_identification(
            {
                "row_id": "mask-me",
                "dialect": "BAR",
                "source_id": "s",
                "text_model": "মুই ভাত খামু",
                "is_synthetic": False,
            },
            self.tokenizer,
            max_length=64,
            source_to_id=self.sources,
            group_to_id=self.groups,
        ).to_dict()
        special_ids = [
            value
            for token in ("<pad>", "<bos>", "<eos>", "<cls>", "<task_id>", "<dial_unknown>")
            if (value := self.tokenizer.token_to_id(token)) is not None
        ]
        dataset = MaskedNextTokenDataset(
            EncodedDataset([example]),
            mask_token_id=self.tokenizer.token_to_id("<mask>"),
            vocab_size=self.tokenizer.get_vocab_size(),
            special_token_ids=special_ids,
            mask_probability=0.5,
            seed=91,
        )
        first = dataset[0]
        second = dataset[0]
        self.assertEqual(first, second)
        self.assertEqual(first["classification_label"], -100)
        supervised = [index for index, label in enumerate(first["labels"]) if label != -100]
        self.assertTrue(supervised)
        self.assertTrue(all(index > 0 for index in supervised))

    def test_data_ablations_filter_and_renormalize_without_changing_budget(self) -> None:
        examples = EncodedDataset(
            [
                {"row_id": "real", "source_id": "authentic"},
                {"row_id": "synthetic", "source_id": "synthetic_robustness_v1"},
            ]
        )
        filtered = exclude_sources(examples, ["synthetic_robustness_v1"])
        self.assertEqual([row["row_id"] for row in filtered.examples], ["real"])
        mixture = renormalize_proportions(
            {"general_replay": 0.65, "dialect_clm": 0.20, "normalization": 0.15},
            ["general_replay"],
        )
        self.assertAlmostEqual(sum(mixture.values()), 1.0)
        self.assertNotIn("general_replay", mixture)
        self.assertAlmostEqual(mixture["dialect_clm"], 0.20 / 0.35)


if __name__ == "__main__":
    unittest.main()
