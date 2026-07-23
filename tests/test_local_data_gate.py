from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd
import yaml


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent


class LocalDataGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(
            (PROJECT / "reports" / "local_archive_audit.json").read_text(
                encoding="utf-8"
            )
        )
        cls.gate = json.loads(
            (PROJECT / "reports" / "data_gate_precheck.json").read_text(
                encoding="utf-8"
            )
        )
        cls.rows = pd.read_parquet(
            PROJECT / "data" / "manifests" / "canonical_rows_preliminary.parquet"
        )
        cls.components = pd.read_parquet(
            PROJECT
            / "data"
            / "manifests"
            / "exact_components_preliminary.parquet"
        )
        cls.registry = yaml.safe_load(
            (PROJECT / "configs" / "experiment_registry.yaml").read_text(
                encoding="utf-8"
            )
        )

    def test_source_archive_identities(self) -> None:
        self.assertEqual(
            self.audit["archives"]["archive(1).zip"]["sha256"],
            "09261dc5eec4cb0c57cf2c6b21f9003ab1acadc21253d9b8c86d2e837aa9e83c",
        )
        self.assertEqual(
            self.audit["archives"]["archive (1).zip"]["sha256"],
            "96a1c3c3114c622280809057e7acbade6960dee1b5dbf71414b1b75b53bacf12",
        )
        self.assertTrue(all(self.gate["archive_hashes_verified"].values()))

    def test_expected_source_counts(self) -> None:
        counts = self.gate["counts"]
        self.assertEqual(counts["all_manifest_rows"], 136768)
        self.assertEqual(counts["raw_rows"], 85667)
        self.assertEqual(counts["aligned_pair_rows"], 22364)
        self.assertEqual(counts["bangladial_rows"], 63303)
        self.assertEqual(counts["derived_rows"], 51101)

    def test_manifest_row_ids_are_unique(self) -> None:
        self.assertEqual(len(self.rows), self.rows["row_id"].nunique())
        self.assertFalse(self.rows["exact_component_id"].isna().any())

    def test_normalization_has_eight_dialects(self) -> None:
        pair_rows = self.rows[
            (~self.rows["derived_archive"])
            & (self.rows["text_role"] == "regional_to_standard_pair")
        ]
        self.assertEqual(
            set(pair_rows["dialect"]),
            {"BAR", "CHI", "KHU", "MYM", "NAR", "NOA", "RAN", "SYL"},
        )
        self.assertEqual(
            set(self.registry["local_data"]["normalization_dialects"]),
            set(pair_rows["dialect"]),
        )

    def test_vashantor_official_counts_preserved(self) -> None:
        vashantor = self.rows[
            (~self.rows["derived_archive"])
            & (self.rows["dataset"] == "Vashantor")
        ]
        self.assertEqual(
            vashantor.groupby("split_original").size().to_dict(),
            {"test": 1875, "train": 9375, "validation": 1250},
        )
        for split, expected in {
            "train": 1875,
            "validation": 250,
            "test": 375,
        }.items():
            per_dialect = vashantor[
                vashantor["split_original"] == split
            ].groupby("dialect").size()
            self.assertEqual(set(per_dialect), {expected})

    def test_unpartitioned_pair_splits_are_reasonably_balanced(self) -> None:
        pair_rows = self.rows[
            (~self.rows["derived_archive"])
            & (self.rows["text_role"] == "regional_to_standard_pair")
            & (self.rows["dataset"] != "Vashantor")
        ]
        for key, group in pair_rows.groupby(["dataset", "dialect"]):
            train_fraction = (
                group["split_preliminary_inherited"].eq("train").mean()
            )
            self.assertGreaterEqual(train_fraction, 0.70, key)
            self.assertLessEqual(train_fraction, 0.88, key)

    def test_no_giant_false_component(self) -> None:
        self.assertLessEqual(int(self.components["row_count"].max()), 50)

    def test_derived_archive_is_fully_quarantined(self) -> None:
        derived = self.rows[self.rows["derived_archive"]]
        self.assertEqual(len(derived), 51101)
        self.assertTrue(derived["split_final_preliminary"].eq("quarantine").all())
        self.assertFalse(
            (
                derived["eligible_normalization_train_preliminary"]
                | derived["eligible_identification_train_preliminary"]
                | derived["eligible_tokenizer_train_preliminary"]
            ).any()
        )
        self.assertEqual(
            self.gate["counts"]["derived_rows_train_eligible"], 0
        )

    def test_protected_components_have_no_train_eligible_rows(self) -> None:
        protected = set(
            self.components[
                self.components["contains_test_or_validation"]
            ]["exact_component_id"]
        )
        leaked = self.rows[
            self.rows["exact_component_id"].isin(protected)
            & (
                self.rows["eligible_normalization_train_preliminary"]
                | self.rows["eligible_identification_train_preliminary"]
                | self.rows["eligible_tokenizer_train_preliminary"]
            )
        ]
        self.assertEqual(len(leaked), 0)
        self.assertEqual(
            self.gate["counts"]["train_rows_in_protected_components"], 0
        )

    def test_obvious_placeholders_are_not_task_eligible(self) -> None:
        placeholder = self.rows["quality_flags"].str.contains(
            "angle_placeholder", regex=False
        )
        self.assertFalse(
            self.rows.loc[placeholder, "eligible_identification_task"].any()
        )
        self.assertFalse(
            self.rows.loc[placeholder, "eligible_normalization_task"].any()
        )

    def test_one_to_many_pair_mappings_are_blocked(self) -> None:
        ambiguous = self.rows["quality_flags"].str.contains(
            "source_maps_multiple_targets", regex=False
        )
        self.assertGreater(int(ambiguous.sum()), 0)
        self.assertFalse(
            self.rows.loc[ambiguous, "eligible_normalization_task"].any()
        )

    def test_bangladial_is_not_preliminarily_trainable(self) -> None:
        bangladial = self.rows[self.rows["dataset"] == "BanglaDial"]
        self.assertEqual(len(bangladial), 63303)
        self.assertFalse(
            bangladial["eligible_identification_train_preliminary"].any()
        )
        self.assertTrue(
            bangladial["preliminary_train_block_reasons"].str.contains(
                "bangladial_pending_component_provenance", regex=False
            ).all()
        )

    def test_model_active_ffn_width_is_matched(self) -> None:
        architecture = self.registry["architecture"]
        dense_width = architecture["common"]["dense_ffn_width"]
        active_moe_width = architecture["moe"]["expert_width"] * (
            architecture["moe"]["routed_top_k"]
            + architecture["moe"]["shared_experts_per_layer"]
        )
        self.assertEqual(dense_width, active_moe_width)

    def test_gate_remains_red_for_explicit_reasons(self) -> None:
        self.assertEqual(self.gate["status"], "RED_REMAINING_GATES")
        remaining = self.gate["remaining_required_gates"]
        self.assertTrue(remaining)
        self.assertFalse(any(remaining.values()))


if __name__ == "__main__":
    unittest.main()
