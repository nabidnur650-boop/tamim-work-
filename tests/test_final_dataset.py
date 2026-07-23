from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import yaml


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
FINAL = PROJECT / "data" / "final" / "v1"
MANIFESTS = PROJECT / "data" / "manifests"
REGIONAL = {
    "BAR",
    "CHI",
    "KHU",
    "KIS",
    "MYM",
    "NAR",
    "NOA",
    "NSD",
    "RAJ",
    "RAN",
    "SYL",
    "TAN",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FinalDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            (PROJECT / "reports" / "final_dataset_report.json").read_text(
                encoding="utf-8"
            )
        )
        cls.acquisition = json.loads(
            (PROJECT / "reports" / "external_source_acquisition.json").read_text(
                encoding="utf-8"
            )
        )
        cls.norm = pd.read_parquet(FINAL / "normalization_all.parquet")
        cls.identification = pd.read_parquet(FINAL / "identification_all.parquet")
        cls.romanized = pd.read_parquet(FINAL / "romanized_test_ood.parquet")
        cls.tokenizer = pd.read_parquet(FINAL / "tokenizer_train.parquet")
        cls.components = pd.read_parquet(MANIFESTS / "semantic_components.parquet")
        cls.licenses = yaml.safe_load(
            (MANIFESTS / "licenses.yaml").read_text(encoding="utf-8")
        )
        cls.registry = yaml.safe_load(
            (PROJECT / "configs" / "experiment_registry.yaml").read_text(
                encoding="utf-8"
            )
        )

    def test_frozen_expected_counts(self) -> None:
        self.assertEqual(len(self.norm), 57923)
        self.assertEqual(int((~self.norm["is_synthetic"]).sum()), 54598)
        self.assertEqual(int(self.norm["is_synthetic"].sum()), 3325)
        self.assertEqual(len(self.identification), 122353)
        self.assertEqual(len(self.romanized), 1342)
        self.assertEqual(len(self.tokenizer), 100236)

    def test_inventory_is_twelve_regional_plus_standard(self) -> None:
        labels = set(self.identification["dialect"])
        self.assertEqual(labels, REGIONAL | {"STD"})
        self.assertEqual(len(labels), 13)
        self.assertNotIn("STD", REGIONAL)

    def test_normalization_scope_is_explicit(self) -> None:
        train_dialects = set(self.norm[self.norm["split"].eq("train")]["dialect"])
        self.assertEqual(
            train_dialects,
            {"BAR", "CHI", "KHU", "MYM", "NAR", "NOA", "RAN", "SYL"},
        )
        raj = self.norm[self.norm["dialect"].eq("RAJ")]
        self.assertTrue(raj["split"].eq("test_ood").all())

    def test_row_ids_are_unique(self) -> None:
        self.assertEqual(len(self.norm), self.norm["row_id"].nunique())
        self.assertEqual(
            len(self.identification), self.identification["row_id"].nunique()
        )
        canonical = pd.read_parquet(MANIFESTS / "canonical_rows.parquet")
        self.assertEqual(len(canonical), canonical["row_id"].nunique())

    def test_synthetic_is_traceable_train_only(self) -> None:
        synthetic = self.norm[self.norm["is_synthetic"]]
        authentic_train = self.norm[
            (~self.norm["is_synthetic"]) & self.norm["split"].eq("train")
        ]
        self.assertTrue(synthetic["split"].eq("train").all())
        self.assertFalse(synthetic["eval_eligible"].any())
        self.assertFalse(synthetic["identification_eligible"].any())
        self.assertEqual(
            set(synthetic["parent_row_id"]) - set(authentic_train["row_id"]), set()
        )
        self.assertTrue(synthetic["example_loss_weight"].eq(0.5).all())
        self.assertLessEqual(len(synthetic) / len(authentic_train), 0.10)
        for dialect, group in synthetic.groupby("dialect"):
            denominator = int(authentic_train["dialect"].eq(dialect).sum())
            self.assertLessEqual(len(group) / denominator, 0.15)

    def test_no_compact_train_eval_overlap(self) -> None:
        authentic = self.norm[~self.norm["is_synthetic"]]
        for column in ["source_text_compact", "target_text_compact"]:
            train = set(authentic[authentic["split"].eq("train")][column])
            protected = set(authentic[~authentic["split"].eq("train")][column])
            self.assertEqual((train & protected) - {""}, set(), column)
        train = set(
            self.identification[self.identification["split"].eq("train")][
                "text_compact"
            ]
        )
        protected = set(
            self.identification[~self.identification["split"].eq("train")][
                "text_compact"
            ]
        )
        self.assertEqual((train & protected) - {""}, set())

    def test_no_identification_cross_label_text(self) -> None:
        conflicts = self.identification.groupby("text_compact")["dialect"].nunique()
        self.assertFalse(conflicts.gt(1).any())

    def test_no_conflicting_normalization_supervision(self) -> None:
        authentic = self.norm[~self.norm["is_synthetic"]]
        conflicts = authentic.groupby(["dialect", "source_text_model"])[
            "target_text_model"
        ].nunique()
        self.assertFalse(conflicts.gt(1).any())

    def test_locked_sources_are_evaluation_only(self) -> None:
        locked = self.norm[
            self.norm["source_id"].isin({"onubad_v2", "bd_dialect_v2"})
        ]
        self.assertTrue(locked["split"].eq("test_ood").all())
        self.assertFalse(locked["train_eligible"].any())

    def test_romanized_rows_link_to_accepted_onubad(self) -> None:
        onubad = set(self.norm[self.norm["source_id"].eq("onubad_v2")]["row_id"])
        self.assertEqual(
            set(self.romanized["parent_normalization_row_id"]) - onubad, set()
        )
        self.assertTrue(self.romanized["split"].eq("test_ood").all())

    def test_hf_mapping_does_not_infer_geographic_labels(self) -> None:
        hf = self.identification[
            self.identification["source_id"].eq("hf_bengali_regional_asr_refine")
        ]
        self.assertNotIn("habiganj", set(hf["dialect"]))
        self.assertNotIn("sandwip", set(hf["dialect"]))
        exclusions = pd.read_parquet(FINAL / "excluded_rows.parquet")
        unmapped = exclusions[
            exclusions["reason"].eq("hf_district_outside_frozen_taxonomy")
        ]
        self.assertEqual(len(unmapped), 1989)

    def test_legacy_derived_archive_is_absent(self) -> None:
        source_ids = set(self.norm["source_id"]) | set(self.identification["source_id"])
        self.assertFalse(any("derived_archive" in source for source in source_ids))
        self.assertEqual(
            self.report["counts"]["legacy_derived_archive_quarantined_rows"], 51101
        )

    def test_licenses_are_resolved_and_caveats_retained(self) -> None:
        self.assertFalse(self.norm["license"].str.contains("UNKNOWN").any())
        self.assertFalse(self.identification["license"].str.contains("UNKNOWN").any())
        hf = self.licenses["sources"]["hf_bengali_regional_asr_refine"]
        self.assertIn("caveat", hf)
        self.assertEqual(hf["revision"], "6e3221bdf7f9c8c426276f1b619d7de597963f42")

    def test_external_download_hashes(self) -> None:
        for source in self.acquisition["sources"]:
            for item in source.get("downloaded_files", []):
                path = PROJECT / item["path"]
                self.assertTrue(path.exists(), path)
                self.assertEqual(sha256_file(path), item["sha256"], path)
        extract = next(
            source["processed_extract"]
            for source in self.acquisition["sources"]
            if source["source_id"] == "hf_bengali_regional_asr_refine"
        )
        self.assertEqual(sha256_file(PROJECT / extract["path"]), extract["sha256"])

    def test_local_file_level_provenance_matches_repositories(self) -> None:
        archive = WORKSPACE / "archive(1).zip"
        expected = {
            "Bangla Dialect Transaction Dataset_v2 - Sheet1.csv": (
                155934,
                "f5d36b030a2b888091b2a9531fa0006403ed642a2b1e03889864ab0fd276f0de",
            ),
            "Dataset_Chittagong_2.0.csv": (
                580143,
                "c009054d79894935200f91bbf9e2bbd97fdac2efdd4222a0005a08345a01425b",
            ),
        }
        # Filename capitalization in the archive is fixed; verify the two local
        # parallel files whose Mendeley API hashes were reconstructed exactly.
        with ZipFile(archive) as zipped:
            for suffix, (size, digest) in expected.items():
                member = next(name for name in zipped.namelist() if name.endswith(suffix))
                payload = zipped.read(member)
                self.assertEqual(len(payload), size)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)

    def test_tokenizer_has_no_exact_evaluation_text(self) -> None:
        protected = set(
            self.norm[~self.norm["split"].eq("train")]["source_text_model"]
        ) | set(self.norm[~self.norm["split"].eq("train")]["target_text_model"])
        protected |= set(
            self.identification[~self.identification["split"].eq("train")][
                "text_model"
            ]
        )
        self.assertEqual(set(self.tokenizer["text_model"]) & protected, set())

    def test_semantic_components_do_not_cross_train_protection(self) -> None:
        crossed = self.components[
            self.components["contains_train"]
            & self.components["contains_protected_evaluation"]
        ]
        self.assertEqual(len(crossed), 0)

    def test_artifact_hash_ledger(self) -> None:
        for relative, metadata in self.report["artifacts"].items():
            path = PROJECT / relative
            self.assertTrue(path.exists(), path)
            self.assertEqual(sha256_file(path), metadata["sha256"], path)

    def test_experiment_registry_points_to_frozen_artifacts(self) -> None:
        manifest_keys = {
            "canonical_rows": "canonical_rows.parquet",
            "connected_components": "semantic_components.parquet",
            "exclusions": "excluded_rows.parquet",
            "split_manifest": "splits_v1.parquet",
            "license_ledger": "licenses.yaml",
        }
        for key, filename in manifest_keys.items():
            entry = self.registry["data_manifests"][key]
            self.assertEqual(entry["path"], f"data/manifests/{filename}")
            self.assertEqual(entry["sha256"], sha256_file(MANIFESTS / filename))
        release = self.registry["dataset_release"]
        self.assertTrue(release["training_authorized"])
        self.assertFalse(release["public_redistribution_authorized"])
        self.assertEqual(release["normalization_all_rows"], len(self.norm))
        self.assertEqual(release["identification_rows"], len(self.identification))

    def test_machine_gate_authorizes_only_internal_training(self) -> None:
        gate = json.loads(
            (PROJECT / "reports" / "data_gate.json").read_text(encoding="utf-8")
        )
        self.assertEqual(gate["status"], "pass")
        self.assertTrue(gate["training_authorized"])
        self.assertFalse(gate["public_redistribution_authorized"])
        self.assertFalse(gate["publication_claim_authorized"])

    def test_publication_gate_is_honest(self) -> None:
        self.assertEqual(self.report["status"], "PASS_INTERNAL_DATA_ENGINEERING")
        self.assertEqual(
            self.report["publication_release_status"],
            "CONDITIONAL_NATIVE_REVIEW_REQUIRED",
        )
        self.assertFalse(self.report["gates"]["native_human_review_complete"])


if __name__ == "__main__":
    unittest.main()
