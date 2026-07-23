#!/usr/bin/env python3
"""Build a conservative preliminary manifest from the audited local ZIP files.

This is intentionally an exact-match data gate, not the final dataset build.
It preserves raw and normalized text, creates exact connected components,
inherits protected Vashantor splits, removes obvious malformed/conflicting
classification rows, and quarantines every row in the derived archive.

The generated gate remains red until near-duplicate, template, licensing,
component-provenance, and human-quality reviews are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import pandas as pd

from audit_local_archives import (
    DERIVED_ARCHIVE,
    RAW_ARCHIVE,
    find_member,
    load_derived_archive,
    load_raw_datasets,
    normalize_text,
    sha256_file,
)


RAW_ARCHIVE_SHA256 = (
    "09261dc5eec4cb0c57cf2c6b21f9003ab1acadc21253d9b8c86d2e837aa9e83c"
)
DERIVED_ARCHIVE_SHA256 = (
    "96a1c3c3114c622280809057e7acbade6960dee1b5dbf71414b1b75b53bacf12"
)
TAXONOMY_VERSION = "boichitro_taxonomy_v1"
MANIFEST_VERSION = "preliminary_exact_v1"

PUNCTUATION_ONLY = re.compile(r"^[\s<>।?!,.;:'\"“”‘’—–_\-/\\|()[\]{}…]+$")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def stable_hash(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


def has_letter(text: str) -> bool:
    return any(unicodedata.category(char).startswith("L") for char in text)


def parse_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def obvious_text_flags(text: str) -> list[str]:
    flags = []
    if not text:
        flags.append("empty_after_cleaning")
        return flags
    if "<" in text or ">" in text:
        flags.append("angle_placeholder")
    if PUNCTUATION_ONLY.fullmatch(text):
        flags.append("punctuation_or_placeholder_only")
    if not has_letter(text):
        flags.append("no_unicode_letter")
    if len(text) > 1000:
        flags.append("extreme_length_over_1000_chars")
    return flags


def archive_members(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        return archive.namelist()


def bangladial_member_name(raw_path: Path) -> str:
    return find_member(
        archive_members(raw_path),
        "BanglaDial_ A Merged and Imbalanced text Dataset for Bengali "
        "Regional dialect analysis. - Sheet1.csv",
    )


def build_pair_records(
    pairs: pd.DataFrame,
    archive_sha256: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in pairs.iterrows():
        dataset = str(row["dataset"])
        member = str(row["archive_member"])
        dialect = str(row["dialect"])
        source_row_id = int(row["source_row_id"])
        row_id = stable_hash(
            archive_sha256,
            member,
            source_row_id,
            dialect,
            "parallel_pair",
        )
        source_clean = normalize_text(row["source_text"])
        target_clean = normalize_text(row["standard_text"])
        flags = obvious_text_flags(source_clean)
        flags.extend(f"target_{flag}" for flag in obvious_text_flags(target_clean))
        if source_clean == target_clean:
            flags.append("source_equals_target_review")
        records.append(
            {
                "row_id": row_id,
                "manifest_version": MANIFEST_VERSION,
                "archive_filename": RAW_ARCHIVE,
                "archive_sha256": archive_sha256,
                "archive_member": member,
                "source_row_id": source_row_id,
                "dataset": dataset,
                "source_kind": "curated_parallel",
                "text_role": "regional_to_standard_pair",
                "dialect": dialect,
                "taxonomy_version": TAXONOMY_VERSION,
                "source_text_raw": str(row.get("source_text_raw", source_clean)),
                "source_text_clean": source_clean,
                "target_text_raw": str(row.get("standard_text_raw", target_clean)),
                "target_text_clean": target_clean,
                "source_romanized": normalize_text(row.get("source_romanized")),
                "target_romanized": normalize_text(row.get("standard_romanized")),
                "english_text": normalize_text(row.get("english_text")),
                "semantic_group_original": str(row.get("semantic_group_id", "")),
                "split_original": str(row.get("split", "unsplit")),
                "is_synthetic": False,
                "derived_archive": False,
                "quality_flags": unique_sorted(flags),
                "source_metadata": compact_json({}),
                "license": "UNKNOWN_ARCHIVE_NO_LICENSE_MEMBER",
                "redistribution_status": "blocked_pending_license_ledger",
            }
        )
    return records


def build_bangladial_records(
    bangladial: pd.DataFrame,
    member: str,
    archive_sha256: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_row_id, row in bangladial.reset_index(drop=True).iterrows():
        dialect = str(row["dialect"])
        text_clean = normalize_text(row["text"])
        row_id = stable_hash(
            archive_sha256,
            member,
            source_row_id,
            dialect,
            "classification_text",
        )
        records.append(
            {
                "row_id": row_id,
                "manifest_version": MANIFEST_VERSION,
                "archive_filename": RAW_ARCHIVE,
                "archive_sha256": archive_sha256,
                "archive_member": member,
                "source_row_id": int(source_row_id),
                "dataset": "BanglaDial",
                "source_kind": "merged_component_unknown",
                "text_role": "classification_text",
                "dialect": dialect,
                "taxonomy_version": TAXONOMY_VERSION,
                "source_text_raw": str(row.get("text_raw", text_clean)),
                "source_text_clean": text_clean,
                "target_text_raw": "",
                "target_text_clean": "",
                "source_romanized": "",
                "target_romanized": "",
                "english_text": "",
                "semantic_group_original": "",
                "split_original": "unknown_merged",
                "is_synthetic": False,
                "derived_archive": False,
                "quality_flags": obvious_text_flags(text_clean),
                "source_metadata": compact_json(
                    {"original_language_label": str(row["language"])}
                ),
                "license": "UNKNOWN_ARCHIVE_NO_LICENSE_MEMBER",
                "redistribution_status": "blocked_pending_license_ledger",
            }
        )
    return records


def derived_source_kind(row: pd.Series) -> str:
    source = str(row.get("source", "")).lower()
    if "augmented" in source:
        return "augmented"
    if bool(row.get("is_synthetic", False)) or "synthetic" in source:
        return "synthetic"
    return "derived_or_copied"


def build_derived_records(
    derived: pd.DataFrame,
    archive_sha256: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    local_index = defaultdict(int)
    for _, row in derived.iterrows():
        member = str(row["archive_member"])
        source_row_id = local_index[member]
        local_index[member] += 1
        dialect = str(row["dialect"])
        text_clean = normalize_text(row["text_normalized"])
        row_id = stable_hash(
            archive_sha256,
            member,
            source_row_id,
            dialect,
            "derived_text",
        )
        metadata_keys = [
            key
            for key in [
                "source",
                "split_original",
                "synthetic_method",
                "domain",
                "translation_pair_id",
                "quality_score",
                "near_duplicate_group",
                "taxonomy_role",
                "generation_version",
                "created_at_utc",
            ]
            if key in row.index and not pd.isna(row[key])
        ]
        metadata = {key: str(row[key]) for key in metadata_keys}
        flags = obvious_text_flags(text_clean)
        flags.append("derived_archive_default_quarantine")
        source_text_raw = row.get("text_raw", row.get("text", text_clean))
        records.append(
            {
                "row_id": row_id,
                "manifest_version": MANIFEST_VERSION,
                "archive_filename": DERIVED_ARCHIVE,
                "archive_sha256": archive_sha256,
                "archive_member": member,
                "source_row_id": int(source_row_id),
                "dataset": f"Derived::{member}",
                "source_kind": derived_source_kind(row),
                "text_role": "derived_classification_text",
                "dialect": dialect,
                "taxonomy_version": TAXONOMY_VERSION,
                "source_text_raw": str(source_text_raw),
                "source_text_clean": text_clean,
                "target_text_raw": "",
                "target_text_clean": "",
                "source_romanized": "",
                "target_romanized": "",
                "english_text": "",
                "semantic_group_original": "",
                "split_original": str(row.get("split_original", "derived_unknown")),
                "is_synthetic": parse_boolean(row.get("is_synthetic", False))
                or derived_source_kind(row) in {"synthetic", "augmented"},
                "derived_archive": True,
                "quality_flags": unique_sorted(flags),
                "source_metadata": compact_json(metadata),
                "license": "UNKNOWN_ARCHIVE_NO_LICENSE_MEMBER",
                "redistribution_status": "blocked_pending_license_ledger",
            }
        )
    return records


def add_bangladial_match_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    pair_rows = frame[
        (frame["text_role"] == "regional_to_standard_pair")
        & (~frame["derived_archive"])
    ]
    match_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for _, row in pair_rows.iterrows():
        match_index[row["source_text_clean"]].append(
            {
                "dataset": row["dataset"],
                "split_original": row["split_original"],
                "dialect": row["dialect"],
                "match_role": "regional_source",
            }
        )
        match_index[row["target_text_clean"]].append(
            {
                "dataset": row["dataset"],
                "split_original": row["split_original"],
                "dialect": "STD",
                "match_role": "standard_target",
            }
        )

    exact_matches = []
    inferred_source = []
    for _, row in frame.iterrows():
        if row["dataset"] != "BanglaDial":
            exact_matches.append("[]")
            inferred_source.append("")
            continue
        matches = sorted(
            match_index.get(row["source_text_clean"], []),
            key=lambda value: compact_json(value),
        )
        exact_matches.append(compact_json(matches))
        datasets = unique_sorted(match["dataset"] for match in matches)
        inferred_source.append("+".join(datasets) if datasets else "unknown")
    frame = frame.copy()
    frame["exact_component_matches"] = exact_matches
    frame["component_source_inferred"] = inferred_source
    return frame


def add_pair_mapping_conflict_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag regional sources that map to multiple Standard targets.

    A small number of exact regional strings have two different targets. If
    retained as graph edges, these one-to-many cases connect otherwise distinct
    semantic groups and can create a giant component. They require human
    adjudication and are excluded from preliminary task eligibility.
    """
    result = frame.copy()
    pair_rows = result[
        (~result["derived_archive"])
        & (result["text_role"] == "regional_to_standard_pair")
    ]
    target_counts = pair_rows.groupby("source_text_clean")[
        "target_text_clean"
    ].nunique()
    ambiguous_sources = set(target_counts[target_counts > 1].index)
    mask = (
        (~result["derived_archive"])
        & (result["text_role"] == "regional_to_standard_pair")
        & result["source_text_clean"].isin(ambiguous_sources)
    )
    result.loc[mask, "quality_flags"] = result.loc[
        mask, "quality_flags"
    ].map(lambda flags: unique_sorted([*flags, "source_maps_multiple_targets"]))
    return result


def build_exact_components(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    union_find = UnionFind(len(frame))
    first_by_source_text: dict[str, int] = {}
    first_by_target_text: dict[str, int] = {}
    first_by_classification_text: dict[str, int] = {}
    first_by_semantic_group: dict[str, int] = {}
    bangladial_label_sets = (
        frame[
            (~frame["derived_archive"])
            & (frame["dataset"] == "BanglaDial")
            & (frame["source_text_clean"] != "")
        ]
        .groupby("source_text_clean")["dialect"]
        .agg(lambda values: set(values))
    )
    ambiguous_bangladial_texts = set(
        bangladial_label_sets[
            bangladial_label_sets.map(len) > 1
        ].index
    )

    for index, row in frame.iterrows():
        # Derived rows are retained for audit but never influence raw split
        # construction. Their parent/source labels are incomplete, and using
        # them as graph bridges can create synthetic template mega-components.
        if row["derived_archive"]:
            continue

        flags = set(row["quality_flags"])
        graph_blocked = bool(
            flags
            & {
                "empty_after_cleaning",
                "angle_placeholder",
                "punctuation_or_placeholder_only",
                "no_unicode_letter",
                "source_maps_multiple_targets",
            }
        )
        if graph_blocked:
            continue

        if (
            row["dataset"] == "BanglaDial"
            and row["source_text_clean"] in ambiguous_bangladial_texts
        ):
            # Ambiguous merged-corpus labels are excluded later and must not
            # bridge Standard-target and regional-source graphs.
            continue

        source_text = row["source_text_clean"]
        target_text = row["target_text_clean"]
        if row["text_role"] == "regional_to_standard_pair":
            if source_text:
                if source_text in first_by_source_text:
                    union_find.union(index, first_by_source_text[source_text])
                else:
                    first_by_source_text[source_text] = index
            if target_text:
                if target_text in first_by_target_text:
                    union_find.union(index, first_by_target_text[target_text])
                else:
                    first_by_target_text[target_text] = index
        else:
            # BanglaDial STD rows match Standard targets; regional rows match
            # regional sources. A separate map still joins exact classification
            # duplicates and conflicts across labels for conservative splitting.
            if source_text in first_by_classification_text:
                union_find.union(
                    index, first_by_classification_text[source_text]
                )
            else:
                first_by_classification_text[source_text] = index
            role_index = (
                first_by_target_text
                if row["dialect"] == "STD"
                else first_by_source_text
            )
            if source_text in role_index:
                union_find.union(index, role_index[source_text])
            else:
                role_index[source_text] = index

        semantic_group = row["semantic_group_original"]
        if semantic_group:
            if semantic_group in first_by_semantic_group:
                union_find.union(index, first_by_semantic_group[semantic_group])
            else:
                first_by_semantic_group[semantic_group] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(frame)):
        groups[union_find.find(index)].append(index)

    row_ids_column = frame["row_id"].tolist()
    dataset_column = frame["dataset"].tolist()
    dialect_column = frame["dialect"].tolist()
    derived_column = frame["derived_archive"].tolist()
    split_column = frame["split_original"].tolist()
    metadata_column = frame["source_metadata"].tolist()
    component_id_by_index: dict[int, str] = {}
    component_rows: list[dict[str, Any]] = []
    for indexes in groups.values():
        row_ids = sorted(row_ids_column[index] for index in indexes)
        component_id = stable_hash("exact_component", *row_ids)
        official_splits = {
            split_column[index]
            for index in indexes
            if dataset_column[index] == "Vashantor"
            and split_column[index] in {"train", "validation", "test"}
        }
        derived_sources = []
        for index in indexes:
            if derived_column[index]:
                parsed = json.loads(metadata_column[index])
                source = str(parsed.get("source", ""))
                if source:
                    derived_sources.append(source)
        derived_protected = set()
        for source in derived_sources:
            if "Test" in source:
                derived_protected.add("test")
            if "Validation" in source:
                derived_protected.add("validation")

        protected_splits = official_splits | derived_protected
        if "test" in protected_splits:
            inherited_split = "test"
        elif "validation" in protected_splits:
            inherited_split = "validation"
        elif "train" in protected_splits:
            inherited_split = "train"
        else:
            bucket = int(component_id[:12], 16) % 1000
            if bucket < 800:
                inherited_split = "train"
            elif bucket < 900:
                inherited_split = "validation"
            else:
                inherited_split = "test"

        for index in indexes:
            component_id_by_index[index] = component_id
        component_rows.append(
            {
                "exact_component_id": component_id,
                "row_count": len(indexes),
                "raw_row_count": sum(
                    not derived_column[index] for index in indexes
                ),
                "derived_row_count": sum(
                    derived_column[index] for index in indexes
                ),
                "datasets": compact_json(
                    unique_sorted(dataset_column[index] for index in indexes)
                ),
                "dialects": compact_json(
                    unique_sorted(dialect_column[index] for index in indexes)
                ),
                "official_splits": compact_json(sorted(official_splits)),
                "derived_protected_splits": compact_json(
                    sorted(derived_protected)
                ),
                "inherited_split": inherited_split,
                "contains_split_conflict": len(official_splits) > 1,
                "contains_test_or_validation": bool(
                    {"test", "validation"} & protected_splits
                ),
            }
        )

    result = frame.copy()
    result["exact_component_id"] = [
        component_id_by_index[index] for index in range(len(result))
    ]
    components = pd.DataFrame(component_rows).sort_values(
        ["row_count", "exact_component_id"], ascending=[False, True]
    )
    return result, components


def mark_conflicts_and_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    raw = result[~result["derived_archive"]]

    classification_label_sets = (
        raw[raw["source_text_clean"] != ""]
        .groupby("source_text_clean")["dialect"]
        .agg(lambda values: sorted(set(values)))
    )
    ambiguous_texts = set(
        classification_label_sets[
            classification_label_sets.map(len) > 1
        ].index
    )
    result["classification_label_conflict"] = result[
        "source_text_clean"
    ].isin(ambiguous_texts)

    result["normalization_representative"] = False
    pair_candidates = result[
        (~result["derived_archive"])
        & (result["text_role"] == "regional_to_standard_pair")
    ].copy()
    pair_candidates["priority"] = pair_candidates["dataset"].map(
        {
            "Vashantor": 0,
            "ChatgaiyyaAlap": 1,
            "Sylheti1200": 1,
            "BanglaRegionalTextCorpus": 1,
        }
    ).fillna(2)
    pair_candidates = pair_candidates.sort_values(["priority", "row_id"])
    pair_representatives = (
        pair_candidates.drop_duplicates(
            ["dialect", "source_text_clean", "target_text_clean"], keep="first"
        )["row_id"]
        .tolist()
    )
    result.loc[
        result["row_id"].isin(pair_representatives),
        "normalization_representative",
    ] = True

    result["identification_representative"] = False
    id_candidates = result[
        (~result["derived_archive"])
        & (result["source_text_clean"] != "")
    ].copy()
    id_candidates["priority"] = id_candidates["dataset"].map(
        {
            "Vashantor": 0,
            "ChatgaiyyaAlap": 0,
            "Sylheti1200": 0,
            "BanglaRegionalTextCorpus": 0,
            "BanglaDial": 1,
        }
    ).fillna(2)
    id_candidates = id_candidates.sort_values(["priority", "row_id"])
    id_representatives = (
        id_candidates.drop_duplicates(
            ["dialect", "source_text_clean"], keep="first"
        )["row_id"]
        .tolist()
    )
    result.loc[
        result["row_id"].isin(id_representatives),
        "identification_representative",
    ] = True
    return result


def assign_eligibility(frame: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    split_map = components.set_index("exact_component_id")[
        "inherited_split"
    ].to_dict()
    result["split_preliminary_inherited"] = result[
        "exact_component_id"
    ].map(split_map)
    result["split_final_preliminary"] = result[
        "split_preliminary_inherited"
    ]
    result.loc[
        result["derived_archive"], "split_final_preliminary"
    ] = "quarantine"

    quality_block = result["quality_flags"].map(
        lambda flags: any(
            flag
            in {
                "empty_after_cleaning",
                "angle_placeholder",
                "punctuation_or_placeholder_only",
                "no_unicode_letter",
                "extreme_length_over_1000_chars",
                "target_empty_after_cleaning",
                "target_angle_placeholder",
                "target_punctuation_or_placeholder_only",
                "target_no_unicode_letter",
                "target_extreme_length_over_1000_chars",
                "source_maps_multiple_targets",
            }
            for flag in flags
        )
    )

    result["eligible_normalization_task"] = (
        (~result["derived_archive"])
        & (result["text_role"] == "regional_to_standard_pair")
        & (~quality_block)
        & result["normalization_representative"]
    )
    result["eligible_identification_task"] = (
        (~result["derived_archive"])
        & (~quality_block)
        & (~result["classification_label_conflict"])
        & result["identification_representative"]
    )
    result["eligible_normalization_train_preliminary"] = (
        result["eligible_normalization_task"]
        & (result["split_final_preliminary"] == "train")
    )
    result["eligible_identification_train_preliminary"] = (
        result["eligible_identification_task"]
        & (result["split_final_preliminary"] == "train")
        & (result["dataset"] != "BanglaDial")
    )
    result["eligible_tokenizer_train_preliminary"] = (
        (result["split_final_preliminary"] == "train")
        & (~result["derived_archive"])
        & (~quality_block)
        & (result["dataset"] != "BanglaDial")
        & (
            result["eligible_normalization_task"]
            | result["eligible_identification_task"]
        )
    )

    block_reasons = []
    for _, row in result.iterrows():
        reasons = []
        if row["derived_archive"]:
            reasons.append("derived_archive_quarantine")
        if row["split_preliminary_inherited"] != "train":
            reasons.append(
                f"nontrain_component_{row['split_preliminary_inherited']}"
            )
        if quality_block.loc[row.name]:
            reasons.append("obvious_quality_block")
        if row["classification_label_conflict"]:
            reasons.append("exact_cross_label_conflict")
        if (
            row["text_role"] == "regional_to_standard_pair"
            and not row["normalization_representative"]
        ):
            reasons.append("duplicate_normalization_nonrepresentative")
        if not row["identification_representative"]:
            reasons.append("duplicate_identification_nonrepresentative")
        if row["dataset"] == "BanglaDial":
            reasons.append("bangladial_pending_component_provenance")
        block_reasons.append(unique_sorted(reasons))
    result["preliminary_train_block_reasons"] = block_reasons

    for column in [
        "quality_flags",
        "preliminary_train_block_reasons",
    ]:
        result[column] = result[column].map(compact_json)
    return result


def data_gate_report(
    frame: pd.DataFrame,
    components: pd.DataFrame,
    raw_sha256: str,
    derived_sha256: str,
) -> dict[str, Any]:
    raw = frame[~frame["derived_archive"]]
    derived = frame[frame["derived_archive"]]
    pair = raw[raw["text_role"] == "regional_to_standard_pair"]
    bangladial = raw[raw["dataset"] == "BanglaDial"]
    protected_component_ids = set(
        components[components["contains_test_or_validation"]][
            "exact_component_id"
        ]
    )
    train_rows_in_protected_components = frame[
        frame["exact_component_id"].isin(protected_component_ids)
        & (
            frame["eligible_normalization_train_preliminary"]
            | frame["eligible_identification_train_preliminary"]
            | frame["eligible_tokenizer_train_preliminary"]
        )
    ]
    split_conflicts = components[components["contains_split_conflict"]]
    return {
        "gate_version": MANIFEST_VERSION,
        "status": "RED_REMAINING_GATES",
        "archive_hashes_verified": {
            RAW_ARCHIVE: raw_sha256 == RAW_ARCHIVE_SHA256,
            DERIVED_ARCHIVE: derived_sha256 == DERIVED_ARCHIVE_SHA256,
        },
        "counts": {
            "all_manifest_rows": len(frame),
            "raw_rows": len(raw),
            "derived_rows": len(derived),
            "aligned_pair_rows": len(pair),
            "bangladial_rows": len(bangladial),
            "exact_components": len(components),
            "split_conflict_components": len(split_conflicts),
            "source_maps_multiple_targets_rows": int(
                frame["quality_flags"].str.contains(
                    "source_maps_multiple_targets", regex=False
                ).sum()
            ),
            "angle_placeholder_rows": int(
                frame["quality_flags"].str.contains(
                    "angle_placeholder", regex=False
                ).sum()
            ),
            "bangladial_rows_with_exact_vashantor_match": int(
                bangladial["exact_component_matches"].str.contains(
                    '"dataset":"Vashantor"', regex=False
                ).sum()
            ),
            "derived_rows_train_eligible": int(
                (
                    derived["eligible_normalization_train_preliminary"]
                    | derived["eligible_identification_train_preliminary"]
                    | derived["eligible_tokenizer_train_preliminary"]
                ).sum()
            ),
            "train_rows_in_protected_components": len(
                train_rows_in_protected_components
            ),
            "normalization_task_eligible": int(
                frame["eligible_normalization_task"].sum()
            ),
            "normalization_train_preliminary": int(
                frame["eligible_normalization_train_preliminary"].sum()
            ),
            "identification_task_eligible": int(
                frame["eligible_identification_task"].sum()
            ),
            "identification_train_preliminary_non_bangladial": int(
                frame["eligible_identification_train_preliminary"].sum()
            ),
            "tokenizer_train_rows_preliminary": int(
                frame["eligible_tokenizer_train_preliminary"].sum()
            ),
        },
        "raw_split_counts": raw.groupby("split_final_preliminary")
        .size()
        .sort_index()
        .to_dict(),
        "normalization_train_by_dialect": frame[
            frame["eligible_normalization_train_preliminary"]
        ].groupby("dialect").size().sort_index().to_dict(),
        "identification_task_by_dialect": frame[
            frame["eligible_identification_task"]
        ].groupby("dialect").size().sort_index().to_dict(),
        "remaining_required_gates": {
            "near_duplicate_components": False,
            "template_components": False,
            "bangladial_original_component_reconstruction": False,
            "license_ledger": False,
            "human_quality_audit": False,
            "external_ood_test_frozen": False,
            "general_bangla_pretraining_manifest": False,
        },
        "passed_preliminary_checks": {
            "source_archive_counts": len(pair) == 22364
            and len(bangladial) == 63303
            and len(derived) == 51101,
            "derived_default_quarantine": int(
                (
                    derived["eligible_normalization_train_preliminary"]
                    | derived["eligible_identification_train_preliminary"]
                    | derived["eligible_tokenizer_train_preliminary"]
                ).sum()
            )
            == 0,
            "protected_exact_component_train_leakage_zero": len(
                train_rows_in_protected_components
            )
            == 0,
            "normalization_dialects_eight": set(
                pair["dialect"].unique()
            )
            == {"BAR", "CHI", "KHU", "MYM", "NAR", "NOA", "RAN", "SYL"},
        },
        "warning": (
            "Preliminary eligibility is not permission to train the final "
            "model. The remaining gates must all be true and a final manifest "
            "hash must be frozen."
        ),
    }


def render_gate_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Preliminary Data Gate",
        "",
        f"Status: {report['status']}",
        "",
        "The exact-match build is reproducible and internally consistent, but "
        "these artifacts are not the final training manifest.",
        "",
        "## Counts",
        "",
        "| Item | Count |",
        "|---|---:|",
    ]
    for key in [
        "all_manifest_rows",
        "raw_rows",
        "aligned_pair_rows",
        "bangladial_rows",
        "derived_rows",
        "exact_components",
        "normalization_task_eligible",
        "normalization_train_preliminary",
        "identification_task_eligible",
        "identification_train_preliminary_non_bangladial",
        "source_maps_multiple_targets_rows",
        "angle_placeholder_rows",
        "bangladial_rows_with_exact_vashantor_match",
        "derived_rows_train_eligible",
        "train_rows_in_protected_components",
    ]:
        lines.append(f"| {key} | {counts[key]:,} |")

    lines += [
        "",
        "## Preliminary normalization train rows",
        "",
        "| Dialect | Rows |",
        "|---|---:|",
    ]
    for dialect, count in report["normalization_train_by_dialect"].items():
        lines.append(f"| {dialect} | {count:,} |")

    lines += [
        "",
        "## Passed checks",
        "",
    ]
    for key, value in report["passed_preliminary_checks"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")

    lines += [
        "",
        "## Remaining gates",
        "",
    ]
    for key, value in report["remaining_required_gates"].items():
        lines.append(f"- {key}: {'PASS' if value else 'NOT COMPLETE'}")

    lines += [
        "",
        "## Deterministic artifacts",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
    ]
    for key, artifact in report["artifacts"].items():
        lines.append(f"| {key} | {artifact['sha256']} |")

    lines += [
        "",
        "## Decision",
        "",
        "Do not start the final tokenizer or model from this preliminary "
        "manifest. Complete near/template deduplication, component provenance, "
        "licenses, human review, the OOD test, and the general Bangla corpus; "
        "then freeze a new final manifest hash.",
        "",
    ]
    return "\n".join(lines)


def build_manifest(workspace: Path, output_dir: Path, report_dir: Path) -> dict[str, Any]:
    raw_path = workspace / RAW_ARCHIVE
    derived_path = workspace / DERIVED_ARCHIVE
    raw_sha256 = sha256_file(raw_path)
    derived_sha256 = sha256_file(derived_path)
    if raw_sha256 != RAW_ARCHIVE_SHA256:
        raise ValueError(f"Unexpected hash for {raw_path}: {raw_sha256}")
    if derived_sha256 != DERIVED_ARCHIVE_SHA256:
        raise ValueError(f"Unexpected hash for {derived_path}: {derived_sha256}")

    pairs, bangladial, _, _ = load_raw_datasets(raw_path)
    derived, _ = load_derived_archive(derived_path)
    records = []
    records.extend(build_pair_records(pairs, raw_sha256))
    records.extend(
        build_bangladial_records(
            bangladial,
            bangladial_member_name(raw_path),
            raw_sha256,
        )
    )
    records.extend(build_derived_records(derived, derived_sha256))
    frame = pd.DataFrame(records).reset_index(drop=True)
    frame = add_pair_mapping_conflict_flags(frame)
    frame = add_bangladial_match_provenance(frame)
    frame, components = build_exact_components(frame)
    frame = mark_conflicts_and_representatives(frame)
    frame = assign_eligibility(frame, components)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "canonical_rows_preliminary.parquet"
    components_path = output_dir / "exact_components_preliminary.parquet"
    quarantine_path = output_dir / "quarantined_rows_preliminary.parquet"
    exclusions_path = output_dir / "blocked_rows_preliminary.parquet"
    gate_path = report_dir / "data_gate_precheck.json"
    gate_markdown_path = report_dir / "PRELIMINARY_DATA_GATE.md"

    frame.to_parquet(canonical_path, index=False)
    components.to_parquet(components_path, index=False)
    frame[frame["derived_archive"]].to_parquet(quarantine_path, index=False)
    frame[
        frame["preliminary_train_block_reasons"] != "[]"
    ].to_parquet(exclusions_path, index=False)

    report = data_gate_report(
        frame,
        components,
        raw_sha256,
        derived_sha256,
    )
    report["artifacts"] = {
        "canonical_rows": {
            "path": str(canonical_path.resolve()),
            "sha256": sha256_file(canonical_path),
        },
        "exact_components": {
            "path": str(components_path.resolve()),
            "sha256": sha256_file(components_path),
        },
        "quarantined_rows": {
            "path": str(quarantine_path.resolve()),
            "sha256": sha256_file(quarantine_path),
        },
        "blocked_rows": {
            "path": str(exclusions_path.resolve()),
            "sha256": sha256_file(exclusions_path),
        },
    }
    gate_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    gate_markdown_path.write_text(
        render_gate_markdown(report),
        encoding="utf-8",
    )
    print(canonical_path)
    print(components_path)
    print(quarantine_path)
    print(exclusions_path)
    print(gate_path)
    print(gate_markdown_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "manifests",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports",
    )
    args = parser.parse_args()
    report = build_manifest(
        args.workspace.resolve(),
        args.output_dir.resolve(),
        args.report_dir.resolve(),
    )
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
