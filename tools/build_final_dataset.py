#!/usr/bin/env python3
"""Build the versioned Boichitro dialect dataset and task manifests.

The build keeps four concepts separate:

* authentic parallel training/development data;
* source-held-out and romanized evaluation data;
* authentic dialect-identification transcripts/text;
* traceable, train-only robustness augmentation.

All raw text is retained. Unicode, clean, model, and compact-dedup levels are
materialized separately so normalization is auditable rather than destructive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import pandas as pd
import yaml

from audit_local_archives import DIALECT_MAP, normalize_text, read_xlsx_first_sheet


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
BUILD_CONFIG = PROJECT / "configs" / "dataset_build.yaml"
EXTERNAL_CONFIG = PROJECT / "configs" / "external_sources.yaml"
PRELIMINARY = PROJECT / "data" / "manifests" / "canonical_rows_preliminary.parquet"
EXTERNAL = PROJECT / "data" / "external"
FINAL_ROOT = PROJECT / "data" / "final" / "v1"
MANIFEST_ROOT = PROJECT / "data" / "manifests"
REPORT_ROOT = PROJECT / "reports"

REGIONAL_LABELS = [
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
]
ALL_LABELS = REGIONAL_LABELS + ["STD"]
SPLIT_RANK = {
    "test_external": 0,
    "test_ood": 1,
    "test": 2,
    "validation": 3,
    "train": 4,
}

LOCAL_SOURCE_META = {
    "Vashantor": {
        "source_id": "local_vashantor_v2",
        "provider": "Mendeley/local ZIP",
        "doi": "10.17632/bj5jgk878b.2",
        "license": "CC BY 4.0",
        "quality_tier": "A_published_parallel",
    },
    "ChatgaiyyaAlap": {
        "source_id": "local_chatgaiyyaalap_v1",
        "provider": "Mendeley/local ZIP",
        "doi": "10.17632/wtms9xbkkw.1",
        "license": "CC BY 4.0",
        "quality_tier": "A_published_parallel",
    },
    "Sylheti1200": {
        "source_id": "local_sylheti_translation_v2",
        "provider": "Mendeley/local ZIP",
        "doi": "10.17632/5rmskrvh6g.2",
        "license": "CC BY 4.0",
        "quality_tier": "A_published_parallel",
    },
    "BanglaRegionalTextCorpus": {
        "source_id": "local_regional_text_corpus_v4",
        "provider": "Mendeley/local ZIP",
        "doi": "10.17632/92r62h4k5k.4",
        "license": "CC BY 4.0",
        "quality_tier": "A_published_parallel",
    },
}

EXTERNAL_SOURCE_META = {
    "kothon_v4": {
        "provider": "Mendeley",
        "doi": "10.17632/2fv6vf9v2z.4",
        "license": "CC BY 4.0",
        "quality_tier": "A_native_validated_parallel",
    },
    "sylheti_translation_v3_novel": {
        "provider": "Mendeley",
        "doi": "10.17632/5rmskrvh6g.3",
        "license": "CC BY 4.0",
        "quality_tier": "B_curated_lexical_and_parallel",
    },
    "chattogramsent_v2_novel": {
        "provider": "Mendeley",
        "doi": "10.17632/k6hts2ktxw.2",
        "license": "CC BY 4.0",
        "quality_tier": "A_native_validated_parallel",
    },
    "onubad_v2": {
        "provider": "Mendeley",
        "doi": "10.17632/6ft99kf89b.2",
        "license": "CC BY 4.0",
        "quality_tier": "A_native_annotated_ood",
    },
    "bd_dialect_v2": {
        "provider": "Mendeley",
        "doi": "10.17632/k769s4vk5z.2",
        "license": "CC BY 4.0",
        "quality_tier": "A_native_validated_ood",
    },
    "bhasabodh_v1": {
        "provider": "Mendeley",
        "doi": "10.17632/2jb4k7bb8x.1",
        "license": "CC BY 4.0",
        "quality_tier": "A_romanized_ood",
    },
}


def stable_hash(*parts: Any) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def normalized_levels(value: Any, *, transcript: bool = False) -> dict[str, str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raw = ""
    else:
        raw = str(value)
    nfc = unicodedata.normalize("NFC", raw)
    nfc = (
        nfc.replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u00a0", " ")
    )
    clean_chars = []
    for character in nfc:
        category = unicodedata.category(character)
        clean_chars.append(" " if category.startswith("C") else character)
    clean = re.sub(r"\s+", " ", "".join(clean_chars)).strip()
    model = clean
    if transcript:
        model = model.replace("<>", " ")
    model = re.sub(r'^["“”„]+|["“”„]+$', "", model).strip()
    model = model.translate(str.maketrans({"–": "—", "−": "—", "‘": "'", "’": "'"}))
    model = re.sub(r"\s+([।?!,.;:])", r"\1", model)
    model = re.sub(r"\s+", " ", model).strip()
    compact = "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", model)
        if unicodedata.category(character)[0] in {"L", "N"}
    )
    return {"raw": raw, "nfc": nfc, "clean": clean, "model": model, "compact": compact}


def bengali_letter_count(text: str) -> int:
    return sum(
        unicodedata.category(character).startswith("L")
        and "\u0980" <= character <= "\u09ff"
        for character in text
    )


def letter_count(text: str) -> int:
    return sum(unicodedata.category(character).startswith("L") for character in text)


def bengali_token_count(text: str) -> int:
    return sum(bool(re.search(r"[\u0980-\u09ff]", token)) for token in text.split())


def text_quality_flags(
    levels: dict[str, str],
    quality: dict[str, Any],
    *,
    transcript: bool = False,
) -> list[str]:
    text = levels["model"]
    flags: list[str] = []
    if not text:
        flags.append("empty_after_model_cleaning")
        return flags
    if len(text) > int(quality["maximum_characters"]):
        flags.append("extreme_length")
    bengali = bengali_letter_count(text)
    letters = letter_count(text)
    if not bengali:
        flags.append("no_bengali_letter")
    if letters and bengali / letters < float(quality["minimum_bengali_letter_ratio"]):
        flags.append("low_bengali_letter_ratio")
    if "<" in levels["clean"] or ">" in levels["clean"]:
        if transcript and "<>" in levels["clean"]:
            flags.append("transcript_inaudible_marker_removed")
            residual = levels["clean"].replace("<>", "")
            if "<" in residual or ">" in residual:
                flags.append("residual_angle_placeholder")
        else:
            flags.append("angle_placeholder")
    if transcript and bengali_token_count(text) < int(
        quality["minimum_bengali_tokens_transcript"]
    ):
        flags.append("transcript_too_short")
    return sorted(set(flags))


def fatal_flags(flags: Iterable[str]) -> list[str]:
    fatal = {
        "empty_after_model_cleaning",
        "extreme_length",
        "no_bengali_letter",
        "low_bengali_letter_ratio",
        "angle_placeholder",
        "residual_angle_placeholder",
        "transcript_too_short",
    }
    return sorted(set(flags) & fatal)


def granularity(source_text: str, target_text: str, sentence_minimum: int) -> str:
    maximum = max(bengali_token_count(source_text), bengali_token_count(target_text))
    if maximum <= 1:
        return "word"
    if maximum < sentence_minimum:
        return "phrase"
    return "sentence"


def deterministic_split(key: str, split_config: dict[str, float], seed: int) -> str:
    value = int(stable_hash(seed, key)[:16], 16) / float(16**16)
    if value < float(split_config["train"]):
        return "train"
    if value < float(split_config["train"]) + float(split_config["validation"]):
        return "validation"
    return "test"


def choose_protected_split(values: Iterable[str]) -> str:
    clean = [value for value in values if value in SPLIT_RANK]
    return min(clean, key=lambda value: SPLIT_RANK[value]) if clean else "train"


def make_pair_record(
    *,
    dataset_version: str,
    taxonomy_version: str,
    quality_config: dict[str, Any],
    source_id: str,
    provider: str,
    doi: str,
    license_name: str,
    quality_tier: str,
    source_row_id: str,
    dialect: str,
    source_text: Any,
    target_text: Any,
    split: str,
    split_origin: str,
    evaluation_track: str,
    source_romanized: Any = "",
    target_romanized: Any = "",
    english_text: Any = "",
    sentiment: Any = "",
    domain: Any = "",
    provenance: dict[str, Any] | None = None,
    inherited_flags: Iterable[str] = (),
) -> dict[str, Any]:
    source = normalized_levels(source_text)
    target = normalized_levels(target_text)
    source_flags = text_quality_flags(source, quality_config)
    target_flags = [
        f"target_{flag}" for flag in text_quality_flags(target, quality_config)
    ]
    flags = sorted(set(inherited_flags) | set(source_flags) | set(target_flags))
    if source["model"] == target["model"]:
        flags.append("source_equals_target")
    row_granularity = granularity(
        source["model"], target["model"], int(quality_config["sentence_minimum_tokens"])
    )
    semantic_group = stable_hash("standard_target", target["model"])
    row_id = stable_hash(
        dataset_version,
        "normalization",
        source_id,
        source_row_id,
        dialect,
        source["model"],
        target["model"],
    )
    return {
        "row_id": row_id,
        "dataset_version": dataset_version,
        "taxonomy_version": taxonomy_version,
        "task": "normalization",
        "source_id": source_id,
        "provider": provider,
        "doi": doi,
        "license": license_name,
        "source_row_id": str(source_row_id),
        "dialect": dialect,
        "source_text_raw": source["raw"],
        "source_text_nfc": source["nfc"],
        "source_text_clean": source["clean"],
        "source_text_model": source["model"],
        "source_text_compact": source["compact"],
        "target_text_raw": target["raw"],
        "target_text_nfc": target["nfc"],
        "target_text_clean": target["clean"],
        "target_text_model": target["model"],
        "target_text_compact": target["compact"],
        "source_romanized": normalize_text(source_romanized),
        "target_romanized": normalize_text(target_romanized),
        "english_text": normalize_text(english_text),
        "sentiment": normalize_text(sentiment),
        "domain": normalize_text(domain),
        "granularity": row_granularity,
        "semantic_group_id": semantic_group,
        "split": split,
        "split_origin": split_origin,
        "evaluation_track": evaluation_track,
        "quality_tier": quality_tier,
        "quality_flags": compact_json(sorted(set(flags))),
        "fatal_quality_flags": compact_json(
            fatal_flags(source_flags)
            + [f"target_{flag}" for flag in fatal_flags([x.removeprefix("target_") for x in target_flags])]
        ),
        "is_synthetic": False,
        "parent_row_id": "",
        "synthetic_method": "",
        "normalization_eligible": not bool(
            fatal_flags(source_flags)
            or fatal_flags([x.removeprefix("target_") for x in target_flags])
        ),
        "identification_eligible": True,
        "train_eligible": split == "train",
        "eval_eligible": split != "train" and row_granularity == "sentence",
        "sampling_weight": 1.0,
        "example_loss_weight": 1.0,
        "provenance": compact_json(provenance or {}),
    }


def exclusion_record(task: str, row: dict[str, Any], reason: str, detail: str = "") -> dict[str, Any]:
    return {
        "exclusion_id": stable_hash(task, row.get("row_id", ""), reason, detail),
        "task": task,
        "row_id": str(row.get("row_id", "")),
        "source_id": str(row.get("source_id", "")),
        "source_row_id": str(row.get("source_row_id", "")),
        "dialect": str(row.get("dialect", "")),
        "split_candidate": str(row.get("split", "")),
        "reason": reason,
        "detail": detail,
        "source_text_model": str(row.get("source_text_model", "")),
        "target_text_model": str(row.get("target_text_model", "")),
    }


def load_local_parallel(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = pd.read_parquet(PRELIMINARY)
    rows = rows[
        (~rows["derived_archive"])
        & rows["text_role"].eq("regional_to_standard_pair")
    ].copy()
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        meta = LOCAL_SOURCE_META[str(row["dataset"])]
        record = make_pair_record(
            dataset_version=config["dataset_version"],
            taxonomy_version=config["taxonomy_version"],
            quality_config=config["quality"],
            source_id=meta["source_id"],
            provider=meta["provider"],
            doi=meta["doi"],
            license_name=meta["license"],
            quality_tier=meta["quality_tier"],
            source_row_id=str(row["row_id"]),
            dialect=str(row["dialect"]),
            source_text=row["source_text_raw"],
            target_text=row["target_text_raw"],
            split=str(row["split_final_preliminary"]),
            split_origin="local_preliminary_component_split",
            evaluation_track=(
                "vashantor_published_iid"
                if str(row["dataset"]) == "Vashantor"
                else "local_group_iid"
            ),
            source_romanized=row["source_romanized"],
            target_romanized=row["target_romanized"],
            english_text=row["english_text"],
            inherited_flags=parse_json_list(row["quality_flags"]),
            provenance={
                "archive_filename": row["archive_filename"],
                "archive_sha256": row["archive_sha256"],
                "archive_member": row["archive_member"],
                "preliminary_row_id": row["row_id"],
                "preliminary_exact_component_id": row["exact_component_id"],
            },
        )
        if not bool(row["eligible_normalization_task"]):
            excluded.append(
                exclusion_record(
                    "normalization", record, "failed_preliminary_local_normalization_gate"
                )
            )
        elif not record["normalization_eligible"]:
            excluded.append(exclusion_record("normalization", record, "fatal_text_quality"))
        else:
            records.append(record)
    return records, excluded


def load_training_external(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # Kothon: native-validated Chittagonian and Sylheti parallel workbooks.
    path = EXTERNAL / "raw" / "mendeley" / "kothon_v4" / "Chittagonian_Sylheti_dataset.zip"
    with ZipFile(path) as archive:
        specifications = [
            ("CHI", "Chittagonian.xlsx", "Chittagonian Dialect"),
            ("SYL", "Sylheti.xlsx", "Sylheti Dialect"),
        ]
        for dialect, suffix, source_column in specifications:
            member = next(name for name in archive.namelist() if name.endswith(suffix))
            frame = read_xlsx_first_sheet(archive.read(member))
            for index, row in frame.iterrows():
                records.append(
                    make_pair_record(
                        dataset_version=config["dataset_version"],
                        taxonomy_version=config["taxonomy_version"],
                        quality_config=config["quality"],
                        source_id="kothon_v4",
                        **{
                            "provider": EXTERNAL_SOURCE_META["kothon_v4"]["provider"],
                            "doi": EXTERNAL_SOURCE_META["kothon_v4"]["doi"],
                            "license_name": EXTERNAL_SOURCE_META["kothon_v4"]["license"],
                            "quality_tier": EXTERNAL_SOURCE_META["kothon_v4"]["quality_tier"],
                        },
                        source_row_id=f"{dialect}:{index}",
                        dialect=dialect,
                        source_text=row[source_column],
                        target_text=row["Standard Bangla"],
                        english_text=row["Translated English"],
                        split="candidate",
                        split_origin="pending_global_semantic_split",
                        evaluation_track="external_group_iid",
                        provenance={"archive_sha256": sha256_file(path), "member": member},
                    )
                )

    # Latest Sylheti release. Exact v2 rows are removed later against the local copy.
    path = (
        EXTERNAL
        / "raw"
        / "mendeley"
        / "sylheti_translation_v3"
        / "Bangla Dialect Transaction Dataset_v3.csv"
    )
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        meta = EXTERNAL_SOURCE_META["sylheti_translation_v3_novel"]
        records.append(
            make_pair_record(
                dataset_version=config["dataset_version"],
                taxonomy_version=config["taxonomy_version"],
                quality_config=config["quality"],
                source_id="sylheti_translation_v3_novel",
                provider=meta["provider"],
                doi=meta["doi"],
                license_name=meta["license"],
                quality_tier=meta["quality_tier"],
                source_row_id=str(index),
                dialect="SYL",
                source_text=row.iloc[0],
                target_text=row.iloc[1],
                split="candidate",
                split_origin="pending_global_semantic_split",
                evaluation_track="external_group_iid",
                provenance={"file_sha256": sha256_file(path)},
            )
        )

    # ChattogramSent contains a large exact ChatgaiyyaAlap ancestor. Only the
    # novel remainder survives global pair deduplication.
    path = (
        EXTERNAL
        / "raw"
        / "mendeley"
        / "chattogramsent_v2"
        / "ChattogramSent _ A Multilingual Sentiment Dataset for the Chattogram Dialect, Standard Bengali, and English full.csv"
    )
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        meta = EXTERNAL_SOURCE_META["chattogramsent_v2_novel"]
        records.append(
            make_pair_record(
                dataset_version=config["dataset_version"],
                taxonomy_version=config["taxonomy_version"],
                quality_config=config["quality"],
                source_id="chattogramsent_v2_novel",
                provider=meta["provider"],
                doi=meta["doi"],
                license_name=meta["license"],
                quality_tier=meta["quality_tier"],
                source_row_id=str(index),
                dialect="CHI",
                source_text=row["Chattogram"],
                target_text=row["Bengali"],
                english_text=row["English"],
                sentiment=row["Sentiment"],
                domain=row["Source of Data"],
                split="candidate",
                split_origin="pending_global_semantic_split",
                evaluation_track="external_group_iid_not_source_ood",
                provenance={"file_sha256": sha256_file(path)},
            )
        )
    return records


def load_ood_parallel(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # ONUBAD sentence track. Clause/word material is not mixed into this OOD set.
    path = EXTERNAL / "raw" / "mendeley" / "onubad_v2" / "ONUBAD Dataset.zip"
    with ZipFile(path) as archive:
        frame = read_xlsx_first_sheet(archive.read("Sentence.xlsx"))
    mapping = {
        "CHI": "Chittagong Language",
        "SYL": "Sylhet Language",
        "BAR": "Barisal Language",
    }
    meta = EXTERNAL_SOURCE_META["onubad_v2"]
    for index, row in frame.iterrows():
        for dialect, source_column in mapping.items():
            records.append(
                make_pair_record(
                    dataset_version=config["dataset_version"],
                    taxonomy_version=config["taxonomy_version"],
                    quality_config=config["quality"],
                    source_id="onubad_v2",
                    provider=meta["provider"],
                    doi=meta["doi"],
                    license_name=meta["license"],
                    quality_tier=meta["quality_tier"],
                    source_row_id=f"sentence:{index}:{dialect}",
                    dialect=dialect,
                    source_text=row[source_column],
                    target_text=row["Standard Bangla Lanuguage"],
                    english_text=row["English Translation"],
                    split="test_ood",
                    split_origin="entire_source_locked",
                    evaluation_track="onubad_source_ood_sentence",
                    provenance={"archive_sha256": sha256_file(path), "member": "Sentence.xlsx"},
                )
            )

    # BD-Dialect is kept entirely locked: clauses and word-level challenge.
    path = EXTERNAL / "raw" / "mendeley" / "bd_dialect_v2" / "BD-Dialect 02.zip"
    mapping = {
        "NOA": "Nowakhali Language",
        "SYL": "Sylheti Language",
        "CHI": "Chittagong Language",
        "RAJ": "Rajshahi Language",
        "MYM": "Mymensingh Language",
    }
    meta = EXTERNAL_SOURCE_META["bd_dialect_v2"]
    with ZipFile(path) as archive:
        for filename, track in [
            ("BD-Dialect_Clauses.csv", "bd_dialect_source_ood_clause"),
            ("BD-Dialect_Words.csv", "bd_dialect_source_ood_lexical"),
        ]:
            member = next(name for name in archive.namelist() if name.endswith(filename))
            frame = pd.read_csv(BytesIO(archive.read(member)))
            for index, row in frame.iterrows():
                for dialect, source_column in mapping.items():
                    records.append(
                        make_pair_record(
                            dataset_version=config["dataset_version"],
                            taxonomy_version=config["taxonomy_version"],
                            quality_config=config["quality"],
                            source_id="bd_dialect_v2",
                            provider=meta["provider"],
                            doi=meta["doi"],
                            license_name=meta["license"],
                            quality_tier=meta["quality_tier"],
                            source_row_id=f"{filename}:{index}:{dialect}",
                            dialect=dialect,
                            source_text=row[source_column],
                            target_text=row["Standard Bangla Languge"],
                            english_text=row["English Translation"],
                            split="test_ood",
                            split_origin="entire_source_locked",
                            evaluation_track=track,
                            provenance={"archive_sha256": sha256_file(path), "member": member},
                        )
                    )
    return records


def assign_external_splits(
    local: list[dict[str, Any]],
    external: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    target_splits: dict[str, list[str]] = defaultdict(list)
    for row in local:
        target_splits[row["target_text_model"]].append(row["split"])
    inherited = {
        target: choose_protected_split(splits) for target, splits in target_splits.items()
    }
    for row in external:
        # Single-word expansions are useful as lexical supervision but must not
        # become an easy, short-item evaluation set.
        if row["granularity"] == "word" and row["source_id"] == "sylheti_translation_v3_novel":
            row["split"] = "train"
            row["split_origin"] = "lexical_train_only"
            row["eval_eligible"] = False
        elif row["target_text_model"] in inherited:
            row["split"] = inherited[row["target_text_model"]]
            row["split_origin"] = "inherited_exact_standard_target_component"
        else:
            row["split"] = deterministic_split(
                row["semantic_group_id"], config["split"], int(config["seed"])
            )
            row["split_origin"] = "global_standard_target_hash"
        row["train_eligible"] = row["split"] == "train"
        row["eval_eligible"] = (
            row["split"] != "train" and row["granularity"] == "sentence"
        )


def resolve_pair_conflicts_and_duplicates(
    rows: list[dict[str, Any]],
    source_priority: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded: list[dict[str, Any]] = []
    eligible = []
    for row in rows:
        if not row["normalization_eligible"]:
            excluded.append(exclusion_record("normalization", row, "fatal_text_quality"))
        else:
            eligible.append(row)

    by_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_source[(row["dialect"], row["source_text_model"])].append(row)

    consistent: list[dict[str, Any]] = []
    for group in by_source.values():
        minimum_priority = min(source_priority.get(row["source_id"], 50) for row in group)
        best = [
            row
            for row in group
            if source_priority.get(row["source_id"], 50) == minimum_priority
        ]
        targets = {row["target_text_model"] for row in best}
        if len(targets) != 1:
            for row in group:
                excluded.append(
                    exclusion_record(
                        "normalization",
                        row,
                        "source_maps_multiple_targets_at_best_priority",
                        compact_json(sorted(targets)),
                    )
                )
            continue
        selected_target = next(iter(targets))
        for row in group:
            if row["target_text_model"] != selected_target:
                excluded.append(
                    exclusion_record(
                        "normalization",
                        row,
                        "conflicting_lower_priority_target",
                        selected_target,
                    )
                )
            else:
                consistent.append(row)

    by_pair: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in consistent:
        by_pair[
            (row["dialect"], row["source_text_model"], row["target_text_model"])
        ].append(row)
    accepted = []
    for group in by_pair.values():
        ordered = sorted(
            group,
            key=lambda row: (
                source_priority.get(row["source_id"], 50),
                SPLIT_RANK.get(row["split"], 99),
                row["row_id"],
            ),
        )
        accepted.append(ordered[0])
        for duplicate in ordered[1:]:
            excluded.append(
                exclusion_record(
                    "normalization",
                    duplicate,
                    "exact_pair_duplicate_lower_priority",
                    ordered[0]["row_id"],
                )
            )
    return accepted, excluded


@lru_cache(maxsize=None)
def ngram_set(compact: str, size: int = 4) -> frozenset[str]:
    if len(compact) <= size:
        return frozenset({compact}) if compact else frozenset()
    return frozenset(compact[index : index + size] for index in range(len(compact) - size + 1))


@lru_cache(maxsize=None)
def simhash64(compact: str, size: int = 4) -> int:
    grams = ngram_set(compact, size)
    if not grams:
        return 0
    vector = [0] * 64
    for gram in grams:
        value = int.from_bytes(
            hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    fingerprint = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def jaccard(left: str, right: str, size: int = 4) -> float:
    left_set = ngram_set(left, size)
    right_set = ngram_set(right, size)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def near_match_map(
    query_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    compact_field: str,
    text_field: str,
    config: dict[str, Any],
) -> dict[str, tuple[str, float, str]]:
    dedup = config["deduplication"]
    minimum_length = int(dedup["near_minimum_compact_characters"])
    threshold = float(dedup["near_jaccard_threshold"])
    max_hamming = int(dedup["simhash_max_hamming_candidate"])
    gram_size = int(dedup["simhash_character_ngram"])

    reference_by_compact: dict[str, dict[str, Any]] = {}
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        compact = row[compact_field]
        if not compact:
            continue
        reference_by_compact.setdefault(compact, row)
        if len(compact) < minimum_length:
            continue
        fingerprint = simhash64(compact, gram_size)
        for band in range(4):
            buckets[(band, (fingerprint >> (band * 16)) & 0xFFFF)].append(row)

    matches: dict[str, tuple[str, float, str]] = {}
    for row in query_rows:
        compact = row[compact_field]
        if not compact:
            continue
        exact = reference_by_compact.get(compact)
        if exact is not None:
            matches[row["row_id"]] = (exact["row_id"], 1.0, "compact_exact")
            continue
        if len(compact) < minimum_length:
            continue
        fingerprint = simhash64(compact, gram_size)
        candidates: dict[str, dict[str, Any]] = {}
        for band in range(4):
            for candidate in buckets.get(
                (band, (fingerprint >> (band * 16)) & 0xFFFF), []
            ):
                candidates[candidate["row_id"]] = candidate
        best: tuple[str, float, str] | None = None
        for candidate in candidates.values():
            other = candidate[compact_field]
            ratio = min(len(compact), len(other)) / max(len(compact), len(other))
            if ratio < 0.80:
                continue
            other_fingerprint = simhash64(other, gram_size)
            hamming = (fingerprint ^ other_fingerprint).bit_count()
            # Hamming is a candidate control; the final decision is an exact
            # character-ngram Jaccard threshold.
            if hamming > max_hamming:
                continue
            score = jaccard(compact, other, gram_size)
            if score >= threshold and (best is None or score > best[1]):
                best = (candidate["row_id"], score, f"simhash_jaccard:{text_field}")
        if best is not None:
            matches[row["row_id"]] = best
    return matches


def remove_train_protected_near_leakage(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in rows if row["split"] == "train"]
    protected = [row for row in rows if row["split"] in {"validation", "test"}]
    source_matches = near_match_map(
        train,
        protected,
        compact_field="source_text_compact",
        text_field="source",
        config=config,
    )
    target_matches = near_match_map(
        train,
        protected,
        compact_field="target_text_compact",
        text_field="target",
        config=config,
    )
    matches = {**source_matches, **target_matches}
    accepted = []
    excluded = []
    for row in rows:
        if row["row_id"] in matches:
            reference_id, score, method = matches[row["row_id"]]
            excluded.append(
                exclusion_record(
                    "normalization",
                    row,
                    "train_near_protected_evaluation",
                    compact_json(
                        {"reference_row_id": reference_id, "score": score, "method": method}
                    ),
                )
            )
        else:
            accepted.append(row)
    return accepted, excluded


def decontaminate_ood(
    ood_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # OOD rows are queries: overlapping benchmark items are removed, never
    # allowed to delete authentic development data.
    source_matches = near_match_map(
        ood_rows,
        development_rows,
        compact_field="source_text_compact",
        text_field="source",
        config=config,
    )
    target_matches = near_match_map(
        ood_rows,
        development_rows,
        compact_field="target_text_compact",
        text_field="target",
        config=config,
    )
    matches = {**source_matches, **target_matches}
    accepted = []
    excluded = []
    for row in ood_rows:
        if row["row_id"] in matches:
            reference_id, score, method = matches[row["row_id"]]
            excluded.append(
                exclusion_record(
                    "normalization",
                    row,
                    "ood_overlap_with_development",
                    compact_json(
                        {"reference_row_id": reference_id, "score": score, "method": method}
                    ),
                )
            )
        else:
            row["train_eligible"] = False
            row["eval_eligible"] = True
            accepted.append(row)
    return accepted, excluded


def make_synthetic_variant(text: str, selector: int) -> tuple[str, str]:
    terminal = re.compile(r"[।?!.,]+$")
    if selector % 3 == 0:
        if terminal.search(text):
            return terminal.sub("", text).strip(), "terminal_punctuation_variant"
        return f"{text} ।", "terminal_punctuation_variant"
    tokens = text.split()
    if selector % 3 == 1 and len(tokens) >= 3:
        position = selector % (len(tokens) - 1)
        tokens[position : position + 2] = [tokens[position] + tokens[position + 1]]
        return " ".join(tokens), "single_safe_boundary_join"
    return f"“{text}”", "quote_and_dash_typography_variant"


def allocate_synthetic(counts: dict[str, int], total: int) -> dict[str, int]:
    caps = {label: int(math.floor(value * 0.15)) for label, value in counts.items()}
    weights = {label: 1.0 / math.sqrt(max(value, 1)) for label, value in counts.items()}
    allocation = {label: 0 for label in counts}
    for _ in range(total):
        available = [label for label in counts if allocation[label] < caps[label]]
        if not available:
            break
        label = min(
            available,
            key=lambda item: (
                allocation[item] / weights[item],
                counts[item],
                item,
            ),
        )
        allocation[label] += 1
    return allocation


def build_synthetic(
    authentic_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in authentic_rows
        if row["split"] == "train"
        and row["granularity"] != "word"
        and bengali_token_count(row["source_text_model"]) >= 2
    ]
    counts = pd.Series([row["dialect"] for row in candidates]).value_counts().to_dict()
    requested = int(math.floor(len(candidates) * 0.10))
    allocation = allocate_synthetic({str(k): int(v) for k, v in counts.items()}, requested)
    # Robustness variants intentionally preserve lexical content, so compact
    # fingerprints may equal the parent. Uniqueness is enforced at the actual
    # model-input level while parent linkage makes that relationship explicit.
    existing = {row["source_text_model"] for row in authentic_rows}
    generated: list[dict[str, Any]] = []
    for dialect, amount in allocation.items():
        ordered = sorted(
            [row for row in candidates if row["dialect"] == dialect],
            key=lambda row: stable_hash(config["seed"], "synthetic", row["row_id"]),
        )
        accepted = 0
        for parent in ordered:
            if accepted >= amount:
                break
            selector = int(stable_hash(parent["row_id"], config["seed"])[:8], 16)
            variant_text, method = make_synthetic_variant(parent["source_text_model"], selector)
            levels = normalized_levels(variant_text)
            if not levels["model"] or levels["model"] in existing:
                continue
            row = dict(parent)
            row["row_id"] = stable_hash(
                config["dataset_version"], "synthetic", parent["row_id"], method
            )
            row["source_id"] = "synthetic_robustness_v1"
            row["provider"] = "deterministic_local_build"
            row["doi"] = parent["doi"]
            row["license"] = f"DERIVED_FROM_PARENT:{parent['license']}"
            row["source_row_id"] = parent["row_id"]
            for level in ["raw", "nfc", "clean", "model", "compact"]:
                row[f"source_text_{level}"] = levels[level]
            row["split"] = "train"
            row["split_origin"] = "synthetic_parent_train_only"
            row["evaluation_track"] = "none"
            row["quality_tier"] = "S_traceable_robustness_non_authentic"
            flags = parse_json_list(row["quality_flags"])
            flags.extend(["synthetic_non_authentic", method])
            row["quality_flags"] = compact_json(sorted(set(flags)))
            row["fatal_quality_flags"] = "[]"
            row["is_synthetic"] = True
            row["parent_row_id"] = parent["row_id"]
            row["synthetic_method"] = method
            row["normalization_eligible"] = True
            row["identification_eligible"] = False
            row["train_eligible"] = True
            row["eval_eligible"] = False
            row["example_loss_weight"] = float(
                config["sampling"]["synthetic_example_loss_weight"]
            )
            row["provenance"] = compact_json(
                {
                    "parent_row_id": parent["row_id"],
                    "parent_source_id": parent["source_id"],
                    "parent_split": parent["split"],
                    "method": method,
                    "authenticity_claim": False,
                }
            )
            existing.add(levels["model"])
            generated.append(row)
            accepted += 1
    return generated


def temperature_weights(
    rows: list[dict[str, Any]],
    label_field: str,
    config: dict[str, Any],
) -> None:
    train = [row for row in rows if row["split"] == "train"]
    counts = pd.Series([row[label_field] for row in train]).value_counts().to_dict()
    total = sum(counts.values())
    alpha = float(config["sampling"]["dialect_temperature_alpha"])
    minimum = float(config["sampling"]["minimum_weight"])
    maximum = float(config["sampling"]["maximum_weight"])
    raw = {
        label: (count / total) ** (alpha - 1.0) for label, count in counts.items()
    }
    denominator = sum(raw[label] * counts[label] for label in raw) / total
    normalized = {
        label: min(maximum, max(minimum, value / denominator))
        for label, value in raw.items()
    }
    for row in rows:
        row["sampling_weight"] = (
            normalized.get(row[label_field], 1.0) if row["split"] == "train" else 1.0
        )


def make_identification_record(
    *,
    config: dict[str, Any],
    source_id: str,
    provider: str,
    doi: str,
    license_name: str,
    source_row_id: str,
    dialect: str,
    text: Any,
    split: str,
    split_origin: str,
    evaluation_track: str,
    quality_tier: str,
    transcript: bool = False,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    levels = normalized_levels(text, transcript=transcript)
    flags = text_quality_flags(levels, config["quality"], transcript=transcript)
    row_id = stable_hash(
        config["dataset_version"],
        "identification",
        source_id,
        source_row_id,
        dialect,
        levels["model"],
    )
    return {
        "row_id": row_id,
        "dataset_version": config["dataset_version"],
        "taxonomy_version": config["taxonomy_version"],
        "task": "identification",
        "source_id": source_id,
        "provider": provider,
        "doi": doi,
        "license": license_name,
        "source_row_id": str(source_row_id),
        "dialect": dialect,
        "text_raw": levels["raw"],
        "text_nfc": levels["nfc"],
        "text_clean": levels["clean"],
        "text_model": levels["model"],
        "text_compact": levels["compact"],
        "split": split,
        "split_origin": split_origin,
        "evaluation_track": evaluation_track,
        "quality_tier": quality_tier,
        "quality_flags": compact_json(flags),
        "is_synthetic": False,
        "train_eligible": split == "train" and not fatal_flags(flags),
        "eval_eligible": split != "train" and not fatal_flags(flags),
        "sampling_weight": 1.0,
        "provenance": compact_json(provenance or {}),
        "fatal": bool(fatal_flags(flags)),
    }


def build_identification(
    normalization_authentic: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    # Every authentic parallel row contributes its regional side, and one
    # Standard-Bangla candidate from its target. Exact dedup later prevents the
    # standard class from being inflated by repeated parallel targets.
    for row in normalization_authentic:
        common = {
            "config": config,
            "provider": row["provider"],
            "doi": row["doi"],
            "license_name": row["license"],
            "split": row["split"],
            "split_origin": f"normalization_component:{row['split_origin']}",
            "evaluation_track": row["evaluation_track"],
            "quality_tier": row["quality_tier"],
        }
        candidates.append(
            make_identification_record(
                **common,
                source_id=f"{row['source_id']}:regional_side",
                source_row_id=row["row_id"],
                dialect=row["dialect"],
                text=row["source_text_raw"],
                provenance={"normalization_row_id": row["row_id"], "text_role": "regional"},
            )
        )
        candidates.append(
            make_identification_record(
                **common,
                source_id=f"{row['source_id']}:standard_target",
                source_row_id=row["row_id"],
                dialect="STD",
                text=row["target_text_raw"],
                provenance={"normalization_row_id": row["row_id"], "text_role": "standard"},
            )
        )

    # Build a split-inheritance index before adding merged classification data.
    normalization_splits: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        normalization_splits[row["text_model"]].append(row["split"])
    inherited_split = {
        text: choose_protected_split(splits)
        for text, splits in normalization_splits.items()
    }

    path = (
        EXTERNAL
        / "raw"
        / "mendeley"
        / "bangladial_v2"
        / "BanglaDial A Merged and Imbalanced text Dataset for Bengali Regional dialect analysis..csv"
    )
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        raw_label = normalize_text(row["label"]).lower()
        dialect = DIALECT_MAP.get(raw_label)
        if dialect is None:
            stub = {"row_id": stable_hash("bangladial_unmapped", index), "source_id": "bangladial_v2", "source_row_id": index, "dialect": raw_label}
            excluded.append(exclusion_record("identification", stub, "unmapped_bangladial_label"))
            continue
        levels = normalized_levels(row["sentence"])
        if levels["model"] in inherited_split:
            split = inherited_split[levels["model"]]
            origin = "inherited_normalization_text_component"
        else:
            split = deterministic_split(
                stable_hash("bangladial_v2", dialect, levels["model"]),
                config["split"],
                int(config["seed"]),
            )
            origin = "merged_text_hash_iid"
        candidates.append(
            make_identification_record(
                config=config,
                source_id="bangladial_v2",
                provider="Mendeley",
                doi="10.17632/sx6ybcps2n.2",
                license_name="CC BY 4.0",
                source_row_id=str(index),
                dialect=dialect,
                text=row["sentence"],
                split=split,
                split_origin=origin,
                evaluation_track="bangladial_merged_iid",
                quality_tier="B_merged_unknown_component_provenance",
                provenance={"file_sha256": sha256_file(path), "original_label": row["label"]},
            )
        )

    hf_path = (
        EXTERNAL
        / "processed"
        / "huggingface"
        / "hf_bengali_regional_asr_refine"
        / "transcripts_text_only.parquet"
    )
    hf = pd.read_parquet(hf_path)
    label_map = {
        "barishal": "BAR",
        "chittagong": "CHI",
        "kishoreganj": "KIS",
        "narail": "NAR",
        "narsingdi": "NSD",
        "rangpur": "RAN",
        "sylhet": "SYL",
        "tangail": "TAN",
    }
    for index, row in hf.iterrows():
        district = str(row["district"])
        if district not in label_map:
            stub = {
                "row_id": stable_hash("hf_unmapped", row["upstream_shard"], row["upstream_row_index"]),
                "source_id": "hf_bengali_regional_asr_refine",
                "source_row_id": f"{row['upstream_shard']}:{row['upstream_row_index']}",
                "dialect": district,
            }
            excluded.append(
                exclusion_record(
                    "identification", stub, "hf_district_outside_frozen_taxonomy"
                )
            )
            continue
        split = "test_external" if row["split_original"] == "eval" else "train"
        candidates.append(
            make_identification_record(
                config=config,
                source_id="hf_bengali_regional_asr_refine",
                provider="Hugging Face",
                doi="",
                license_name="apache-2.0",
                source_row_id=f"{row['upstream_shard']}:{row['upstream_row_index']}",
                dialect=label_map[district],
                text=row["transcriptions"],
                split=split,
                split_origin="upstream_asr_split_preserved",
                evaluation_track="hf_regional_asr_transcript_external",
                quality_tier="A_authentic_speech_transcript_conditional_license",
                transcript=True,
                provenance={
                    "repo_id": "sha1779/bengali_regional_dataset_refine",
                    "revision": "6e3221bdf7f9c8c426276f1b619d7de597963f42",
                    "upstream_shard": row["upstream_shard"],
                    "upstream_row_index": int(row["upstream_row_index"]),
                    "district": district,
                },
            )
        )

    valid = []
    for row in candidates:
        if row["fatal"]:
            excluded.append(exclusion_record("identification", row, "fatal_text_quality"))
        else:
            valid.append(row)

    # Remove exact and formatting-equivalent cross-label ambiguity globally.
    conflict_ids: set[str] = set()
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_model[row["text_model"]].append(row)
    for group in by_model.values():
        if len({row["dialect"] for row in group}) > 1:
            conflict_ids.update(row["row_id"] for row in group)
    by_compact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_compact[row["text_compact"]].append(row)
    for group in by_compact.values():
        if len({row["dialect"] for row in group}) > 1:
            conflict_ids.update(row["row_id"] for row in group)
    nonconflicting = []
    for row in valid:
        if row["row_id"] in conflict_ids:
            excluded.append(
                exclusion_record("identification", row, "cross_label_text_conflict")
            )
        else:
            nonconflicting.append(row)

    source_priority = {
        "hf_bengali_regional_asr_refine": 0,
        "bangladial_v2": 3,
    }

    def id_source_priority(source_id: str) -> int:
        if source_id in source_priority:
            return source_priority[source_id]
        if source_id.endswith(":regional_side") or source_id.endswith(":standard_target"):
            return 1
        return 2

    # One representative per label and compact text. The protected split wins.
    representatives = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in nonconflicting:
        grouped[(row["dialect"], row["text_compact"])].append(row)
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda row: (
                SPLIT_RANK.get(row["split"], 99),
                id_source_priority(row["source_id"]),
                row["row_id"],
            ),
        )
        representatives.append(ordered[0])
        for duplicate in ordered[1:]:
            excluded.append(
                exclusion_record(
                    "identification",
                    duplicate,
                    "same_label_compact_duplicate",
                    ordered[0]["row_id"],
                )
            )

    train = [row for row in representatives if row["split"] == "train"]
    protected = [row for row in representatives if row["split"] != "train"]
    matches = near_match_map(
        train,
        protected,
        compact_field="text_compact",
        text_field="identification_text",
        config=config,
    )
    final = []
    for row in representatives:
        if row["row_id"] in matches:
            reference_id, score, method = matches[row["row_id"]]
            excluded.append(
                exclusion_record(
                    "identification",
                    {**row, "source_text_model": row["text_model"]},
                    "train_near_protected_evaluation",
                    compact_json(
                        {"reference_row_id": reference_id, "score": score, "method": method}
                    ),
                )
            )
        else:
            row.pop("fatal", None)
            row["train_eligible"] = row["split"] == "train"
            row["eval_eligible"] = row["split"] != "train"
            final.append(row)
    temperature_weights(final, "dialect", config)
    return final, excluded


def build_romanized_eval(
    accepted_ood: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    onubad_lookup = {
        (row["dialect"], row["source_text_model"], row["target_text_model"]): row
        for row in accepted_ood
        if row["source_id"] == "onubad_v2"
    }
    path = EXTERNAL / "raw" / "mendeley" / "bhasabodh_v1" / "BhashaBodh.csv"
    frame = pd.read_csv(path)
    specifications = [
        ("CHI", "Chittagong Language", "Romanized Chittagong Language"),
        ("SYL", "Sylhet Language", "Romanized Sylhet Language"),
    ]
    records = []
    excluded = []
    seen_parent: set[str] = set()
    for index, row in frame.iterrows():
        target = normalized_levels(row["Standard Bangla Lanuguage"])
        for dialect, bengali_column, romanized_column in specifications:
            bengali = normalized_levels(row[bengali_column])
            parent = onubad_lookup.get((dialect, bengali["model"], target["model"]))
            stub = {
                "row_id": stable_hash("bhasabodh", index, dialect),
                "source_id": "bhasabodh_v1",
                "source_row_id": index,
                "dialect": dialect,
                "source_text_model": normalize_text(row[romanized_column]),
                "target_text_model": target["model"],
            }
            if parent is None:
                excluded.append(
                    exclusion_record(
                        "romanized_normalization",
                        stub,
                        "bengali_parent_removed_from_onubad_ood",
                    )
                )
                continue
            if parent["row_id"] in seen_parent:
                excluded.append(
                    exclusion_record(
                        "romanized_normalization",
                        stub,
                        "duplicate_romanized_parent",
                        parent["row_id"],
                    )
                )
                continue
            romanized = normalized_levels(row[romanized_column])
            record = {
                "row_id": stable_hash(
                    config["dataset_version"], "romanized_ood", parent["row_id"], romanized["model"]
                ),
                "dataset_version": config["dataset_version"],
                "taxonomy_version": config["taxonomy_version"],
                "task": "romanized_normalization",
                "source_id": "bhasabodh_v1",
                "provider": "Mendeley",
                "doi": "10.17632/2jb4k7bb8x.1",
                "license": "CC BY 4.0",
                "source_row_id": str(index),
                "dialect": dialect,
                "romanized_input_raw": romanized["raw"],
                "romanized_input_model": romanized["model"],
                "bengali_reference": bengali["model"],
                "target_text_model": target["model"],
                "target_romanized": normalize_text(row["Romanized Bangla"]),
                "english_text": normalize_text(row["English Translation"]),
                "split": "test_ood",
                "evaluation_track": "bhasabodh_romanized_companion_to_onubad",
                "parent_normalization_row_id": parent["row_id"],
                "quality_tier": "A_romanized_ood",
                "eval_eligible": True,
                "provenance": compact_json(
                    {
                        "file_sha256": sha256_file(path),
                        "onubad_parent_row_id": parent["row_id"],
                        "shared_bengali_pairs_not_double_counted": True,
                    }
                ),
            }
            records.append(record)
            seen_parent.add(parent["row_id"])
    return records, excluded


def build_tokenizer_manifest(
    normalization_rows: list[dict[str, Any]],
    identification_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    def add(
        text: Any,
        source_id: str,
        source_row_id: str,
        dialect: str,
        license_name: str,
        role: str,
        weight: float,
        safety_sensitive: bool = False,
    ) -> None:
        levels = normalized_levels(text)
        candidates.append(
            {
                "row_id": stable_hash(
                    config["dataset_version"], "tokenizer", source_id, source_row_id, role, levels["model"]
                ),
                "dataset_version": config["dataset_version"],
                "source_id": source_id,
                "source_row_id": source_row_id,
                "dialect": dialect,
                "text_role": role,
                "text_raw": levels["raw"],
                "text_model": levels["model"],
                "text_compact": levels["compact"],
                "license": license_name,
                "sampling_weight": weight,
                "safety_sensitive": safety_sensitive,
            }
        )

    for row in normalization_rows:
        if row["split"] != "train" or row["is_synthetic"]:
            continue
        add(
            row["source_text_raw"], row["source_id"], row["row_id"], row["dialect"], row["license"], "regional", 1.0
        )
        add(
            row["target_text_raw"], row["source_id"], row["row_id"], "STD", row["license"], "standard", 1.0
        )
    for row in identification_rows:
        if row["split"] == "train":
            add(
                row["text_raw"], row["source_id"], row["row_id"], row["dialect"], row["license"], "identification", 0.75
            )

    lexicon_path = (
        EXTERNAL
        / "raw"
        / "huggingface"
        / "hf_chittagonian_vulgar_lexicon"
        / "Vulgar Word.csv"
    )
    lexicon = pd.read_csv(lexicon_path)
    for index, value in lexicon.iloc[:, 0].items():
        add(
            value,
            "hf_chittagonian_vulgar_lexicon",
            str(index),
            "CHI",
            "apache-2.0",
            "safety_lexicon_auxiliary",
            0.05,
            True,
        )

    protected_text = {
        row["source_text_model"]
        for row in normalization_rows
        if row["split"] != "train"
    } | {
        row["target_text_model"]
        for row in normalization_rows
        if row["split"] != "train"
    } | {
        row["text_model"] for row in identification_rows if row["split"] != "train"
    }
    representatives: dict[str, dict[str, Any]] = {}
    for row in candidates:
        if not row["text_model"] or not row["text_compact"]:
            excluded.append(exclusion_record("tokenizer", row, "empty_tokenizer_text"))
            continue
        if row["text_model"] in protected_text:
            excluded.append(
                exclusion_record("tokenizer", row, "exact_protected_evaluation_text")
            )
            continue
        current = representatives.get(row["text_model"])
        if current is None or row["sampling_weight"] > current["sampling_weight"]:
            if current is not None:
                excluded.append(
                    exclusion_record("tokenizer", current, "duplicate_tokenizer_text", row["row_id"])
                )
            representatives[row["text_model"]] = row
        else:
            excluded.append(
                exclusion_record("tokenizer", row, "duplicate_tokenizer_text", current["row_id"])
            )
    return list(representatives.values()), excluded


def write_parquet(path: Path, rows: list[dict[str, Any]], sort_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    frame.to_parquet(path, index=False, compression="zstd")


def dataframe_counts(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    result = frame.groupby(fields, dropna=False).size().reset_index(name="rows")
    return result.sort_values(fields, kind="stable").to_dict(orient="records")


def build_license_ledger() -> dict[str, Any]:
    sources = {}
    for meta in LOCAL_SOURCE_META.values():
        sources[meta["source_id"]] = {
            "provider": meta["provider"],
            "doi": meta["doi"],
            "license": meta["license"],
            "redistribution": "allowed_with_attribution",
            "verification": "versioned_repository_metadata_and_local_title/fingerprint audit",
        }
    for source_id, meta in EXTERNAL_SOURCE_META.items():
        sources[source_id] = {
            "provider": meta["provider"],
            "doi": meta["doi"],
            "license": meta["license"],
            "redistribution": "allowed_with_attribution",
            "verification": "immutable_repository_version_and_download_sha256",
        }
    sources["bangladial_v2"] = {
        "provider": "Mendeley",
        "doi": "10.17632/sx6ybcps2n.2",
        "license": "CC BY 4.0",
        "redistribution": "allowed_with_attribution",
        "verification": "immutable_repository_version_and_download_sha256",
        "caveat": "merged component provenance remains lower-confidence",
    }
    sources["hf_bengali_regional_asr_refine"] = {
        "provider": "Hugging Face",
        "repository": "sha1779/bengali_regional_dataset_refine",
        "revision": "6e3221bdf7f9c8c426276f1b619d7de597963f42",
        "license": "apache-2.0",
        "redistribution": "text-column extract retained locally",
        "verification": "pinned commit and repository card license tag",
        "caveat": "upstream competition provenance should be cited and rechecked before public redistribution",
    }
    sources["hf_chittagonian_vulgar_lexicon"] = {
        "provider": "Hugging Face",
        "repository": "kit-nlp/Vulgar_Lexicon_of_Chittagonian_Dialect_of_Bangla_or_Bengali",
        "revision": "ce4be3814b8eca1a4c0fa85f983625565f887088",
        "license": "apache-2.0",
        "redistribution": "auxiliary_only",
        "verification": "pinned commit and repository card license tag",
    }
    sources["synthetic_robustness_v1"] = {
        "provider": "deterministic_local_build",
        "license": "inherits_each_parent_license",
        "redistribution": "subject_to_parent_attribution",
        "authenticity_claim": False,
    }
    return {
        "ledger_version": "1.0.0",
        "generated_on": "2026-07-19",
        "sources": sources,
        "required_release_action": "include per-source attribution and indicate normalization/augmentation modifications",
    }


def markdown_table(records: list[dict[str, Any]], columns: list[str]) -> str:
    if not records:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for record in records:
        values = []
        for column in columns:
            value = record.get(column, "")
            if isinstance(value, int):
                values.append(f"{value:,}")
            else:
                values.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def make_human_review_sample(authentic: list[dict[str, Any]], path: Path) -> int:
    selected = []
    frame = pd.DataFrame(authentic)
    for _, group in frame.groupby(["source_id", "dialect"], sort=True):
        group = group.copy()
        group["review_order"] = group["row_id"].map(
            lambda value: stable_hash("native_review", value)
        )
        selected.append(group.sort_values("review_order").head(10))
    sample = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    keep = [
        "row_id",
        "source_id",
        "dialect",
        "split",
        "granularity",
        "source_text_raw",
        "source_text_model",
        "target_text_raw",
        "target_text_model",
        "quality_flags",
    ]
    sample = sample[keep].copy()
    sample["reviewer_id"] = ""
    sample["dialect_authenticity_1_to_5"] = ""
    sample["target_adequacy_1_to_5"] = ""
    sample["target_fluency_1_to_5"] = ""
    sample["label_correct_yes_no"] = ""
    sample["unsafe_or_pii_yes_no"] = ""
    sample["review_notes"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    return len(sample)


def make_semantic_components(
    normalization: list[dict[str, Any]], identification: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalization:
        groups[("normalization", row["semantic_group_id"])].append(row)
    for row in identification:
        groups[("identification", stable_hash("id_compact", row["text_compact"]))].append(row)
    components = []
    for (task, component_id), rows in groups.items():
        splits = sorted({row["split"] for row in rows}, key=lambda x: SPLIT_RANK.get(x, 99))
        components.append(
            {
                "task": task,
                "component_id": component_id,
                "row_count": len(rows),
                "splits": compact_json(splits),
                "sources": compact_json(sorted({row["source_id"] for row in rows})),
                "dialects": compact_json(sorted({row["dialect"] for row in rows})),
                "contains_train": "train" in splits,
                "contains_protected_evaluation": any(split != "train" for split in splits),
            }
        )
    return components


def split_compact_overlap(rows: list[dict[str, Any]], compact_field: str) -> int:
    train = {row[compact_field] for row in rows if row["split"] == "train"}
    protected = {row[compact_field] for row in rows if row["split"] != "train"}
    return len((train & protected) - {""})


def build_report(
    *,
    normalization: list[dict[str, Any]],
    authentic: list[dict[str, Any]],
    synthetic: list[dict[str, Any]],
    identification: list[dict[str, Any]],
    romanized: list[dict[str, Any]],
    tokenizer: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
    human_review_rows: int,
) -> dict[str, Any]:
    norm_frame = pd.DataFrame(normalization)
    id_frame = pd.DataFrame(identification)
    exclusion_frame = pd.DataFrame(exclusions)
    norm_overlap = split_compact_overlap(authentic, "source_text_compact") + split_compact_overlap(
        authentic, "target_text_compact"
    )
    id_overlap = split_compact_overlap(identification, "text_compact")
    id_cross_label = int(
        id_frame.groupby("text_compact")["dialect"].nunique().gt(1).sum()
    )
    labels = sorted(id_frame["dialect"].unique().tolist())
    gates = {
        "thirteen_label_inventory_present": labels == sorted(ALL_LABELS),
        "inventory_is_twelve_regional_plus_standard": (
            set(labels) == set(REGIONAL_LABELS) | {"STD"}
            and len(REGIONAL_LABELS) == 12
        ),
        "no_compact_train_evaluation_overlap_normalization": norm_overlap == 0,
        "no_compact_train_evaluation_overlap_identification": id_overlap == 0,
        "no_cross_label_identification_text": id_cross_label == 0,
        "synthetic_train_only": all(row["split"] == "train" for row in synthetic),
        "synthetic_excluded_from_identification": not any(
            row["is_synthetic"] for row in identification
        ),
        "locked_ood_sources_never_train": not any(
            row["source_id"] in {"onubad_v2", "bd_dialect_v2"}
            and row["split"] == "train"
            for row in normalization
        ),
        "legacy_derived_archive_absent": not any(
            "derived_archive" in row["source_id"] for row in normalization + identification
        ),
        "all_task_licenses_resolved": not any(
            not row["license"] or "UNKNOWN" in row["license"]
            for row in normalization + identification + tokenizer
        ),
        "native_human_review_complete": False,
    }
    engineering_pass = all(
        value for key, value in gates.items() if key != "native_human_review_complete"
    )
    report = {
        "dataset_version": normalization[0]["dataset_version"],
        "status": "PASS_INTERNAL_DATA_ENGINEERING" if engineering_pass else "FAIL_DATA_ENGINEERING",
        "publication_release_status": "CONDITIONAL_NATIVE_REVIEW_REQUIRED",
        "taxonomy": {
            "labels": ALL_LABELS,
            "regional_labels": REGIONAL_LABELS,
            "standard_label": "STD",
            "clarification": "13 labels = 12 regional varieties + Standard Bangla",
        },
        "counts": {
            "normalization_all": len(normalization),
            "normalization_authentic": len(authentic),
            "normalization_synthetic": len(synthetic),
            "identification_all": len(identification),
            "romanized_ood": len(romanized),
            "tokenizer_train_unique_texts": len(tokenizer),
            "excluded_row_decisions": len(exclusions),
            "legacy_derived_archive_quarantined_rows": 51101,
            "human_review_sample_rows": human_review_rows,
        },
        "normalization_by_source_dialect_split": dataframe_counts(
            normalization, ["source_id", "dialect", "split"]
        ),
        "normalization_by_dialect_split_synthetic": dataframe_counts(
            normalization, ["dialect", "split", "is_synthetic"]
        ),
        "identification_by_label_split": dataframe_counts(
            identification, ["dialect", "split"]
        ),
        "romanized_by_dialect": dataframe_counts(romanized, ["dialect"]),
        "exclusions_by_task_reason": (
            exclusion_frame.groupby(["task", "reason"]).size().reset_index(name="rows").sort_values(["task", "reason"]).to_dict(orient="records")
            if not exclusion_frame.empty
            else []
        ),
        "lineage_controls": {
            "bhasabodh_bengali_pairs_counted_as_onubad_companion_not_independent": True,
            "chattogramsent_exact_chatgaiyya_ancestry_removed_by_priority": True,
            "sylheti_v3_exact_v2_rows_removed_by_priority": True,
            "bangladial_v2_used_as_lower_confidence_merged_identification_source": True,
            "hf_habiganj_and_sandwip_not_geographically_relabelled": True,
        },
        "near_duplicate_method": {
            "compact_exact": True,
            "simhash_bits": 64,
            "character_ngram": 4,
            "maximum_hamming_candidate": 6,
            "minimum_jaccard": 0.90,
            "direction": "remove_train_against_iid_protected; remove_ood_against_all_development",
        },
        "gates": gates,
        "artifacts": artifacts,
        "required_manual_action": (
            "Complete the stratified native-speaker review sheet before any "
            "paper claim that the corpus is publication-ready or linguistically validated."
        ),
    }
    return report


def write_markdown_reports(report: dict[str, Any]) -> None:
    counts = report["counts"]
    norm_table = report["normalization_by_dialect_split_synthetic"]
    id_table = report["identification_by_label_split"]
    exclusion_table = report["exclusions_by_task_reason"]
    card = f"""# Boichitro Data v1.0.0

## Scope

This is the frozen data release for the Bangla dialect SLM/MoE experiments.
The label inventory has **13 classes: 12 regional varieties plus Standard
Bangla (`STD`)**. It must not be described as 13 regional dialects.

The primary supervised task is dialect-to-Standard-Bangla normalization. The
secondary task is 13-label dialect identification. Bengali-script parallel,
romanized robustness, source-held-out OOD, synthetic robustness, and tokenizer
data are stored separately so a result cannot silently mix these conditions.

## Release state

- Data-engineering gate: `{report['status']}`
- Public/paper release gate: `{report['publication_release_status']}`
- Authentic normalization rows: {counts['normalization_authentic']:,}
- Train-only synthetic robustness rows: {counts['normalization_synthetic']:,}
- Identification rows: {counts['identification_all']:,}
- Romanized source-held-out rows: {counts['romanized_ood']:,}
- Unique tokenizer-training texts: {counts['tokenizer_train_unique_texts']:,}

The algorithmic build is frozen and reproducible. Linguistic publication
claims remain conditional on completing `reports/HUMAN_NATIVE_REVIEW_SAMPLE.csv`.

## Text normalization levels

1. `raw`: original source cell, unchanged.
2. `nfc`: Unicode NFC with BOM/zero-width storage artifacts removed.
3. `clean`: control-character and whitespace cleanup.
4. `model`: conservative quote, dash, punctuation-spacing, and transcript-marker cleanup.
5. `compact`: NFKC letters/numbers only, used **only** for leakage and duplicate detection.

The compact form is never used as the model input. Punctuation and dialectal
spelling are not broadly standardized away.

## Normalization composition

{markdown_table(norm_table, ['dialect', 'split', 'is_synthetic', 'rows'])}

## Identification composition

{markdown_table(id_table, ['dialect', 'split', 'rows'])}

## External-source roles

- Kothon v4: authentic CHI/SYL train-development candidate after decontamination.
- Sylheti translation v3: only novel rows; lexical items are train-only.
- ChattogramSent v2: only the novel remainder; not claimed as independent OOD.
- ONUBAD v2: locked source-OOD Bengali-script sentence benchmark.
- BhasaBodh v1: romanized companion to accepted ONUBAD rows, not double-counted.
- BD-Dialect v2: fully locked phrase/lexical OOD challenge, including RAJ.
- Hugging Face regional ASR: exact-name mapped transcript identification data.
- BanglaDial v2: lower-confidence merged-provenance identification pool.
- Chittagonian vulgar lexicon: tokenizer/safety auxiliary only.

Habiganj and Sandwip are not collapsed into Sylhet or Chittagong. They remain
audited exclusions because geographic proximity is not a defensible label map.

## Synthetic policy

Synthetic examples are deterministic perturbations of accepted authentic
**training parents only**, capped at 10% overall and 15% per dialect. They keep
parent IDs, receive 0.5 example loss weight, make no authenticity claim, and
are ineligible for dialect identification, validation, or test sets. The old
51,101-row derived ZIP remains quarantined and contributes zero final rows.

## Licensing and attribution

Mendeley inputs used here are pinned CC BY 4.0 releases; Hugging Face inputs are
pinned Apache-2.0 releases. See `data/manifests/licenses.yaml` for per-source
DOIs, commits, caveats, and release obligations. Modifications and derived
augmentation must be disclosed in any redistributed version.

## Known limitations

- BanglaDial is a merged corpus whose original component provenance is incomplete.
- The Hugging Face ASR card self-reports Apache-2.0; upstream competition terms
  should be rechecked before redistributing its text extract.
- Written dialect labels are regional proxies, not speaker-identity ground truth.
- Native-speaker review is deliberately not fabricated by this automated build.
- Source-held-out coverage is strongest for BAR/CHI/SYL and the five BD-Dialect varieties.
"""
    (FINAL_ROOT / "DATASET_CARD.md").write_text(card, encoding="utf-8")

    report_md = f"""# Final dataset audit

Status: **{report['status']}**
Publication status: **{report['publication_release_status']}**

## Core counts

| Item | Rows |
|---|---:|
| Authentic normalization | {counts['normalization_authentic']:,} |
| Synthetic normalization (train only) | {counts['normalization_synthetic']:,} |
| Identification | {counts['identification_all']:,} |
| Romanized OOD | {counts['romanized_ood']:,} |
| Tokenizer unique training text | {counts['tokenizer_train_unique_texts']:,} |
| Exclusion decisions | {counts['excluded_row_decisions']:,} |
| Legacy derived ZIP still quarantined | {counts['legacy_derived_archive_quarantined_rows']:,} |

## Gate results

{markdown_table([{'gate': key, 'pass': value} for key, value in report['gates'].items()], ['gate', 'pass'])}

## Exclusion decisions

{markdown_table(exclusion_table, ['task', 'reason', 'rows'])}

## Correct interpretation

The corpus has 13 labels, consisting of 12 regional labels and Standard Bangla.
The data-engineering checks pass for internal experimentation. The native review
sheet contains {counts['human_review_sample_rows']:,} stratified rows and must be
completed before a Q1 manuscript calls the release linguistically validated.
"""
    (REPORT_ROOT / "FINAL_DATASET_REPORT.md").write_text(report_md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=BUILD_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not (REPORT_ROOT / "external_source_acquisition.json").exists():
        raise RuntimeError("Run fetch_external_sources.py before the final build")

    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    exclusions: list[dict[str, Any]] = []

    print("Loading local and external parallel data", flush=True)
    local, local_excluded = load_local_parallel(config)
    external_training = load_training_external(config)
    exclusions.extend(local_excluded)
    assign_external_splits(local, external_training, config)

    development, duplicate_excluded = resolve_pair_conflicts_and_duplicates(
        local + external_training, config["source_priority"]
    )
    exclusions.extend(duplicate_excluded)
    development, leakage_excluded = remove_train_protected_near_leakage(
        development, config
    )
    exclusions.extend(leakage_excluded)

    print("Building and decontaminating locked OOD sets", flush=True)
    ood_candidates = load_ood_parallel(config)
    ood_candidates, ood_duplicate_excluded = resolve_pair_conflicts_and_duplicates(
        ood_candidates, {"onubad_v2": 0, "bd_dialect_v2": 0}
    )
    exclusions.extend(ood_duplicate_excluded)
    ood, ood_overlap_excluded = decontaminate_ood(
        ood_candidates, development, config
    )
    exclusions.extend(ood_overlap_excluded)

    authentic = development + ood
    romanized, romanized_excluded = build_romanized_eval(ood, config)
    exclusions.extend(romanized_excluded)

    print("Generating traceable train-only robustness tier", flush=True)
    synthetic = build_synthetic(authentic, config)
    normalization = authentic + synthetic
    temperature_weights(normalization, "dialect", config)

    print("Building 13-label identification and tokenizer manifests", flush=True)
    identification, id_excluded = build_identification(authentic, config)
    exclusions.extend(id_excluded)
    tokenizer, tokenizer_excluded = build_tokenizer_manifest(
        normalization, identification, config
    )
    exclusions.extend(tokenizer_excluded)

    # Deterministic task artifacts.
    outputs: dict[str, tuple[list[dict[str, Any]], list[str]]] = {
        "normalization_all.parquet": (normalization, ["split", "dialect", "source_id", "row_id"]),
        "normalization_train.parquet": ([row for row in normalization if row["split"] == "train"], ["dialect", "source_id", "row_id"]),
        "normalization_validation.parquet": ([row for row in normalization if row["split"] == "validation"], ["dialect", "source_id", "row_id"]),
        "normalization_test_iid.parquet": ([row for row in normalization if row["split"] == "test"], ["dialect", "source_id", "row_id"]),
        "normalization_test_ood.parquet": ([row for row in normalization if row["split"] == "test_ood"], ["evaluation_track", "dialect", "source_id", "row_id"]),
        "normalization_synthetic_train.parquet": (synthetic, ["dialect", "parent_row_id", "row_id"]),
        "identification_all.parquet": (identification, ["split", "dialect", "source_id", "row_id"]),
        "identification_train.parquet": ([row for row in identification if row["split"] == "train"], ["dialect", "source_id", "row_id"]),
        "identification_evaluation.parquet": ([row for row in identification if row["split"] != "train"], ["split", "dialect", "source_id", "row_id"]),
        "romanized_test_ood.parquet": (romanized, ["dialect", "source_row_id", "row_id"]),
        "tokenizer_train.parquet": (tokenizer, ["dialect", "source_id", "row_id"]),
        "excluded_rows.parquet": (exclusions, ["task", "reason", "source_id", "exclusion_id"]),
    }
    for filename, (rows, sort_columns) in outputs.items():
        write_parquet(FINAL_ROOT / filename, rows, sort_columns)

    # Canonical machine-readable manifests used by the experiment registry.
    norm_frame = pd.DataFrame(normalization)
    id_frame = pd.DataFrame(identification)
    canonical = pd.concat([norm_frame, id_frame], ignore_index=True, sort=False)
    canonical = canonical.sort_values(["task", "split", "dialect", "row_id"], kind="stable")
    canonical.to_parquet(
        MANIFEST_ROOT / "canonical_rows.parquet", index=False, compression="zstd"
    )
    split_manifest = canonical[
        ["task", "row_id", "source_id", "dialect", "split", "train_eligible", "eval_eligible"]
    ].copy()
    split_manifest.to_parquet(
        MANIFEST_ROOT / "splits_v1.parquet", index=False, compression="zstd"
    )
    components = make_semantic_components(normalization, identification)
    write_parquet(
        MANIFEST_ROOT / "semantic_components.parquet",
        components,
        ["task", "component_id"],
    )
    write_parquet(
        MANIFEST_ROOT / "excluded_rows.parquet",
        exclusions,
        ["task", "reason", "source_id", "exclusion_id"],
    )
    write_parquet(
        MANIFEST_ROOT / "tokenizer_train_v1.parquet",
        tokenizer,
        ["dialect", "source_id", "row_id"],
    )
    ledger = build_license_ledger()
    (MANIFEST_ROOT / "licenses.yaml").write_text(
        yaml.safe_dump(ledger, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    review_rows = make_human_review_sample(
        authentic, REPORT_ROOT / "HUMAN_NATIVE_REVIEW_SAMPLE.csv"
    )
    artifact_paths = [
        *(FINAL_ROOT / filename for filename in outputs),
        MANIFEST_ROOT / "canonical_rows.parquet",
        MANIFEST_ROOT / "splits_v1.parquet",
        MANIFEST_ROOT / "semantic_components.parquet",
        MANIFEST_ROOT / "excluded_rows.parquet",
        MANIFEST_ROOT / "tokenizer_train_v1.parquet",
        MANIFEST_ROOT / "licenses.yaml",
    ]
    artifacts = {
        str(path.relative_to(PROJECT)): {
            "rows": (
                len(pd.read_parquet(path)) if path.suffix == ".parquet" else None
            ),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
    }
    report = build_report(
        normalization=normalization,
        authentic=authentic,
        synthetic=synthetic,
        identification=identification,
        romanized=romanized,
        tokenizer=tokenizer,
        exclusions=exclusions,
        artifacts=artifacts,
        human_review_rows=review_rows,
    )
    (REPORT_ROOT / "final_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    data_gate = {
        "dataset_version": config["dataset_version"],
        "status": "pass" if report["status"] == "PASS_INTERNAL_DATA_ENGINEERING" else "fail",
        "scope": "internal_model_and_tokenizer_experiments",
        "training_authorized": report["status"] == "PASS_INTERNAL_DATA_ENGINEERING",
        "public_redistribution_authorized": False,
        "publication_claim_authorized": False,
        "publication_blocker": "stratified_native_speaker_review_pending",
        "report": "reports/final_dataset_report.json",
        "gates": report["gates"],
    }
    (REPORT_ROOT / "data_gate.json").write_text(
        json.dumps(data_gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown_reports(report)
    print(json.dumps({"status": report["status"], "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
