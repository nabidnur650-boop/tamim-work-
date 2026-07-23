#!/usr/bin/env python3
"""Reproducible member-level audit of the two local Bangla dialect archives.

The script reads both ZIP files without modifying them, canonicalizes the CSV
and XLSX members, and writes a JSON result plus a concise Markdown report.
Only the CSV version of Vashantor is treated as canonical because several JSON
test files contain split rows and inconsistent keys.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import pandas as pd


RAW_ARCHIVE = "archive(1).zip"
DERIVED_ARCHIVE = "archive (1).zip"

DIALECT_MAP = {
    "barishal": "BAR",
    "barisal": "BAR",
    "chittagong": "CHI",
    "khulna": "KHU",
    "kishoreganj": "KIS",
    "mymensingh": "MYM",
    "narail": "NAR",
    "noakhali": "NOA",
    "narsingdi": "NSD",
    "rajshahi": "RAJ",
    "rangpur": "RAN",
    "standard_bangla": "STD",
    "sylhet": "SYL",
    "tangail": "TAN",
}

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = (
        text.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\ufeff", "")
    )
    return re.sub(r"\s+", " ", text).strip()


def preserve_raw_text(value: Any) -> str:
    """Return the source cell as text without normalization."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_member(archive: ZipFile, member: str) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(archive.read(member)), low_memory=False)
    frame.columns = [normalize_text(column) for column in frame.columns]
    return frame


def excel_column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference)
    if not letters:
        return 0
    result = 0
    for char in letters.group(0):
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def read_xlsx_first_sheet(payload: bytes) -> pd.DataFrame:
    """Read the first XLSX worksheet using only the Python standard library."""
    with ZipFile(BytesIO(payload)) as workbook_zip:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in workbook_zip.namelist():
            shared_root = ET.fromstring(workbook_zip.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f".//{{{MAIN_NS}}}si"):
                shared.append(
                    "".join(
                        node.text or ""
                        for node in item.iter(f"{{{MAIN_NS}}}t")
                    )
                )

        workbook_root = ET.fromstring(workbook_zip.read("xl/workbook.xml"))
        first_sheet = workbook_root.find(f".//{{{MAIN_NS}}}sheet")
        if first_sheet is None:
            return pd.DataFrame()
        relationship_id = first_sheet.attrib[f"{{{REL_NS}}}id"]

        rels_root = ET.fromstring(
            workbook_zip.read("xl/_rels/workbook.xml.rels")
        )
        target = None
        for relation in rels_root.findall(f".//{{{PKG_REL_NS}}}Relationship"):
            if relation.attrib.get("Id") == relationship_id:
                target = relation.attrib["Target"]
                break
        if target is None:
            raise ValueError("Unable to resolve first XLSX worksheet")

        if target.startswith("/"):
            sheet_path = target.lstrip("/")
        else:
            sheet_path = str(PurePosixPath("xl") / target)
        sheet_path = str(PurePosixPath(sheet_path))
        sheet_root = ET.fromstring(workbook_zip.read(sheet_path))

        rows: list[list[Any]] = []
        for row in sheet_root.findall(f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row"):
            values: dict[int, Any] = {}
            for cell in row.findall(f"{{{MAIN_NS}}}c"):
                column_index = excel_column_index(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or ""
                        for node in cell.iter(f"{{{MAIN_NS}}}t")
                    )
                else:
                    value_node = cell.find(f"{{{MAIN_NS}}}v")
                    value = value_node.text if value_node is not None else None
                    if cell_type == "s" and value is not None:
                        value = shared[int(value)]
                    elif cell_type == "b" and value is not None:
                        value = value == "1"
                values[column_index] = value
            if values:
                width = max(values) + 1
                rows.append([values.get(index) for index in range(width)])

    if not rows:
        return pd.DataFrame()
    width = max(len(row) for row in rows)
    padded = [row + [None] * (width - len(row)) for row in rows]
    headers = [
        normalize_text(value) if value is not None else f"column_{index}"
        for index, value in enumerate(padded[0])
    ]
    return pd.DataFrame(padded[1:], columns=headers)


def find_member(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one member ending in {suffix!r}; got {matches}")
    return matches[0]


def archive_inventory(path: Path) -> dict[str, Any]:
    with ZipFile(path) as archive:
        members = []
        for info in archive.infolist():
            payload = archive.read(info.filename)
            members.append(
                {
                    "name": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "sha256": sha256_bytes(payload),
                }
            )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "member_count": len(members),
        "members": members,
    }


def load_vashantor(raw_zip: ZipFile) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    json_diagnostics: list[dict[str, Any]] = []
    for member in raw_zip.namelist():
        if member.endswith(".csv") and "Vashantor_CSV_Format" in member:
            frame = read_csv_member(raw_zip, member)
            if "/Train/" in member:
                split = "train"
            elif "/Validation/" in member:
                split = "validation"
            elif "/Test/" in member:
                split = "test"
            else:
                raise ValueError(f"Unknown Vashantor split: {member}")

            dialect_name = Path(member).name.split()[0].lower()
            dialect = DIALECT_MAP[dialect_name]
            regional_candidates = [
                column
                for column in frame.columns
                if column.endswith("_bangla_speech")
                and column != "bangla_speech"
            ]
            if len(regional_candidates) != 1:
                raise ValueError(
                    f"Expected one regional Bangla column in {member}: "
                    f"{regional_candidates}"
                )
            regional_column = regional_candidates[0]
            for row_index, row in frame.iterrows():
                records.append(
                    {
                        "dataset": "Vashantor",
                        "archive_member": member,
                        "split": split,
                        "dialect": dialect,
                        "source_text_raw": preserve_raw_text(row[regional_column]),
                        "source_text": normalize_text(row[regional_column]),
                        "standard_text_raw": preserve_raw_text(row["bangla_speech"]),
                        "standard_text": normalize_text(row["bangla_speech"]),
                        "source_romanized": normalize_text(
                            row.get(regional_column.replace("_bangla_", "_banglish_"))
                        ),
                        "standard_romanized": normalize_text(
                            row.get("banglish_speech")
                        ),
                        "english_text": normalize_text(row.get("english_speech")),
                        "source_row_id": int(row_index),
                        "semantic_group_id": f"vashantor:{split}:{row_index}",
                    }
                )

        if member.endswith(".json") and "Vashantor_JSON_Format" in member:
            rows = json.loads(raw_zip.read(member))
            normalized_rows = [
                {normalize_text(key): value for key, value in row.items()}
                for row in rows
            ]
            filename_dialect = Path(member).name.split()[0].lower()
            regional_column = f"{filename_dialect}_bangla_speech"
            json_diagnostics.append(
                {
                    "archive_member": member,
                    "rows": len(rows),
                    "missing_standard": sum(
                        not normalize_text(row.get("bangla_speech"))
                        for row in normalized_rows
                    ),
                    "missing_regional": sum(
                        not normalize_text(row.get(regional_column))
                        for row in normalized_rows
                    ),
                    "inconsistent_trailing_space_keys": len(
                        {
                            tuple(sorted(row.keys()))
                            for row in rows
                        }
                    )
                    > 1,
                }
            )

    vashantor = pd.DataFrame(records)
    diagnostics: dict[str, Any] = {"json_files": json_diagnostics}

    alignments = []
    for split, split_frame in vashantor.groupby("split"):
        target_sets = {
            dialect: set(group["standard_text"])
            for dialect, group in split_frame.groupby("dialect")
        }
        source_to_labels = split_frame.groupby("source_text")["dialect"].nunique()
        wrong_input_to_labels = split_frame.groupby("standard_text")[
            "dialect"
        ].nunique()
        alignments.append(
            {
                "split": split,
                "rows": len(split_frame),
                "per_dialect_rows": split_frame.groupby("dialect")
                .size()
                .sort_index()
                .to_dict(),
                "standard_target_intersection": len(
                    set.intersection(*target_sets.values())
                ),
                "standard_target_union": len(set.union(*target_sets.values())),
                "cross_dialect_identical_source_texts": int(
                    (source_to_labels > 1).sum()
                ),
                "wrong_column_unique_inputs": int(
                    split_frame["standard_text"].nunique()
                ),
                "wrong_column_inputs_shared_by_at_least_two_labels": int(
                    (wrong_input_to_labels >= 2).sum()
                ),
                "wrong_column_inputs_shared_by_all_five_labels": int(
                    (wrong_input_to_labels == 5).sum()
                ),
                "wrong_column_rows_in_label_collisions": int(
                    split_frame[
                        split_frame["standard_text"].isin(
                            wrong_input_to_labels[
                                wrong_input_to_labels >= 2
                            ].index
                        )
                    ].shape[0]
                ),
            }
        )
    diagnostics["split_alignment"] = alignments

    split_sets = {
        split: set(group["source_text"])
        for split, group in vashantor.groupby("split")
    }
    diagnostics["regional_source_overlap_between_splits"] = {
        f"{left}_x_{right}": len(split_sets[left] & split_sets[right])
        for left, right in itertools.combinations(sorted(split_sets), 2)
    }
    return vashantor, diagnostics


def load_raw_datasets(
    raw_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    with ZipFile(raw_path) as raw_zip:
        names = raw_zip.namelist()
        vashantor, vashantor_diagnostics = load_vashantor(raw_zip)

        sylhet_member = find_member(
            names, "Bangla Dialect Transaction Dataset_v2 - Sheet1.csv"
        )
        sylhet = read_csv_member(raw_zip, sylhet_member)

        chat_member = find_member(names, "Dataset_Chittagong_2.0.csv")
        chat = read_csv_member(raw_zip, chat_member)
        dictionary_member = find_member(names, "dictionary.csv")
        dictionary = read_csv_member(raw_zip, dictionary_member)

        regional_member = find_member(names, "BanglaRegionalTextCorpus.xlsx")
        regional = read_xlsx_first_sheet(raw_zip.read(regional_member))
        clean_regional_member = find_member(
            names, "Regional_cleaned_dataset.xlsx"
        )
        regional_clean = read_xlsx_first_sheet(
            raw_zip.read(clean_regional_member)
        )

        bangladial_csv_member = find_member(
            names,
            "BanglaDial_ A Merged and Imbalanced text Dataset for Bengali "
            "Regional dialect analysis. - Sheet1.csv",
        )
        bangladial_csv = read_csv_member(raw_zip, bangladial_csv_member)
        bangladial_xlsx_member = find_member(
            names,
            "BanglaDial_ A Merged and Imbalanced text Dataset for Bengali "
            "Regional dialect analysis..xlsx",
        )
        bangladial_xlsx = read_xlsx_first_sheet(
            raw_zip.read(bangladial_xlsx_member)
        )

    pair_records = vashantor.to_dict("records")

    def append_pairs(
        dataset: str,
        dialect: str,
        sources: pd.Series,
        targets: pd.Series,
        member: str,
    ) -> None:
        for row_index, (source, target) in enumerate(zip(sources, targets)):
            pair_records.append(
                {
                    "dataset": dataset,
                    "archive_member": member,
                    "split": "unsplit",
                    "dialect": dialect,
                    "source_text_raw": preserve_raw_text(source),
                    "source_text": normalize_text(source),
                    "standard_text_raw": preserve_raw_text(target),
                    "standard_text": normalize_text(target),
                    "source_romanized": "",
                    "standard_romanized": "",
                    "english_text": "",
                    "source_row_id": row_index,
                    # Non-Vashantor sources do not provide evidence that row
                    # positions align across dialects. Include the dialect so
                    # separately grouped workbooks cannot create false pairs.
                    "semantic_group_id": f"{dataset}:{dialect}:{row_index}",
                }
            )

    append_pairs(
        "Sylheti1200",
        "SYL",
        sylhet["Local Bangla Dialect(Sylheti) Text"],
        sylhet["Standard Bangla Text"],
        sylhet_member,
    )
    append_pairs(
        "ChatgaiyyaAlap",
        "CHI",
        chat["চট্টগ্রাম"],
        chat["বাংলা"],
        chat_member,
    )
    regional_map = {
        "Barisal": "BAR",
        "Khulna": "KHU",
        "Narail": "NAR",
        "Rangpur": "RAN",
    }
    for region_name, group in regional.groupby("Region_Labels", sort=True):
        append_pairs(
            "BanglaRegionalTextCorpus",
            regional_map[normalize_text(region_name)],
            group["Regional_Texts"],
            group["Standard_Bangla_Texts"],
            regional_member,
        )

    pairs = pd.DataFrame(pair_records)
    for column in ["source_text", "standard_text"]:
        pairs[column] = pairs[column].map(normalize_text)

    bangladial = bangladial_csv.rename(
        columns={"Sentence": "text", "Language": "language"}
    ).copy()
    bangladial["text_raw"] = bangladial["text"].map(preserve_raw_text)
    bangladial["text"] = bangladial["text"].map(normalize_text)
    bangladial["dialect"] = bangladial["language"].map(
        lambda value: DIALECT_MAP[normalize_text(value).lower()]
    )

    regional_text_set = set(regional["Regional_Texts"].map(normalize_text))
    regional_clean_set = set(regional_clean["Sentence"].map(normalize_text))
    diagnostics = {
        "vashantor": vashantor_diagnostics,
        "bangladial_csv_xlsx_exact_equal": bool(
            bangladial_csv.fillna("").astype(str).equals(
                bangladial_xlsx.fillna("").astype(str)
            )
        ),
        "regional_workbooks": {
            "full_rows": len(regional),
            "clean_rows": len(regional_clean),
            "full_unique_normalized_regional_texts": len(regional_text_set),
            "clean_unique_normalized_regional_texts": len(regional_clean_set),
            "regional_text_sets_equal": regional_text_set
            == regional_clean_set,
            "decision": (
                "Use BanglaRegionalTextCorpus.xlsx once because it retains "
                "the Standard Bangla target; exclude Regional_cleaned_dataset.xlsx "
                "as a duplicate representation."
            ),
        },
        "dictionary_rows": len(dictionary),
    }
    frames = {
        "vashantor": vashantor,
        "sylhet": sylhet,
        "chat": chat,
        "dictionary": dictionary,
        "regional": regional,
        "bangladial": bangladial,
    }
    return pairs, bangladial, diagnostics, frames


def load_derived_archive(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    with ZipFile(path) as archive:
        for member in archive.namelist():
            frame = read_csv_member(archive, member)
            frame["archive_member"] = member
            frame["text_raw"] = frame["text"].map(preserve_raw_text)
            frame["text_normalized"] = frame["text"].map(normalize_text)
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)

    conflict_sizes = combined.groupby("text_normalized")["dialect"].nunique()
    conflict_texts = set(conflict_sizes[conflict_sizes > 1].index)
    per_file = []
    for member, group in combined.groupby("archive_member"):
        file_conflicts = group.groupby("text_normalized")["dialect"].nunique()
        file_conflict_texts = set(file_conflicts[file_conflicts > 1].index)
        per_file.append(
            {
                "archive_member": member,
                "rows": len(group),
                "unique_normalized_texts": int(
                    group["text_normalized"].nunique()
                ),
                "duplicate_rows_involved": int(
                    group.duplicated("text_normalized", keep=False).sum()
                ),
                "cross_label_texts": len(file_conflict_texts),
                "cross_label_rows": int(
                    group["text_normalized"].isin(file_conflict_texts).sum()
                ),
                "dialect_counts": group.groupby("dialect")
                .size()
                .sort_index()
                .to_dict(),
                "source_counts": group.groupby("source")
                .size()
                .sort_values(ascending=False)
                .to_dict(),
            }
        )

    heldout_mask = combined["source"].fillna("").str.contains(
        "Test|Validation", case=False, regex=True
    )
    heldout = combined[heldout_mask]
    diagnostics = {
        "rows": len(combined),
        "unique_normalized_texts": int(
            combined["text_normalized"].nunique()
        ),
        "duplicate_rows_involved": int(
            combined.duplicated("text_normalized", keep=False).sum()
        ),
        "cross_label_texts": len(conflict_texts),
        "cross_label_rows": int(
            combined["text_normalized"].isin(conflict_texts).sum()
        ),
        "dialect_counts": combined.groupby("dialect")
        .size()
        .sort_index()
        .to_dict(),
        "explicit_heldout_derived_rows": len(heldout),
        "explicit_heldout_source_counts": heldout.groupby("source")
        .size()
        .sort_values(ascending=False)
        .to_dict(),
        "per_file": per_file,
    }
    return combined, diagnostics


def pair_summary(pairs: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    grouped = pairs.groupby(["dataset", "split", "dialect"], sort=True)
    for (dataset, split, dialect), group in grouped:
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "dialect": dialect,
                "rows": len(group),
                "unique_source_texts": int(group["source_text"].nunique()),
                "unique_standard_targets": int(
                    group["standard_text"].nunique()
                ),
                "unique_pairs": int(
                    group[["source_text", "standard_text"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "source_equals_target": int(
                    (group["source_text"] == group["standard_text"]).sum()
                ),
                "empty_source_or_target": int(
                    (
                        (group["source_text"] == "")
                        | (group["standard_text"] == "")
                    ).sum()
                ),
            }
        )
    return rows


def analyze_overlaps(
    pairs: pd.DataFrame,
    bangladial: pd.DataFrame,
    derived: pd.DataFrame,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    bangladial_set = set(bangladial["text"])

    pair_source_sets = {
        dataset: set(group["source_text"])
        for dataset, group in pairs.groupby("dataset")
    }
    pair_target_sets = {
        dataset: set(group["standard_text"])
        for dataset, group in pairs.groupby("dataset")
    }
    dataset_overlap = []
    for left, right in itertools.combinations(sorted(pair_source_sets), 2):
        dataset_overlap.append(
            {
                "left": left,
                "right": right,
                "shared_unique_regional_source_texts": len(
                    pair_source_sets[left] & pair_source_sets[right]
                ),
            }
        )
    results["pair_dataset_source_overlap"] = dataset_overlap
    results["bangladial_pair_overlap"] = [
        {
            "dataset": dataset,
            "regional_source_overlap": len(
                source_set & bangladial_set
            ),
            "standard_target_overlap": len(
                pair_target_sets[dataset] & bangladial_set
            ),
        }
        for dataset, source_set in sorted(pair_source_sets.items())
    ]

    vashantor = pairs[pairs["dataset"] == "Vashantor"]
    vashantor_overlap = []
    for (split, dialect), group in vashantor.groupby(["split", "dialect"]):
        same_label_bangladial = set(
            bangladial[bangladial["dialect"] == dialect]["text"]
        )
        vashantor_overlap.append(
            {
                "split": split,
                "dialect": dialect,
                "unique_regional_source_texts": int(
                    group["source_text"].nunique()
                ),
                "same_label_overlap_with_bangladial": len(
                    set(group["source_text"]) & same_label_bangladial
                ),
            }
        )
    results["vashantor_overlap_with_bangladial_by_split"] = (
        vashantor_overlap
    )

    derived_overlap = []
    for member, derived_group in derived.groupby("archive_member"):
        entry: dict[str, Any] = {
            "archive_member": member,
            "overlap_with_bangladial": len(
                set(derived_group["text_normalized"]) & bangladial_set
            ),
            "vashantor_exact_overlap": [],
        }
        for (split, dialect), group in vashantor.groupby(
            ["split", "dialect"]
        ):
            dialect_derived = derived_group[
                derived_group["dialect"] == dialect
            ]
            overlap = len(
                set(dialect_derived["text_normalized"])
                & set(group["source_text"])
            )
            if overlap:
                entry["vashantor_exact_overlap"].append(
                    {
                        "split": split,
                        "dialect": dialect,
                        "unique_texts": overlap,
                    }
                )
        derived_overlap.append(entry)
    results["derived_overlap"] = derived_overlap
    return results


def bangladial_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    conflict_counts = frame.groupby("text")["dialect"].nunique()
    conflict_texts = set(conflict_counts[conflict_counts > 1].index)
    placeholder_pattern = re.compile(r"^[\s<>।?!,.\-]*$")
    return {
        "rows": len(frame),
        "unique_normalized_texts": int(frame["text"].nunique()),
        "duplicate_rows_involved": int(
            frame.duplicated("text", keep=False).sum()
        ),
        "cross_label_texts": len(conflict_texts),
        "cross_label_rows": int(frame["text"].isin(conflict_texts).sum()),
        "label_counts": frame.groupby("dialect")
        .size()
        .sort_index()
        .to_dict(),
        "placeholder_or_punctuation_only_rows": int(
            frame["text"].map(lambda text: bool(placeholder_pattern.fullmatch(text))).sum()
        ),
        "rows_containing_angle_placeholder": int(
            frame["text"].str.contains("<", regex=False).sum()
        ),
        "rows_length_at_most_two": int(
            frame["text"].str.len().le(2).sum()
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    pair_rows = result["pair_summary"]
    raw = result["raw_data"]
    derived = result["derived_data"]
    bangladial = result["bangladial"]
    vashantor = result["raw_diagnostics"]["vashantor"]

    lines = [
        "# Local Bangla Dialect Archive Audit",
        "",
        f"Generated by audit_local_archives.py. Audit schema version: "
        f"{result['audit_schema_version']}.",
        "",
        "## Executive decision",
        "",
        "- archive(1).zip is the canonical raw-data archive.",
        "- archive (1).zip is a derived/augmented archive and is quarantined "
        "from all main training runs until ancestry and human-quality gates pass.",
        "- The local raw archive supports dialect-to-standard normalization "
        "for eight dialects and classification for twelve regional labels plus "
        "Standard Bangla.",
        "- Use Vashantor CSV files, not its JSON duplicates. Several JSON test "
        "files contain split/incomplete rows.",
        "- Use BanglaRegionalTextCorpus.xlsx once; Regional_cleaned_dataset.xlsx "
        "duplicates the regional texts and drops the Standard Bangla target.",
        "",
        "## Archive identity",
        "",
        "| Archive | SHA-256 | Members | Bytes | Role |",
        "|---|---|---:|---:|---|",
        f"| {RAW_ARCHIVE} | {result['archives'][RAW_ARCHIVE]['sha256']} | "
        f"{result['archives'][RAW_ARCHIVE]['member_count']} | "
        f"{result['archives'][RAW_ARCHIVE]['bytes']} | canonical raw |",
        f"| {DERIVED_ARCHIVE} | "
        f"{result['archives'][DERIVED_ARCHIVE]['sha256']} | "
        f"{result['archives'][DERIVED_ARCHIVE]['member_count']} | "
        f"{result['archives'][DERIVED_ARCHIVE]['bytes']} | quarantined derived |",
        "",
        "## Usable aligned normalization data",
        "",
        f"Total local aligned rows: {raw['aligned_pair_rows']:,}; unique "
        f"dialect-tagged pairs: {raw['unique_dialect_tagged_pairs']:,}; "
        f"dialects: {', '.join(raw['normalization_dialects'])}.",
        "",
        "| Dataset | Split | Dialect | Rows | Unique source | Unique target | "
        "Unique pairs | Source=target |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['dialect']} | "
            f"{row['rows']} | {row['unique_source_texts']} | "
            f"{row['unique_standard_targets']} | {row['unique_pairs']} | "
            f"{row['source_equals_target']} |"
        )

    lines += [
        "",
        "## Critical integrity findings",
        "",
        "1. Vashantor has separate Standard Bangla and regional Bangla columns. "
        "Using bangla_speech as the classifier input creates near-identical "
        "inputs under five different labels.",
    ]
    for row in sorted(vashantor["split_alignment"], key=lambda item: item["split"]):
        lines.append(
            f"   - {row['split']}: "
            f"{row['wrong_column_rows_in_label_collisions']:,} of "
            f"{row['rows']:,} rows fall in cross-label collisions; "
            f"{row['wrong_column_inputs_shared_by_all_five_labels']:,} unique "
            "Standard Bangla inputs are shared by all five labels."
        )
    lines += [
        "2. BanglaDial contains Vashantor train, validation, and test regional "
        "sentences. It cannot be mixed with Vashantor and then treated as an "
        "independent training source.",
        f"3. The derived archive contains "
        f"{derived['explicit_heldout_derived_rows']:,} rows whose source name "
        "explicitly identifies Vashantor validation or test ancestry.",
        f"4. Across the derived archive, {derived['cross_label_texts']:,} "
        f"normalized texts have multiple dialect labels, involving "
        f"{derived['cross_label_rows']:,} rows.",
        f"5. BanglaDial itself has {bangladial['cross_label_texts']:,} "
        f"cross-label normalized texts and {bangladial['duplicate_rows_involved']:,} "
        "rows involved in exact normalized duplication.",
        "6. Vashantor official regional inputs have six exact normalized "
        "train-validation overlaps. Connected-group filtering must supersede "
        "blind trust in the published split.",
        "7. No license or README member is present in either ZIP; licensing "
        "must be reconstructed from the original dataset releases before "
        "redistribution.",
        "",
        "## BanglaDial classification pool",
        "",
        f"Rows: {bangladial['rows']:,}; unique normalized texts: "
        f"{bangladial['unique_normalized_texts']:,}; label counts:",
        "",
        "| Label | Rows |",
        "|---|---:|",
    ]
    for label, count in bangladial["label_counts"].items():
        lines.append(f"| {label} | {count} |")

    lines += [
        "",
        f"Placeholder/punctuation-only rows: "
        f"{bangladial['placeholder_or_punctuation_only_rows']:,}; rows "
        f"containing angle-bracket placeholders: "
        f"{bangladial['rows_containing_angle_placeholder']:,}.",
        "",
        "## Derived archive",
        "",
        f"Rows: {derived['rows']:,}; unique normalized texts: "
        f"{derived['unique_normalized_texts']:,}; duplicate rows involved: "
        f"{derived['duplicate_rows_involved']:,}.",
        "",
        "| Member | Rows | Unique text | Duplicate rows | Cross-label texts | "
        "Cross-label rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in derived["per_file"]:
        lines.append(
            f"| {row['archive_member']} | {row['rows']} | "
            f"{row['unique_normalized_texts']} | "
            f"{row['duplicate_rows_involved']} | {row['cross_label_texts']} | "
            f"{row['cross_label_rows']} |"
        )

    lines += [
        "",
        "## Frozen local-data roles",
        "",
        "| Material | Main normalization | Main classification | Tokenizer | "
        "Ablation | Decision |",
        "|---|---|---|---|---|---|",
        "| Vashantor CSV train | yes, after connected filtering | yes | train "
        "portion only | yes | canonical |",
        "| Vashantor CSV validation/test | evaluation only | evaluation only | "
        "no | no | protected |",
        "| Vashantor JSON | no | no | no | no | exclude duplicate/malformed |",
        "| ChatgaiyyaAlap sentence pairs | yes, after grouped split | yes | "
        "train portion only | yes | canonical pair source |",
        "| Chatgaiyya dictionary | lexicon feature only | no | training lexicon "
        "only | yes | never count words as sentence examples |",
        "| Sylheti 1,200 pairs | yes, after grouped split | yes | train portion "
        "only | yes | canonical pair source |",
        "| BanglaRegionalTextCorpus | yes, once | yes | train portion only | "
        "yes | retain four-column workbook |",
        "| Regional_cleaned_dataset | no | no | no | no | duplicate exclusion |",
        "| BanglaDial | no until provenance reconstruction | conditional | "
        "conditional train-only | yes | remove protected overlaps/conflicts |",
        "| Entire derived archive | no | no | no | yes, after audit | "
        "quarantine |",
        "",
        "## Immediate data gate",
        "",
        "Before tokenizer or model training: build row-level provenance, assign "
        "semantic groups, remove every connected train-to-test path, repair "
        "BanglaDial placeholders/conflicts, freeze splits, and document licenses. "
        "The main model must never see the derived ZIP by default.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Directory containing both archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = workspace / RAW_ARCHIVE
    derived_path = workspace / DERIVED_ARCHIVE
    if not raw_path.exists() or not derived_path.exists():
        raise FileNotFoundError(
            f"Expected {raw_path} and {derived_path}"
        )

    pairs, bangladial, raw_diagnostics, _ = load_raw_datasets(raw_path)
    derived, derived_diagnostics = load_derived_archive(derived_path)
    overlaps = analyze_overlaps(pairs, bangladial, derived)

    label_pair_conflicts = bangladial.groupby("text")["dialect"].nunique()
    pair_dialects = sorted(pairs["dialect"].unique())
    result = {
        "audit_schema_version": "1.0.0",
        "archives": {
            RAW_ARCHIVE: archive_inventory(raw_path),
            DERIVED_ARCHIVE: archive_inventory(derived_path),
        },
        "raw_data": {
            "aligned_pair_rows": len(pairs),
            "unique_dialect_tagged_pairs": int(
                pairs[["dialect", "source_text", "standard_text"]]
                .drop_duplicates()
                .shape[0]
            ),
            "unique_source_texts": int(pairs["source_text"].nunique()),
            "unique_standard_targets": int(
                pairs["standard_text"].nunique()
            ),
            "normalization_dialects": pair_dialects,
            "normalization_dialect_rows": pairs.groupby("dialect")
            .size()
            .sort_index()
            .to_dict(),
        },
        "pair_summary": pair_summary(pairs),
        "raw_diagnostics": raw_diagnostics,
        "bangladial": bangladial_diagnostics(bangladial),
        "derived_data": derived_diagnostics,
        "overlaps": overlaps,
        "integrity_assertions": {
            "bangladial_cross_label_texts_recomputed": int(
                (label_pair_conflicts > 1).sum()
            ),
            "no_empty_aligned_pair": bool(
                (
                    (pairs["source_text"] != "")
                    & (pairs["standard_text"] != "")
                ).all()
            ),
            "normalization_dialect_count": len(pair_dialects),
            "license_or_readme_members": [
                item["name"]
                for archive in [
                    archive_inventory(raw_path),
                    archive_inventory(derived_path),
                ]
                for item in archive["members"]
                if any(
                    token in item["name"].lower()
                    for token in ["license", "readme", "citation"]
                )
            ],
        },
    }

    json_path = output_dir / "local_archive_audit.json"
    markdown_path = output_dir / "LOCAL_ARCHIVE_AUDIT.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
