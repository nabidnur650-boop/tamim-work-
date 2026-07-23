#!/usr/bin/env python3
"""Audit cross-task input memorization without emitting protected text."""
from __future__ import annotations

import datetime as dt
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

from build_final_dataset import near_match_map


PROJECT = Path(__file__).resolve().parents[1]


def normalized(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def compact(text: object) -> str:
    return re.sub(r"\W+", "", normalized(text), flags=re.UNICODE)


def main() -> None:
    root = PROJECT / "data/final/v1"
    identification_train = pd.read_parquet(
        root / "identification_train.parquet",
        columns=["row_id", "text_model", "text_compact"],
    )
    config = yaml.safe_load(
        (PROJECT / "configs/dataset_build.yaml").read_text(encoding="utf-8")
    )
    id_exact = set(identification_train["text_model"].map(normalized))
    id_compact = set(identification_train["text_compact"].astype(str)) - {""}
    reference_rows = [
        {"row_id": str(row.row_id), "text_compact": str(row.text_compact)}
        for row in identification_train.itertuples(index=False)
    ]
    tracks = (
        (
            "normalization_validation",
            root / "normalization_validation.parquet",
            "source_text_model",
            "source_text_compact",
        ),
        (
            "normalization_iid_test",
            root / "normalization_test_iid.parquet",
            "source_text_model",
            "source_text_compact",
        ),
        (
            "normalization_ood_and_zero_shot",
            root / "normalization_test_ood.parquet",
            "source_text_model",
            "source_text_compact",
        ),
        (
            "romanized_ood",
            root / "romanized_test_ood.parquet",
            "romanized_input_model",
            None,
        ),
    )
    rows = []
    for track, path, text_column, compact_column in tracks:
        columns = ["row_id", text_column] + ([compact_column] if compact_column else [])
        frame = pd.read_parquet(path, columns=columns)
        exact_values = frame[text_column].map(normalized)
        compact_values = (
            frame[compact_column].astype(str)
            if compact_column
            else frame[text_column].map(compact)
        )
        query_rows = [
            {"row_id": str(row_id), "text_compact": str(value)}
            for row_id, value in zip(frame["row_id"], compact_values, strict=True)
        ]
        near_matches = near_match_map(
            query_rows,
            reference_rows,
            compact_field="text_compact",
            text_field="input",
            config=config,
        )
        near_scores = [float(match[1]) for match in near_matches.values()]
        rows.append(
            {
                "track": track,
                "rows": int(len(frame)),
                "exact_overlap_rows_with_id_train": int(exact_values.isin(id_exact).sum()),
                "compact_overlap_rows_with_id_train": int(
                    compact_values.isin(id_compact).sum()
                ),
                "near_overlap_rows_with_id_train": int(len(near_matches)),
                "maximum_near_jaccard": max(near_scores, default=None),
            }
        )
    passed = all(
        row["exact_overlap_rows_with_id_train"] == 0
        and row["compact_overlap_rows_with_id_train"] == 0
        and row["near_overlap_rows_with_id_train"] == 0
        for row in rows
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "protocol_id": "boichitro_cross_task_input_firewall_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "identification_training_rows": int(len(identification_train)),
        "scope": (
            "automated aggregate-only overlap audit between the frozen ID-classifier "
            "training inputs and normalization evaluation inputs"
        ),
        "protected_text_emitted": False,
        "model_or_hyperparameter_selection_use": False,
        "prior_test_access_disclosure_applies": True,
        "near_duplicate_method": {
            "simhash_bits": int(config["deduplication"]["simhash_bits"]),
            "character_ngram": int(
                config["deduplication"]["simhash_character_ngram"]
            ),
            "maximum_hamming_candidate": int(
                config["deduplication"]["simhash_max_hamming_candidate"]
            ),
            "minimum_jaccard": float(
                config["deduplication"]["near_jaccard_threshold"]
            ),
            "minimum_compact_characters": int(
                config["deduplication"]["near_minimum_compact_characters"]
            ),
        },
        "tracks": rows,
    }
    destination = PROJECT / "reports/model/cross_task_input_firewall.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("Cross-task input overlap detected")


if __name__ == "__main__":
    main()
