#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import (  # noqa: E402
    build_training_maps,
    encode_clm,
    encode_identification,
    encode_normalization,
    save_encoded_cache,
)
from boichitro.tokenization import load_tokenizer, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize final task manifests without opening metrics.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--tokenizer", type=Path, default=PROJECT / "artifacts/tokenizers/frozen"
    )
    return parser.parse_args()


def records(frame: pd.DataFrame):
    yield from frame.to_dict("records")


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    data_dir = PROJECT / "data/final/v1"
    normalization_train = pd.read_parquet(data_dir / "normalization_train.parquet")
    identification_train = pd.read_parquet(data_dir / "identification_train.parquet")
    identification_evaluation = pd.read_parquet(
        data_dir / "identification_evaluation.parquet"
    )
    maps = build_training_maps(normalization_train, identification_train)
    output = PROJECT / "cache/tasks"
    output.mkdir(parents=True, exist_ok=True)
    (output / "maps.json").write_text(json.dumps(maps, indent=2) + "\n", encoding="utf-8")
    tokenizer_sha = sha256_file(args.tokenizer / "tokenizer.json")
    common = {
        "tokenizer_sha256": tokenizer_sha,
        "max_length": args.max_length,
        "metrics_computed": False,
        "test_predictions_computed": False,
    }

    specifications = [
        ("normalization_train", normalization_train, encode_normalization),
        (
            "normalization_validation",
            pd.read_parquet(data_dir / "normalization_validation.parquet"),
            encode_normalization,
        ),
        (
            "normalization_test_iid",
            pd.read_parquet(data_dir / "normalization_test_iid.parquet"),
            encode_normalization,
        ),
        (
            "normalization_test_ood",
            pd.read_parquet(data_dir / "normalization_test_ood.parquet"),
            encode_normalization,
        ),
        ("identification_train", identification_train, encode_identification),
        (
            "identification_evaluation",
            identification_evaluation,
            encode_identification,
        ),
    ]
    for split, name in (
        ("validation", "identification_validation"),
        ("test", "identification_test_iid"),
        ("test_ood", "identification_test_ood"),
        ("test_external", "identification_test_external"),
    ):
        specifications.append(
            (
                name,
                identification_evaluation.loc[
                    identification_evaluation["split"].eq(split)
                ].copy(),
                encode_identification,
            )
        )

    # Train-only real romanizations become a robustness component. They keep
    # the parent source/group identity but receive distinct row IDs so paired
    # sampling and prediction audits remain unambiguous.
    romanized = normalization_train.loc[
        normalization_train["source_romanized"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    romanized["row_id"] = romanized["row_id"].astype(str) + ":romanized_train"
    romanized["source_text_model"] = romanized["source_romanized"].astype(str)
    specifications.append(
        ("normalization_romanized_train", romanized, encode_normalization)
    )
    romanized_evaluation = pd.read_parquet(data_dir / "romanized_test_ood.parquet").copy()
    romanized_evaluation["source_text_model"] = romanized_evaluation[
        "romanized_input_model"
    ].astype(str)
    specifications.append(
        ("normalization_romanized_test_ood", romanized_evaluation, encode_normalization)
    )
    summary = []
    for name, frame, encoder in specifications:
        print(f"Encoding {name}: {len(frame):,} rows", flush=True)
        encoded = [
            encoder(
                row,
                tokenizer,
                max_length=args.max_length,
                source_to_id=maps["source_to_id"],
                group_to_id=maps["group_to_id"],
            )
            for row in records(frame)
        ]
        metadata = {**common, "name": name, "rows": len(encoded)}
        save_encoded_cache(output / f"{name}.pt", encoded, metadata)
        summary.append(metadata)

    # Train-only CLM component for dialect replay during specialization.
    clm = [
        encode_clm(
            row,
            tokenizer,
            max_length=args.max_length,
            source_to_id=maps["source_to_id"],
            group_to_id=maps["group_to_id"],
        )
        for row in records(identification_train)
    ]
    metadata = {**common, "name": "dialect_clm_train", "rows": len(clm)}
    save_encoded_cache(output / "dialect_clm_train.pt", clm, metadata)
    summary.append(metadata)
    (PROJECT / "reports/model/task_cache_report.json").write_text(
        json.dumps({"status": "PASS", "artifacts": summary, "maps": maps}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Task cache complete: {output}")


if __name__ == "__main__":
    main()
