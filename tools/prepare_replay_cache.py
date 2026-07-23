#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import encode_clm, save_encoded_cache  # noqa: E402
from boichitro.tokenization import load_tokenizer, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic train-only general-Bangla replay cache."
    )
    parser.add_argument(
        "--corpus", type=Path, default=PROJECT / "data/pretraining/fineweb2_bn_v1"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "cache/tasks/general_replay_train.pt"
    )
    parser.add_argument("--max-examples", type=int, default=60000)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def stable_priority(document_id: str) -> int:
    return int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:16], 16)


def main() -> None:
    args = parse_args()
    manifest_path = args.corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS_TARGETS":
        raise RuntimeError("General replay source did not pass the corpus gate")
    # Max-heap via negative priorities retains the globally smallest stable
    # hashes, preventing source-order bias in the sequential acquisition shards.
    selected: list[tuple[int, str, str]] = []
    train_seen = 0
    for path in sorted(args.corpus.glob("shard_*.parquet")):
        frame = pd.read_parquet(path, columns=["document_id", "text", "split"])
        for row in frame.loc[frame["split"].eq("train")].itertuples(index=False):
            train_seen += 1
            priority = stable_priority(str(row.document_id))
            item = (-priority, str(row.document_id), str(row.text))
            if len(selected) < args.max_examples:
                heapq.heappush(selected, item)
            elif priority < -selected[0][0]:
                heapq.heapreplace(selected, item)
    rows = [
        {
            "row_id": f"fineweb2:{document_id}",
            "text_model": text,
            "dialect": "STD",
            "source_id": "general_replay",
            "is_synthetic": False,
            "example_loss_weight": 1.0,
        }
        for _, document_id, text in sorted(selected, key=lambda item: (-item[0], item[1]))
    ]
    tokenizer_dir = PROJECT / "artifacts/tokenizers/frozen"
    tokenizer = load_tokenizer(tokenizer_dir)
    encoded = []
    for row in rows:
        example = encode_clm(
            row,
            tokenizer,
            max_length=args.max_length,
            source_to_id={},
            group_to_id={},
        )
        # FineWeb is used only for LM replay; do not assign dialect or source
        # labels to web documents whose regional provenance is unknown.
        example.dialect_label = -100
        example.source_label = -100
        example.group_id = -100
        encoded.append(example)
    metadata = {
        "name": "general_replay_train",
        "rows": len(encoded),
        "train_documents_seen": train_seen,
        "selection": "bottom_k_sha256_document_id",
        "max_length": args.max_length,
        "tokenizer_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
        "corpus_manifest_sha256": sha256_file(manifest_path),
        "test_data_access": False,
    }
    save_encoded_cache(args.output, encoded, metadata)
    report_path = PROJECT / "reports/model/general_replay_cache_report.json"
    report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
