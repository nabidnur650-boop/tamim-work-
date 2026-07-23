#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.tokenization import load_tokenizer, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack the frozen FineWeb2 subset into CLM blocks.")
    parser.add_argument(
        "--corpus", type=Path, default=PROJECT / "data/pretraining/fineweb2_bn_v1"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "data/pretraining/packed_1024"
    )
    parser.add_argument("--block-size", type=int, default=1024)
    return parser.parse_args()


def pack_split(
    files: list[Path],
    split: str,
    tokenizer,
    output_path: Path,
    block_size: int,
) -> dict:
    bos = int(tokenizer.token_to_id("<bos>"))
    eos = int(tokenizer.token_to_id("<eos>"))
    buffer: list[int] = []
    blocks = 0
    documents = 0
    raw_tokens = 0
    characters = 0
    bytes_count = 0
    digest = hashlib.sha256()
    started = time.perf_counter()
    with output_path.open("wb") as output:
        for file_path in files:
            frame = pd.read_parquet(file_path)
            frame = frame.loc[frame["split"] == split]
            for row in frame.itertuples(index=False):
                ids = tokenizer.encode(row.text, add_special_tokens=False).ids
                document = [bos, *ids, eos]
                buffer.extend(document)
                documents += 1
                raw_tokens += len(document)
                characters += int(row.character_count)
                bytes_count += int(row.byte_count)
                digest.update(str(row.document_id).encode("utf-8"))
                digest.update(b"\n")
                complete = len(buffer) // block_size
                if complete:
                    token_count = complete * block_size
                    array = np.asarray(buffer[:token_count], dtype=np.uint16)
                    array.tofile(output)
                    blocks += complete
                    del buffer[:token_count]
            print(
                f"[{split}] {file_path.name}: docs={documents:,} blocks={blocks:,}",
                flush=True,
            )
    return {
        "filename": output_path.name,
        "sha256": sha256_file(output_path),
        "documents": documents,
        "raw_tokens_with_boundaries": raw_tokens,
        "blocks": blocks,
        "packed_tokens": blocks * block_size,
        "dropped_tail_tokens": len(buffer),
        "characters": characters,
        "bytes": bytes_count,
        "document_order_sha256": digest.hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    corpus_manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    if corpus_manifest["status"] != "PASS_TARGETS":
        raise RuntimeError("Pretraining corpus did not pass its target gate")
    tokenizer_path = PROJECT / "artifacts/tokenizers/frozen/tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise ValueError("uint16 packing cannot represent this vocabulary")
    args.output.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(args.corpus.glob("shard_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No corpus shards in {args.corpus}")
    splits = {}
    for split in ("train", "validation"):
        path = args.output / f"{split}.uint16.bin"
        if path.exists():
            raise RuntimeError(f"Packed output exists: {path}")
        splits[split] = pack_split(
            shard_files, split, tokenizer, path, args.block_size
        )
    metadata = {
        "status": "PASS",
        "block_size": args.block_size,
        "dtype": "uint16",
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "corpus_manifest_sha256": sha256_file(args.corpus / "manifest.json"),
        "splits": splits,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if args.output.resolve() == (PROJECT / "data/pretraining/packed_1024").resolve():
        (PROJECT / "reports/model/pretraining_packing_report.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
