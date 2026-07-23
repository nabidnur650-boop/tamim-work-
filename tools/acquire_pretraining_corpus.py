#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

import ahocorasick
import pandas as pd
import yaml
from datasets import load_dataset

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.tokenization import load_tokenizer, nfc, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire a pinned, decontaminated Bangla foundation corpus.")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/pretraining_corpus.yaml")
    parser.add_argument("--target-train-tokens", type=int)
    parser.add_argument("--target-validation-tokens", type=int)
    parser.add_argument("--max-raw-documents", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def compact(text: str) -> str:
    return "".join(character for character in nfc(text).casefold() if character.isalnum())


def protected_texts() -> tuple[set[str], set[str]]:
    data_dir = PROJECT / "data/final/v1"
    direct: set[str] = set()
    for filename, columns in (
        ("normalization_all.parquet", ("source_text_model", "target_text_model")),
        ("identification_all.parquet", ("text_model",)),
        (
            "romanized_test_ood.parquet",
            ("romanized_input_model", "bengali_reference", "target_text_model"),
        ),
    ):
        frame = pd.read_parquet(data_dir / filename, columns=list(columns))
        for column in columns:
            direct.update(nfc(value) for value in frame[column].dropna() if len(nfc(value)) >= 20)
    compacted = {compact(value) for value in direct if len(compact(value)) >= 20}
    return direct, compacted


def automaton(patterns: Iterable[str]) -> ahocorasick.Automaton:
    machine = ahocorasick.Automaton()
    for index, pattern in enumerate(sorted(set(patterns))):
        machine.add_word(pattern, index)
    machine.make_automaton()
    return machine


def bengali_letter_ratio(text: str) -> float:
    letters = [character for character in text if unicodedata.category(character).startswith("L")]
    if not letters:
        return 0.0
    bengali = sum("\u0980" <= character <= "\u09ff" for character in letters)
    return bengali / len(letters)


def quality_reason(text: str, row: dict, config: dict) -> str | None:
    if len(text) < int(config["minimum_characters"]):
        return "too_short"
    if len(text) > int(config["maximum_characters"]):
        return "too_long"
    if float(row.get("language_score") or 0.0) < float(config["minimum_language_score"]):
        return "low_language_score"
    if bengali_letter_ratio(text) < float(config["minimum_bengali_letter_ratio"]):
        return "low_bengali_letter_ratio"
    if "\ufffd" in text or any(unicodedata.category(character) == "Cs" for character in text):
        return "invalid_unicode"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.35:
        return "repetitive_lines"
    return None


def split_for(document_id: str, per_mille: int) -> str:
    value = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:16], 16) % 1000
    return "validation" if value < per_mille else "train"


def stream_file(config: dict, filename: str):
    local_root = config.get("local_raw_dir")
    local_path = (PROJECT / local_root / filename).resolve() if local_root else None
    if local_path is not None and local_path.exists():
        return load_dataset("parquet", data_files=str(local_path), split="train", streaming=True)
    revision = config["revision"]
    dataset_id = config["dataset_id"]
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{filename}"
    return load_dataset("parquet", data_files=url, split="train", streaming=True)


def balanced_quota(total: int, index: int, count: int) -> int:
    """Split a fixed token target exactly across pinned source shards."""

    return total // count + int(index < total % count)


def verify_local_sources(config: dict) -> list[dict[str, object]]:
    records = []
    local_root = config.get("local_raw_dir")
    expected_hashes = config.get("source_lfs_sha256", {})
    for filename in config["train_files"]:
        path = (PROJECT / local_root / filename).resolve() if local_root else None
        if path is None or not path.exists():
            records.append({"source_file": filename, "mode": "revision_pinned_remote_stream"})
            continue
        actual = sha256_file(path)
        expected = expected_hashes.get(filename)
        if expected and actual != expected:
            raise RuntimeError(
                f"Local source hash mismatch for {filename}: {actual} != {expected}"
            )
        records.append(
            {
                "source_file": filename,
                "mode": "verified_local_parquet",
                "path": str(path.relative_to(PROJECT)),
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return records


def write_shard(rows: list[dict], output_dir: Path, index: int) -> dict:
    path = output_dir / f"shard_{index:05d}.parquet"
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False, compression="zstd")
    return {
        "path": str(path.relative_to(PROJECT)),
        "rows": len(frame),
        "tokens": int(frame["token_count"].sum()),
        "characters": int(frame["character_count"].sum()),
        "train_rows": int(frame["split"].eq("train").sum()),
        "validation_rows": int(frame["split"].eq("validation").sum()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    target_train = args.target_train_tokens or int(config["target_train_tokens"])
    target_validation = args.target_validation_tokens or int(config["target_validation_tokens"])
    tokenizer = load_tokenizer(PROJECT / "artifacts/tokenizers/frozen")
    direct, compacted = protected_texts()
    print(f"Building decontamination automata: direct={len(direct):,}, compact={len(compacted):,}")
    direct_machine = automaton(direct)
    compact_machine = automaton(compacted)
    default_output_dir = (PROJECT / "data/pretraining/fineweb2_bn_v1").resolve()
    output_dir = (args.output_dir or default_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("shard_*.parquet")):
        raise RuntimeError(
            f"Output already has shards ({output_dir}); move them explicitly before rebuilding"
        )
    source_verification = verify_local_sources(config)

    rejection = Counter()
    shard_rows: list[dict] = []
    shards: list[dict] = []
    tokens = {"train": 0, "validation": 0}
    rows = {"train": 0, "validation": 0}
    raw_documents = 0
    accepted_documents = 0
    started = time.perf_counter()
    files = list(config["train_files"])
    per_file: list[dict] = []
    stop_for_raw_limit = False
    for file_index, filename in enumerate(files):
        file_targets = {
            "train": balanced_quota(target_train, file_index, len(files)),
            "validation": balanced_quota(target_validation, file_index, len(files)),
        }
        file_tokens = {"train": 0, "validation": 0}
        file_rows = {"train": 0, "validation": 0}
        file_raw = 0
        dataset = stream_file(config, filename)
        for raw in dataset:
            raw_documents += 1
            file_raw += 1
            if args.max_raw_documents and raw_documents > args.max_raw_documents:
                stop_for_raw_limit = True
                break
            text = nfc(raw.get("text", ""))
            reason = quality_reason(text, raw, config)
            if reason is None and next(direct_machine.iter(text), None) is not None:
                reason = "benchmark_direct_substring"
            if reason is None and next(compact_machine.iter(compact(text)), None) is not None:
                reason = "benchmark_compact_substring"
            if reason is not None:
                rejection[reason] += 1
                continue

            document_id = str(raw["id"])
            split = split_for(document_id, int(config["validation_hash_per_mille"]))
            if file_tokens[split] >= file_targets[split]:
                continue
            token_count = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2
            record = {
                "document_id": document_id,
                "text": text,
                "split": split,
                "token_count": token_count,
                "character_count": len(text),
                "byte_count": len(text.encode("utf-8")),
                "url": str(raw.get("url", "")),
                "dump": str(raw.get("dump", "")),
                "date": str(raw.get("date", "")),
                "language_score": float(raw.get("language_score") or 0.0),
                "minhash_cluster_size": int(raw.get("minhash_cluster_size") or 1),
                "source_dataset": config["dataset_id"],
                "source_revision": config["revision"],
                "source_file": filename,
                "license": config["license"],
            }
            shard_rows.append(record)
            tokens[split] += token_count
            rows[split] += 1
            file_tokens[split] += token_count
            file_rows[split] += 1
            accepted_documents += 1
            if len(shard_rows) >= int(config["shard_documents"]):
                shard = write_shard(shard_rows, output_dir, len(shards))
                shards.append(shard)
                shard_rows = []
                elapsed = time.perf_counter() - started
                print(
                    f"raw={raw_documents:,} accepted={accepted_documents:,} "
                    f"train_tok={tokens['train']:,}/{target_train:,} "
                    f"val_tok={tokens['validation']:,}/{target_validation:,} "
                    f"docs/s={raw_documents / elapsed:,.1f}",
                    flush=True,
                )
            if all(file_tokens[split] >= file_targets[split] for split in file_targets):
                break
        per_file.append(
            {
                "source_file": filename,
                "raw_documents_seen": file_raw,
                "train_rows": file_rows["train"],
                "validation_rows": file_rows["validation"],
                "train_tokens": file_tokens["train"],
                "validation_tokens": file_tokens["validation"],
                "target_train_tokens": file_targets["train"],
                "target_validation_tokens": file_targets["validation"],
            }
        )
        print(
            f"finished source {file_index + 1}/{len(files)}: {filename} "
            f"train_tok={file_tokens['train']:,} val_tok={file_tokens['validation']:,}",
            flush=True,
        )
        del dataset
        gc.collect()
        if stop_for_raw_limit:
            break
    if shard_rows:
        shards.append(write_shard(shard_rows, output_dir, len(shards)))

    status = (
        "PASS_TARGETS"
        if tokens["train"] >= target_train and tokens["validation"] >= target_validation
        else "INCOMPLETE"
    )
    report = {
        "status": status,
        "dataset_id": config["dataset_id"],
        "revision": config["revision"],
        "subset": config["subset"],
        "license": config["license"],
        "common_crawl_terms_apply": bool(config["common_crawl_terms_apply"]),
        "source_lfs_sha256": config.get("source_lfs_sha256", {}),
        "source_verification": source_verification,
        "tokenizer_sha256": sha256_file(PROJECT / "artifacts/tokenizers/frozen/tokenizer.json"),
        "raw_documents_seen": raw_documents,
        "accepted_documents": accepted_documents,
        "train_rows": rows["train"],
        "validation_rows": rows["validation"],
        "train_tokens": tokens["train"],
        "validation_tokens": tokens["validation"],
        "target_train_tokens": target_train,
        "target_validation_tokens": target_validation,
        "rejections": dict(rejection),
        "direct_decontamination_patterns": len(direct),
        "compact_decontamination_patterns": len(compacted),
        "per_source_file": per_file,
        "shards": shards,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Ad-hoc smoke builds retain their own manifest but must never overwrite the
    # canonical report consumed by the experiment registry.
    if output_dir == default_output_dir:
        report_dir = PROJECT / "reports/model"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "pretraining_corpus_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({key: report[key] for key in report if key != "shards"}, indent=2))
    return 0 if status == "PASS_TARGETS" else 2


if __name__ == "__main__":
    exit_code = main()
    # huggingface_hub's Xet range reader may retain an aiohttp helper thread
    # after an intentionally shortened streaming scan on aarch64. All artifacts
    # are atomically closed above; a direct clean exit avoids a CPython shutdown
    # race without weakening acquisition checks or revision pinning.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
