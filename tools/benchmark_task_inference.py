#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import EncodedDataset, load_encoded_cache  # noqa: E402
from boichitro.inference import greedy_normalize, predict_identification  # noqa: E402
from boichitro.modeling import BoichitroConfig, BoichitroForMultiTask  # noqa: E402
from boichitro.tokenization import load_tokenizer, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end task inference on validation-only inputs."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/inference_benchmark.yaml"
    )
    return parser.parse_args()


def hash_subset_frame(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    order = sorted(
        range(len(frame)),
        key=lambda index: hashlib.sha256(str(frame.iloc[index]["row_id"]).encode()).digest(),
    )[:count]
    return frame.iloc[order].copy()


def hash_subset_dataset(dataset: EncodedDataset, count: int) -> EncodedDataset:
    order = sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(
            str(dataset.examples[index]["row_id"]).encode()
        ).digest(),
    )[:count]
    return EncodedDataset([dataset.examples[index] for index in order])


def load_model(path: Path, device: torch.device) -> BoichitroForMultiTask:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = BoichitroForMultiTask(BoichitroConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def selected_normalization_checkpoint(root: Path) -> Path:
    candidate = root / "stage_s/best_checkpoint.pt"
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    guard = bool(payload.get("extra", {}).get("validation", {}).get("replay_guard_pass"))
    return candidate if guard else root / "stage_a/last_checkpoint.pt"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(PROJECT / "artifacts/tokenizers/frozen")
    pad_token_id = int(tokenizer.token_to_id("<pad>"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    norm_values = config["normalization"]
    norm_frame = hash_subset_frame(
        pd.read_parquet(PROJECT / norm_values["validation_frame"]),
        int(norm_values["examples"]),
    )
    norm_texts = norm_frame["source_text_model"].astype(str).tolist()
    id_values = config["identification"]
    id_dataset = hash_subset_dataset(
        load_encoded_cache(PROJECT / id_values["validation_cache"])[0],
        int(id_values["examples"]),
    )
    rows = []
    seed = int(config["seed"])
    for variant in config["variants"]:
        root = (
            PROJECT
            / "runs/task"
            / str(config["task_protocol_id"])
            / f"{variant}__base"
            / str(seed)
        )
        norm_checkpoint = selected_normalization_checkpoint(root)
        id_checkpoint = root / "stage_id/best_checkpoint.pt"

        model = load_model(norm_checkpoint, device)
        greedy_normalize(
            model,
            tokenizer,
            norm_texts[: int(norm_values["warmup_examples"])],
            device=device,
            batch_size=min(8, int(norm_values["warmup_examples"])),
            max_new_tokens=8,
        )
        for batch_size in norm_values["batch_sizes"]:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            synchronize(device)
            started = time.perf_counter()
            predictions = greedy_normalize(
                model,
                tokenizer,
                norm_texts,
                device=device,
                batch_size=int(batch_size),
                max_new_tokens=int(norm_values["max_new_tokens"]),
            )
            synchronize(device)
            elapsed = time.perf_counter() - started
            generated_tokens = sum(
                len(tokenizer.encode(value, add_special_tokens=False).ids)
                for value in predictions
            )
            rows.append(
                {
                    "variant": variant,
                    "task": "normalization",
                    "batch_size": int(batch_size),
                    "examples": len(norm_texts),
                    "elapsed_seconds": elapsed,
                    "examples_per_second": len(norm_texts) / elapsed,
                    "milliseconds_per_example": elapsed * 1000 / len(norm_texts),
                    "generated_tokens": generated_tokens,
                    "generated_tokens_per_second": generated_tokens / elapsed,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30
                    if device.type == "cuda"
                    else 0.0,
                    "checkpoint_sha256": sha256_file(norm_checkpoint),
                    **model.parameter_report(),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model = load_model(id_checkpoint, device)
        warmup = EncodedDataset(
            id_dataset.examples[: int(id_values["warmup_examples"])]
        )
        predict_identification(
            model,
            warmup,
            device=device,
            pad_token_id=pad_token_id,
            batch_size=len(warmup),
        )
        for batch_size in id_values["batch_sizes"]:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            synchronize(device)
            started = time.perf_counter()
            predict_identification(
                model,
                id_dataset,
                device=device,
                pad_token_id=pad_token_id,
                batch_size=int(batch_size),
            )
            synchronize(device)
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "variant": variant,
                    "task": "identification",
                    "batch_size": int(batch_size),
                    "examples": len(id_dataset),
                    "elapsed_seconds": elapsed,
                    "examples_per_second": len(id_dataset) / elapsed,
                    "milliseconds_per_example": elapsed * 1000 / len(id_dataset),
                    "generated_tokens": 0,
                    "generated_tokens_per_second": 0.0,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30
                    if device.type == "cuda"
                    else 0.0,
                    "checkpoint_sha256": sha256_file(id_checkpoint),
                    **model.parameter_report(),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    output_dir = PROJECT / "reports/model"
    frame.to_csv(output_dir / "task_inference_benchmark.csv", index=False)
    report = {
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "device": str(device),
        "test_data_access": False,
        "records": frame.to_dict("records"),
    }
    (output_dir / "task_inference_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
