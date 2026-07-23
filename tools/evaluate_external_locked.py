#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import DIALECT_TO_ID  # noqa: E402
from boichitro.metrics import classification_metrics, normalization_metrics  # noqa: E402
from boichitro.protocol import require_frozen_artifact, require_protocol_freeze  # noqa: E402
from boichitro.tokenization import nfc, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen external baselines once on the locked test tracks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/external_locked_evaluation.yaml",
    )
    parser.add_argument("--tasks", nargs="+", choices=("normalization", "identification"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--allow-identical-rerun", action="store_true")
    return parser.parse_args()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No frozen model files under {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def prepare_output(
    path: Path,
    *,
    checkpoint_hash: str,
    allow_identical_rerun: bool,
) -> bool:
    manifest_path = path / "evaluation_manifest.json"
    if not manifest_path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return True
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("checkpoint_tree_sha256") != checkpoint_hash:
        raise RuntimeError(
            f"Refusing to overwrite outputs from a different frozen model: {path}"
        )
    if allow_identical_rerun:
        return True
    print(f"Locked external evaluation already complete, skipping {path}", flush=True)
    return False


class EncodedRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


@torch.inference_mode()
def evaluate_normalization(
    model,
    tokenizer,
    frame: pd.DataFrame,
    *,
    prefix: str,
    max_source_length: int,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    source_column = (
        "romanized_input_model"
        if "romanized_input_model" in frame.columns
        else "source_text_model"
    )
    sources = frame[source_column].fillna("").astype(str).map(nfc).tolist()
    predictions: list[str] = []
    model.eval()
    for start in range(0, len(sources), batch_size):
        encoded = tokenizer(
            [prefix + value for value in sources[start : start + batch_size]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )
        predictions.extend(
            nfc(value).strip()
            for value in tokenizer.batch_decode(generated, skip_special_tokens=True)
        )
    output = pd.DataFrame(
        {
            "row_id": frame["row_id"].astype(str),
            "semantic_group_id": frame.get("semantic_group_id", frame["row_id"]).astype(str),
            "dialect": frame["dialect"].astype(str),
            "source_id": frame["source_id"].astype(str),
            "source": sources,
            "reference": frame["target_text_model"].fillna("").astype(str).map(nfc),
            "prediction": predictions,
        }
    )
    metrics, by_dialect = normalization_metrics(output)
    return metrics, by_dialect, output


def encode_identification(frame: pd.DataFrame, tokenizer, max_length: int) -> list[dict]:
    encoded = tokenizer(
        frame["text_model"].fillna("").astype(str).map(nfc).tolist(),
        truncation=True,
        max_length=max_length,
    )
    rows = []
    for index, row in frame.reset_index(drop=True).iterrows():
        rows.append(
            {
                "input_ids": encoded["input_ids"][index],
                "attention_mask": encoded["attention_mask"][index],
                "label_id": DIALECT_TO_ID[str(row["dialect"])],
                "row_id": str(row["row_id"]),
                "dialect": str(row["dialect"]),
                "source_id": str(row["source_id"]),
                "semantic_group_id": str(row["row_id"]),
            }
        )
    return rows


@torch.inference_mode()
def evaluate_identification(
    model,
    tokenizer,
    frame: pd.DataFrame,
    *,
    max_length: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, pd.DataFrame]:
    metadata_keys = ("row_id", "dialect", "source_id", "semantic_group_id", "label_id")
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def collate(examples):
        metadata = {key: [row[key] for row in examples] for key in metadata_keys}
        features = [
            {key: value for key, value in row.items() if key not in metadata_keys}
            for row in examples
        ]
        return collator(features), metadata

    loader = DataLoader(
        EncodedRows(encode_identification(frame, tokenizer, max_length)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )
    rows = []
    model.eval()
    for batch, metadata in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**batch).logits.float() / max(1e-6, temperature)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        for index, prediction in enumerate(predictions):
            rows.append(
                {
                    "row_id": metadata["row_id"][index],
                    "semantic_group_id": metadata["semantic_group_id"][index],
                    "dialect": metadata["dialect"][index],
                    "source_id": metadata["source_id"][index],
                    "label_id": int(metadata["label_id"][index]),
                    "prediction_id": int(prediction),
                    "probabilities": probabilities[index].tolist(),
                }
            )
    output = pd.DataFrame(rows)
    metrics, by_class, matrix = classification_metrics(
        output["label_id"].to_numpy(),
        output["prediction_id"].to_numpy(),
        np.asarray(output["probabilities"].tolist(), dtype=np.float64),
    )
    return metrics, by_class, matrix, output


def save_normalization_results(
    output_dir: Path,
    track: str,
    metrics: dict,
    by_dialect: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    predictions.to_parquet(output_dir / f"normalization_{track}.parquet", index=False)
    by_dialect.to_csv(output_dir / f"normalization_{track}_by_dialect.csv", index=False)
    (output_dir / f"normalization_{track}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def save_identification_results(
    output_dir: Path,
    track: str,
    metrics: dict,
    by_class: pd.DataFrame,
    matrix: np.ndarray,
    predictions: pd.DataFrame,
) -> None:
    predictions.to_parquet(output_dir / f"identification_{track}.parquet", index=False)
    by_class.to_csv(output_dir / f"identification_{track}_by_class.csv", index=False)
    np.save(output_dir / f"identification_{track}_confusion.npy", matrix)
    (output_dir / f"identification_{track}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    freeze = (
        require_protocol_freeze(PROJECT, str(config["protocol_freeze_id"]))
        if config.get("protocol_freeze_id")
        else None
    )
    baseline_config_path = PROJECT / config["baseline_config"]
    baselines = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    tasks = args.tasks or ["normalization", "identification"]
    seeds = args.seeds or [int(seed) for seed in baselines["seeds"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    for task in tasks:
        values = baselines[task]
        model_names = args.models or list(values["models"])
        unknown = sorted(set(model_names) - set(values["models"]))
        if unknown:
            raise ValueError(f"Unknown {task} external model(s): {unknown}")
        for model_name in model_names:
            specification = values["models"][model_name]
            for seed in seeds:
                run_dir = PROJECT / "runs/external" / task / model_name / str(seed)
                best_dir = run_dir / "best_model"
                report_path = run_dir / "training_report.json"
                if not report_path.exists() or not best_dir.exists():
                    raise FileNotFoundError(f"Frozen external run missing under {run_dir}")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("status") != "COMPLETE_VALIDATION_ONLY":
                    raise RuntimeError(f"External run is not frozen: {run_dir}")
                if freeze is not None:
                    require_frozen_artifact(PROJECT, freeze, report_path)
                    for model_file in sorted(
                        path for path in best_dir.rglob("*") if path.is_file()
                    ):
                        require_frozen_artifact(PROJECT, freeze, model_file)
                    calibration_path = run_dir / "temperature_calibration.json"
                    if task == "identification":
                        require_frozen_artifact(PROJECT, freeze, calibration_path)
                checkpoint_hash = tree_sha256(best_dir)
                output_dir = (
                    PROJECT
                    / "predictions"
                    / str(config["protocol_id"])
                    / task
                    / model_name
                    / str(seed)
                )
                if not prepare_output(
                    output_dir,
                    checkpoint_hash=checkpoint_hash,
                    allow_identical_rerun=args.allow_identical_rerun,
                ):
                    continue
                tokenizer = AutoTokenizer.from_pretrained(best_dir)
                results = {}
                if task == "normalization":
                    model = AutoModelForSeq2SeqLM.from_pretrained(best_dir).to(device)
                    for track, details in config["normalization"]["tracks"].items():
                        if isinstance(details, str):
                            details = {"frame": details}
                        frame = pd.read_parquet(PROJECT / details["frame"])
                        if details.get("include_dialects"):
                            frame = frame.loc[
                                frame["dialect"].isin(details["include_dialects"])
                            ].copy()
                        if details.get("exclude_dialects"):
                            frame = frame.loc[
                                ~frame["dialect"].isin(details["exclude_dialects"])
                            ].copy()
                        if frame.empty:
                            raise ValueError(
                                f"Dialect filter produced an empty external track: {track}"
                            )
                        result = evaluate_normalization(
                            model,
                            tokenizer,
                            frame,
                            prefix=str(specification["prefix"]),
                            max_source_length=int(values["max_source_length"]),
                            batch_size=int(config["normalization"]["batch_size"]),
                            max_new_tokens=int(config["normalization"]["max_new_tokens"]),
                            device=device,
                        )
                        results[track] = result[0]
                        save_normalization_results(output_dir, track, *result)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
                    calibration = json.loads(
                        (run_dir / "temperature_calibration.json").read_text(encoding="utf-8")
                    )
                    temperature = float(calibration["temperature"])
                    full_frame = pd.read_parquet(PROJECT / config["identification"]["frame"])
                    for track, split in config["identification"]["tracks"].items():
                        frame = full_frame.loc[full_frame["split"].eq(split)].copy()
                        result = evaluate_identification(
                            model,
                            tokenizer,
                            frame,
                            max_length=int(values["max_length"]),
                            batch_size=int(config["identification"]["batch_size"]),
                            temperature=temperature,
                            device=device,
                        )
                        results[track] = result[0]
                        save_identification_results(output_dir, track, *result)
                del model, tokenizer
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                manifest = {
                    "status": "COMPLETE_LOCKED_EXTERNAL_EVALUATION",
                    "protocol_id": config["protocol_id"],
                    "task": task,
                    "model_name": model_name,
                    "model_id": specification["model_id"],
                    "revision": specification["revision"],
                    "license": specification["license"],
                    "seed": seed,
                    "checkpoint_tree_sha256": checkpoint_hash,
                    "training_report_sha256": sha256_file(report_path),
                    "baseline_config_sha256": sha256_file(baseline_config_path),
                    "evaluation_config_sha256": sha256_file(args.config),
                    "protocol_freeze": {
                        "protocol_id": freeze["protocol_id"],
                        "code_sha256": freeze["code_sha256"],
                        "config_sha256": freeze["config_sha256"],
                        "selected_artifacts_sha256": freeze[
                            "selected_artifacts_sha256"
                        ],
                    }
                    if freeze is not None
                    else None,
                    "results": results,
                    "test_policy": config["test_policy"],
                }
                (output_dir / "evaluation_manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"Locked external evaluation complete: {task} {model_name} seed {seed}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
