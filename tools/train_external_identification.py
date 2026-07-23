#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import DIALECT_TO_ID  # noqa: E402
from boichitro.metrics import classification_metrics  # noqa: E402
from boichitro.tokenization import DIALECTS, nfc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune registered external encoder baselines.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/external_model_baselines.yaml"
    )
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class Rows(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(frame: pd.DataFrame, tokenizer, max_length: int) -> list[dict]:
    tokenized = tokenizer(
        [nfc(value) for value in frame["text_model"].astype(str)],
        truncation=True,
        max_length=max_length,
    )
    return [
        {
            "input_ids": tokenized["input_ids"][index],
            "attention_mask": tokenized["attention_mask"][index],
            "labels": DIALECT_TO_ID[str(frame.iloc[index]["dialect"])],
            "row_id": str(frame.iloc[index]["row_id"]),
            "dialect": str(frame.iloc[index]["dialect"]),
            "source_id": str(frame.iloc[index]["source_id"]),
            "semantic_group_id": str(frame.iloc[index]["row_id"]),
        }
        for index in range(len(frame))
    ]


def build_loader(rows, tokenizer, batch_size: int, shuffle: bool, seed: int):
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def collate(examples):
        metadata = {
            key: [row[key] for row in examples]
            for key in ("row_id", "dialect", "source_id", "semantic_group_id")
        }
        features = [
            {key: value for key, value in row.items() if key not in metadata}
            for row in examples
        ]
        batch = collator(features)
        return batch, metadata

    return DataLoader(
        Rows(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )


@torch.inference_mode()
def predict(model, loader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch, metadata in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits.float()
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        for index in range(len(predictions)):
            rows.append(
                {
                    "row_id": metadata["row_id"][index],
                    "dialect": metadata["dialect"][index],
                    "source_id": metadata["source_id"][index],
                    "semantic_group_id": metadata["semantic_group_id"][index],
                    "label_id": int(labels[index]),
                    "prediction_id": int(predictions[index]),
                    "probabilities": probabilities[index].tolist(),
                }
            )
    return pd.DataFrame(rows)


def fit_temperature(frame: pd.DataFrame) -> dict[str, float | bool]:
    probabilities = np.asarray(frame["probabilities"].tolist(), dtype=np.float64)
    labels = frame["label_id"].to_numpy(dtype=np.int64)
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))

    def objective(log_temperature: float) -> float:
        logits = log_probabilities / math.exp(log_temperature)
        logits -= logits.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(logits).sum(axis=1))
        return float(-(logits[np.arange(len(labels)), labels] - log_norm).mean())

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return {
        "temperature": float(math.exp(result.x)),
        "validation_nll": float(result.fun),
        "optimizer_success": bool(result.success),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    values = config["identification"]
    model_names = args.models or list(values["models"])
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    train_frame = pd.read_parquet(PROJECT / "data/final/v1/identification_train.parquet")
    validation_frame = pd.read_parquet(
        PROJECT / "data/final/v1/identification_evaluation.parquet"
    )
    validation_frame = validation_frame.loc[validation_frame["split"].eq("validation")].copy()
    counts = train_frame["dialect"].map(DIALECT_TO_ID).value_counts().sort_index()
    class_weights = np.sqrt(len(train_frame) / (len(DIALECTS) * counts.to_numpy()))
    class_weights /= class_weights.mean()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    for model_name in model_names:
        specification = values["models"][model_name]
        for seed in seeds:
            run_dir = PROJECT / "runs/external/identification" / model_name / str(seed)
            best_dir = run_dir / "best_model"
            report_path = run_dir / "training_report.json"
            complete = False
            if report_path.exists():
                complete = (
                    json.loads(report_path.read_text(encoding="utf-8")).get("status")
                    == "COMPLETE_VALIDATION_ONLY"
                )
            if complete and best_dir.exists() and not args.force:
                print(f"Skipping complete run {model_name} seed {seed}")
                continue
            set_seed(seed)
            tokenizer = AutoTokenizer.from_pretrained(
                specification["model_id"],
                revision=specification["revision"],
                use_fast=bool(specification["use_fast_tokenizer"]),
                local_files_only=bool(config.get("local_files_only", False)),
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                specification["model_id"],
                revision=specification["revision"],
                num_labels=len(DIALECTS),
                id2label={index: label for index, label in enumerate(DIALECTS)},
                label2id=DIALECT_TO_ID,
                ignore_mismatched_sizes=True,
                local_files_only=bool(config.get("local_files_only", False)),
            ).to(device)
            train_rows = encode(train_frame, tokenizer, int(values["max_length"]))
            validation_rows = encode(validation_frame, tokenizer, int(values["max_length"]))
            train_loader = build_loader(
                train_rows, tokenizer, int(values["batch_size"]), True, seed
            )
            validation_loader = build_loader(
                validation_rows,
                tokenizer,
                int(values["evaluation_batch_size"]),
                False,
                seed,
            )
            accumulation = int(values["gradient_accumulation_steps"])
            total_steps = math.ceil(len(train_loader) / accumulation) * int(values["epochs"])
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(values["learning_rate"]),
                weight_decay=float(values["weight_decay"]),
                betas=(0.9, 0.98),
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(1, round(total_steps * float(values["warmup_fraction"]))),
                num_training_steps=total_steps,
            )
            weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            run_dir.mkdir(parents=True, exist_ok=True)
            best = -math.inf
            patience = 0
            global_step = 0
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with (run_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_handle:
                for epoch in range(1, int(values["epochs"]) + 1):
                    model.train()
                    accumulated_weight = torch.zeros((), device=device)
                    for batch_index, (batch, _) in enumerate(train_loader):
                        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                        labels = batch.pop("labels")
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(**batch).logits
                            loss = F.cross_entropy(logits.float(), labels, weight=weights)
                            per_example = F.cross_entropy(
                                logits.float(), labels, weight=weights, reduction="none"
                            )
                            loss_numerator = per_example.sum()
                        loss_numerator.backward()
                        accumulated_weight.add_(weights[labels].sum())
                        boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
                        if not boundary:
                            continue
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(accumulated_weight.clamp_min(1e-9))
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(values["gradient_clip"])
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_weight.zero_()
                        global_step += 1
                        if global_step % 25 == 0:
                            record = {
                                "epoch": epoch,
                                "step": global_step,
                                "loss": float(loss.detach()),
                                "gradient_norm": float(gradient_norm),
                                "learning_rate": scheduler.get_last_lr()[0],
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                            log_handle.write(json.dumps(record) + "\n")
                            log_handle.flush()
                            print(f"{model_name} seed={seed} {record}", flush=True)
                    predictions = predict(model, validation_loader, device)
                    probabilities = np.asarray(predictions["probabilities"].tolist())
                    metrics, by_class, matrix = classification_metrics(
                        predictions["label_id"], predictions["prediction_id"], probabilities
                    )
                    score = float(metrics["regional_macro_f1"])
                    predictions.to_parquet(
                        run_dir / f"validation_predictions_epoch_{epoch:02d}.parquet", index=False
                    )
                    by_class.to_csv(
                        run_dir / f"validation_by_class_epoch_{epoch:02d}.csv", index=False
                    )
                    np.save(run_dir / f"validation_confusion_epoch_{epoch:02d}.npy", matrix)
                    if score > best:
                        best = score
                        patience = 0
                        model.save_pretrained(best_dir, safe_serialization=True)
                        tokenizer.save_pretrained(best_dir)
                        (run_dir / "best_selection.json").write_text(
                            json.dumps({"epoch": epoch, "regional_macro_f1": score, "metrics": metrics}, indent=2)
                            + "\n",
                            encoding="utf-8",
                        )
                    else:
                        patience += 1
                    print(f"validation {model_name} seed={seed} epoch={epoch} regional_F1={score:.4f}", flush=True)
                    if patience >= int(values["early_stopping_patience"]):
                        break
            selection = json.loads((run_dir / "best_selection.json").read_text(encoding="utf-8"))
            best_predictions = pd.read_parquet(
                run_dir / f"validation_predictions_epoch_{int(selection['epoch']):02d}.parquet"
            )
            calibration = fit_temperature(best_predictions)
            (run_dir / "temperature_calibration.json").write_text(
                json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
            )
            report = {
                "status": "COMPLETE_VALIDATION_ONLY",
                "model_name": model_name,
                "model_id": specification["model_id"],
                "revision": specification["revision"],
                "license": specification["license"],
                "seed": seed,
                "best_regional_macro_f1": best,
                "optimizer_steps": global_step,
                "elapsed_seconds": time.perf_counter() - started,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "class_weighting": "inverse_sqrt_frequency_normalized",
                "test_data_access": False,
            }
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            del model, tokenizer, optimizer, scheduler, train_loader, validation_loader
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
