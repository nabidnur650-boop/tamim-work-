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
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
)

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.metrics import normalization_metrics  # noqa: E402
from boichitro.tokenization import nfc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune registered external seq2seq baselines.")
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


def tokenize_training(frame, tokenizer, prefix: str, source_length: int, target_length: int):
    sources = [prefix + nfc(value) for value in frame["source_text_model"].astype(str)]
    targets = [nfc(value) for value in frame["target_text_model"].astype(str)]
    inputs = tokenizer(sources, truncation=True, max_length=source_length)
    labels = tokenizer(text_target=targets, truncation=True, max_length=target_length)
    return [
        {
            "input_ids": inputs["input_ids"][index],
            "attention_mask": inputs["attention_mask"][index],
            "labels": labels["input_ids"][index],
            "example_weight": float(frame.iloc[index].get("example_loss_weight", 1.0)),
        }
        for index in range(len(frame))
    ]


@torch.inference_mode()
def validate(model, tokenizer, frame, values, prefix: str, device: torch.device):
    model.eval()
    predictions = []
    batch_size = int(values["generation_batch_size"])
    sources = frame["source_text_model"].astype(str).tolist()
    for start in range(0, len(sources), batch_size):
        batch = [prefix + nfc(value) for value in sources[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(values["max_source_length"]),
        ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(
                **encoded,
                max_new_tokens=int(values["generation_max_new_tokens"]),
                num_beams=1,
            )
        predictions.extend(
            nfc(value).strip()
            for value in tokenizer.batch_decode(generated, skip_special_tokens=True)
        )
    prediction_frame = pd.DataFrame(
        {
            "row_id": frame["row_id"].astype(str),
            "semantic_group_id": frame["semantic_group_id"].astype(str),
            "dialect": frame["dialect"].astype(str),
            "source_id": frame["source_id"].astype(str),
            "source": frame["source_text_model"].astype(str),
            "reference": frame["target_text_model"].astype(str),
            "prediction": predictions,
        }
    )
    metrics, by_dialect = normalization_metrics(prediction_frame)
    return metrics, by_dialect, prediction_frame


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    values = config["normalization"]
    model_names = args.models or list(values["models"])
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    train_frame = pd.read_parquet(PROJECT / "data/final/v1/normalization_train.parquet")
    validation_frame = pd.read_parquet(
        PROJECT / "data/final/v1/normalization_validation.parquet"
    )
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    for model_name in model_names:
        specification = values["models"][model_name]
        for seed in seeds:
            run_dir = PROJECT / "runs/external/normalization" / model_name / str(seed)
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
            model = AutoModelForSeq2SeqLM.from_pretrained(
                specification["model_id"],
                revision=specification["revision"],
                local_files_only=bool(config.get("local_files_only", False)),
            ).to(device)
            encoded_rows = tokenize_training(
                train_frame,
                tokenizer,
                str(specification["prefix"]),
                int(values["max_source_length"]),
                int(values["max_target_length"]),
            )
            base_collator = DataCollatorForSeq2Seq(
                tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
            )

            def collate(rows):
                weights = torch.tensor(
                    [row["example_weight"] for row in rows], dtype=torch.float32
                )
                features = [
                    {key: value for key, value in row.items() if key != "example_weight"}
                    for row in rows
                ]
                batch = base_collator(features)
                batch["example_weights"] = weights
                return batch

            loader = DataLoader(
                Rows(encoded_rows),
                batch_size=int(values["batch_size"]),
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
                num_workers=2,
                pin_memory=True,
                collate_fn=collate,
            )
            accumulation = int(values["gradient_accumulation_steps"])
            total_steps = math.ceil(len(loader) / accumulation) * int(values["epochs"])
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
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train_log.jsonl"
            best = -math.inf
            patience = 0
            global_step = 0
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with log_path.open("w", encoding="utf-8") as log_handle:
                for epoch in range(1, int(values["epochs"]) + 1):
                    model.train()
                    accumulated_weight = torch.zeros((), device=device)
                    for batch_index, batch in enumerate(loader):
                        weights = batch.pop("example_weights").to(device)
                        batch = {key: tensor.to(device, non_blocking=True) for key, tensor in batch.items()}
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output = model(**batch)
                            token_loss = F.cross_entropy(
                                output.logits.float().reshape(-1, output.logits.size(-1)),
                                batch["labels"].reshape(-1),
                                ignore_index=-100,
                                reduction="none",
                            ).view(batch["labels"].shape)
                            valid = batch["labels"].ne(-100)
                            per_example = (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
                            loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-9)
                            loss_numerator = (per_example * weights).sum()
                        loss_numerator.backward()
                        accumulated_weight.add_(weights.sum())
                        boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
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
                    metrics, by_dialect, predictions = validate(
                        model,
                        tokenizer,
                        validation_frame,
                        values,
                        str(specification["prefix"]),
                        device,
                    )
                    score = float(metrics["macro_chrfpp"])
                    predictions.to_parquet(
                        run_dir / f"validation_predictions_epoch_{epoch:02d}.parquet",
                        index=False,
                    )
                    by_dialect.to_csv(
                        run_dir / f"validation_by_dialect_epoch_{epoch:02d}.csv", index=False
                    )
                    if score > best:
                        best = score
                        patience = 0
                        model.save_pretrained(best_dir, safe_serialization=True)
                        tokenizer.save_pretrained(best_dir)
                        (run_dir / "best_selection.json").write_text(
                            json.dumps({"epoch": epoch, "macro_chrfpp": score, "metrics": metrics}, indent=2)
                            + "\n",
                            encoding="utf-8",
                        )
                    else:
                        patience += 1
                    print(f"validation {model_name} seed={seed} epoch={epoch} macro_chrF++={score:.4f}", flush=True)
                    if patience >= int(values["early_stopping_patience"]):
                        break
            report = {
                "status": "COMPLETE_VALIDATION_ONLY",
                "model_name": model_name,
                "model_id": specification["model_id"],
                "revision": specification["revision"],
                "license": specification["license"],
                "seed": seed,
                "best_macro_chrfpp": best,
                "optimizer_steps": global_step,
                "elapsed_seconds": time.perf_counter() - started,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "test_data_access": False,
            }
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            del model, tokenizer, optimizer, scheduler, loader, encoded_rows
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
