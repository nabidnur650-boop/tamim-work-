#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import minimize_scalar

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import (  # noqa: E402
    EncodedDataset,
    FixedTokenMixtureDataset,
    MaskedNextTokenDataset,
    load_encoded_cache,
)
from boichitro.experiments import compatible_state_load  # noqa: E402
from boichitro.inference import predict_identification  # noqa: E402
from boichitro.metrics import classification_metrics  # noqa: E402
from boichitro.modeling import BoichitroConfig, BoichitroForMultiTask  # noqa: E402
from boichitro.tokenization import load_tokenizer, sha256_file  # noqa: E402
from boichitro.training import StageTrainerConfig, set_seed, train_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run registered MNTP + contrastive bidirectional ID specialization."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/bidirectional_identification.yaml",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def model_from_checkpoint(path: Path, **overrides) -> tuple[BoichitroForMultiTask, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    values = dict(payload["model_config"])
    values.update(overrides)
    model = BoichitroForMultiTask(BoichitroConfig.from_dict(values))
    report = compatible_state_load(model, payload["model_state_dict"])
    report.update(
        source_checkpoint=str(path.relative_to(PROJECT)),
        source_tokens_seen=int(payload.get("tokens_seen", 0)),
        overrides=overrides,
    )
    return model, report


def stage_complete(run_dir: Path, *required: str) -> bool:
    """Return true only for an explicitly completed stage with all selected outputs."""

    report_path = run_dir / "training_report.json"
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return report.get("status") == "COMPLETE" and all(
        (run_dir / relative).exists() for relative in required
    )


def identification_validator(
    dataset: EncodedDataset,
    *,
    run_dir: Path,
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
):
    def validate(model: BoichitroForMultiTask, epoch: int) -> dict[str, float]:
        predictions, _ = predict_identification(
            model,
            dataset,
            device=device,
            pad_token_id=pad_token_id,
            batch_size=batch_size,
        )
        probabilities = np.asarray(predictions["probabilities"].tolist(), dtype=np.float64)
        metrics, by_class, matrix = classification_metrics(
            predictions["label_id"].to_numpy(),
            predictions["prediction_id"].to_numpy(),
            probabilities,
        )
        predictions.to_parquet(
            run_dir / f"validation_predictions_epoch_{epoch:02d}.parquet", index=False
        )
        by_class.to_csv(run_dir / f"validation_by_class_epoch_{epoch:02d}.csv", index=False)
        np.save(run_dir / f"validation_confusion_epoch_{epoch:02d}.npy", matrix)
        (run_dir / f"validation_metrics_epoch_{epoch:02d}.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        return {
            "objective": float(metrics["regional_macro_f1"]),
            **{key: float(value) for key, value in metrics.items()},
        }

    return validate


def fit_temperature(predictions: pd.DataFrame) -> dict[str, float | bool]:
    probabilities = np.asarray(predictions["probabilities"].tolist(), dtype=np.float64)
    labels = predictions["label_id"].to_numpy(dtype=np.int64)
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))

    def objective(log_temperature: float) -> float:
        logits = log_probabilities / math.exp(log_temperature)
        logits -= logits.max(axis=1, keepdims=True)
        normalized = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(-normalized[np.arange(len(labels)), labels].mean())

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return {
        "temperature": float(math.exp(result.x)),
        "validation_nll": float(result.fun),
        "optimizer_success": bool(result.success),
    }


def trainer_config(values: dict, seed: int) -> StageTrainerConfig:
    return StageTrainerConfig(seed=seed, use_groupdro=True, use_muon=True, **values)


def contrastive_coverage(dataset: EncodedDataset, *, batch_size: int, seed: int) -> dict:
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    eligible = 0
    by_dialect: dict[str, dict[str, object]] = {}
    for example in dataset.examples:
        dialect = str(example["dialect"])
        summary = by_dialect.setdefault(dialect, {"rows": 0, "source_ids": set()})
        summary["rows"] = int(summary["rows"]) + 1
        summary["source_ids"].add(str(example["source_id"]))
    for start in range(0, len(order), batch_size):
        batch = [dataset.examples[index] for index in order[start : start + batch_size]]
        source_sets: dict[int, set[int]] = {}
        for example in batch:
            source_sets.setdefault(int(example["dialect_label"]), set()).add(
                int(example["source_label"])
            )
        eligible += sum(
            bool(
                source_sets[int(example["dialect_label"])]
                - {int(example["source_label"])}
            )
            for example in batch
        )
    return {
        "policy": "same_dialect_different_source_positives",
        "seed": seed,
        "batch_size": batch_size,
        "rows": len(dataset),
        "prospective_rows_with_positive": eligible,
        "prospective_positive_coverage": eligible / max(1, len(dataset)),
        "prospective_order": "torch_randperm_with_registered_seed",
        "by_dialect": {
            dialect: {
                "rows": values["rows"],
                "source_count": len(values["source_ids"]),
                "source_ids": sorted(values["source_ids"]),
            }
            for dialect, values in sorted(by_dialect.items())
        },
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    tokenizer = load_tokenizer(PROJECT / "artifacts/tokenizers/frozen")
    pad_token_id = int(tokenizer.token_to_id("<pad>"))
    mask_token_id = int(tokenizer.token_to_id("<mask>"))
    special_ids = [
        int(value)
        for token in tokenizer.get_vocab()
        if token.startswith("<") and token.endswith(">")
        if (value := tokenizer.token_to_id(token)) is not None
    ]
    maps = json.loads((PROJECT / "cache/tasks/maps.json").read_text(encoding="utf-8"))
    group_count = len(maps["group_to_id"])
    cache_root = PROJECT / "cache/tasks"
    components = {
        "identification": load_encoded_cache(cache_root / "identification_train.pt")[0],
        "dialect_clm": load_encoded_cache(cache_root / "dialect_clm_train.pt")[0],
    }
    validation = load_encoded_cache(cache_root / "identification_validation.pt")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    for seed in seeds:
        set_seed(seed)
        source_checkpoint = (
            PROJECT
            / "runs/task"
            / str(config["source_protocol_id"])
            / str(config["source_variant"])
            / str(seed)
            / "stage_a/last_checkpoint.pt"
        )
        if not source_checkpoint.exists():
            raise FileNotFoundError(f"Frozen causal Stage A checkpoint missing: {source_checkpoint}")
        run_root = (
            PROJECT
            / "runs/task"
            / str(config["protocol_id"])
            / str(config["variant_id"])
            / str(seed)
        )
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "contrastive_coverage_report.json").write_text(
            json.dumps(
                contrastive_coverage(
                    components["identification"],
                    batch_size=int(
                        config["identification"]["trainer"]["micro_batch_size"]
                    ),
                    seed=seed,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        mntp_dir = run_root / "stage_mntp"
        mntp_checkpoint = mntp_dir / "last_checkpoint.pt"
        if args.force or not stage_complete(mntp_dir, "last_checkpoint.pt"):
            model, initialization = model_from_checkpoint(
                source_checkpoint,
                bidirectional_attention=True,
                use_mtp=False,
                contrastive_loss_weight=0.0,
            )
            (run_root / "initialization_report.json").write_text(
                json.dumps(initialization, indent=2) + "\n", encoding="utf-8"
            )
            mixture = FixedTokenMixtureDataset(
                components,
                config["mntp"]["mixture"],
                token_budget=int(config["mntp"]["token_budget"]),
                seed=seed + 700_001,
            )
            masked = MaskedNextTokenDataset(
                mixture,
                mask_token_id=mask_token_id,
                vocab_size=tokenizer.get_vocab_size(),
                special_token_ids=special_ids,
                mask_probability=float(config["mntp"]["mask_probability"]),
                seed=seed + 700_003,
            )
            mntp_dir.mkdir(parents=True, exist_ok=True)
            (mntp_dir / "mixture_report.json").write_text(
                json.dumps(
                    {
                        **mixture.report(),
                        "objective": "masked_next_token_prediction",
                        "mask_probability": config["mntp"]["mask_probability"],
                        "masking_policy": config["mntp"]["masking_policy"],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            train_stage(
                model,
                masked,
                pad_token_id=pad_token_id,
                group_count=group_count,
                config=trainer_config(config["mntp"]["trainer"], seed),
                run_dir=mntp_dir,
                device=device,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not stage_complete(mntp_dir, "last_checkpoint.pt"):
            raise RuntimeError(f"MNTP stage is not complete: {mntp_dir}")

        id_dir = run_root / "stage_id"
        if args.force or not stage_complete(
            id_dir,
            "best_checkpoint.pt",
            "best_selection.json",
            "temperature_calibration.json",
        ):
            model, _ = model_from_checkpoint(
                mntp_checkpoint,
                contrastive_loss_weight=float(
                    config["identification"]["contrastive_loss_weight"]
                ),
                contrastive_temperature=float(
                    config["identification"]["contrastive_temperature"]
                ),
            )
            validate = identification_validator(
                validation,
                run_dir=id_dir,
                device=device,
                pad_token_id=pad_token_id,
                batch_size=int(config["evaluation"]["batch_size"]),
            )
            train_stage(
                model,
                components["identification"],
                pad_token_id=pad_token_id,
                group_count=group_count,
                config=trainer_config(config["identification"]["trainer"], seed),
                run_dir=id_dir,
                validation_fn=validate,
                device=device,
            )
            best = torch.load(id_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            epoch = int(best["epoch"])
            predictions = pd.read_parquet(
                id_dir / f"validation_predictions_epoch_{epoch:02d}.parquet"
            )
            (id_dir / "temperature_calibration.json").write_text(
                json.dumps(fit_temperature(predictions), indent=2) + "\n",
                encoding="utf-8",
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        manifest = {
            "status": "COMPLETE_VALIDATION_ONLY",
            "protocol_id": config["protocol_id"],
            "variant": config["variant_id"],
            "seed": seed,
            "source_checkpoint": str(source_checkpoint.relative_to(PROJECT)),
            "mntp_checkpoint": str(mntp_checkpoint.relative_to(PROJECT)),
            "bidirectional_attention": True,
            "mntp_token_budget": config["mntp"]["token_budget"],
            "contrastive_loss_weight": config["identification"]["contrastive_loss_weight"],
            "trainer_configs": {
                "mntp": dataclasses.asdict(trainer_config(config["mntp"]["trainer"], seed)),
                "identification": dataclasses.asdict(
                    trainer_config(config["identification"]["trainer"], seed)
                ),
            },
            "tokenizer_sha256": sha256_file(
                PROJECT / "artifacts/tokenizers/frozen/tokenizer.json"
            ),
            "test_data_access": False,
        }
        (run_root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Bidirectional ID specialization complete: seed={seed}", flush=True)


if __name__ == "__main__":
    main()
