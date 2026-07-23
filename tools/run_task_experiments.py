#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, Subset

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import (  # noqa: E402
    EncodedDataset,
    FixedTokenMixtureDataset,
    collate_examples,
    exclude_sources,
    load_encoded_cache,
    renormalize_proportions,
)
from boichitro.experiments import task_model_from_checkpoint  # noqa: E402
from boichitro.inference import greedy_normalize, predict_identification  # noqa: E402
from boichitro.metrics import classification_metrics, normalization_metrics  # noqa: E402
from boichitro.modeling import BoichitroConfig, BoichitroForMultiTask  # noqa: E402
from boichitro.tokenization import DIALECTS, load_tokenizer, sha256_file  # noqa: E402
from boichitro.training import StageTrainerConfig, set_seed, train_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe task adaptation, normalization, and ID specialization."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/task_experiments.yaml"
    )
    parser.add_argument("--variants", nargs="+", choices=("M0", "M1", "M2", "M3"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--stages", nargs="+", choices=("a", "s", "id"), default=["a", "s", "id"]
    )
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        choices=(
            "no_lexical_prior",
            "no_dialect_head",
            "no_source_adversary",
            "no_task_conditioning",
            "no_groupdro",
            "no_mtp",
            "adamw_only",
            "bidirectional",
            "randomized_lexical_prior",
            "no_synthetic",
            "no_general_replay",
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--protocol-id")
    parser.add_argument("--stage-a-token-budget", type=int)
    parser.add_argument("--stage-s-token-budget", type=int)
    parser.add_argument("--adamw-learning-rate", type=float)
    parser.add_argument("--normalization-validation-limit", type=int)
    parser.add_argument("--replay-validation-examples", type=int)
    parser.add_argument("--stage-s-validation-checkpoints", type=int)
    return parser.parse_args()


def model_from_task_checkpoint(path: Path) -> BoichitroForMultiTask:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = BoichitroForMultiTask(BoichitroConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    return model


def stratified_frame_subset(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    selected: list[int] = []
    unused: list[int] = []
    dialects = sorted(frame["dialect"].astype(str).unique())
    base, remainder = divmod(min(count, len(frame)), max(1, len(dialects)))
    for dialect_index, dialect in enumerate(dialects):
        group = frame.loc[frame["dialect"].astype(str).eq(dialect)]
        ranked = sorted(
            group.index,
            key=lambda index: hashlib.sha256(
                str(frame.loc[index, "row_id"]).encode("utf-8")
            ).digest(),
        )
        quota = min(len(ranked), base + int(dialect_index < remainder))
        selected.extend(ranked[:quota])
        unused.extend(ranked[quota:])
    if len(selected) < min(count, len(frame)):
        unused.sort(
            key=lambda index: hashlib.sha256(
                str(frame.loc[index, "row_id"]).encode("utf-8")
            ).digest()
        )
        selected.extend(unused[: min(count, len(frame)) - len(selected)])
    return frame.loc[selected].copy().reset_index(drop=True)


def stratified_dataset_subset(dataset: EncodedDataset, count: int) -> EncodedDataset:
    frame = pd.DataFrame(
        {
            "index": range(len(dataset)),
            "row_id": [str(row["row_id"]) for row in dataset.examples],
            "dialect": [str(row["dialect"]) for row in dataset.examples],
        }
    )
    selected = stratified_frame_subset(frame, count)
    return EncodedDataset(
        [dataset.examples[int(index)] for index in selected["index"].tolist()]
    )


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


def task_run_complete(run_root: Path, requested_stages: list[str]) -> bool:
    """Recognize a finalized run even after a reconstructable Stage-A checkpoint is retired."""

    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("status") != "COMPLETE_VALIDATION_ONLY":
        return False
    completed = set(str(value) for value in manifest.get("stages_requested", []))
    if not set(requested_stages).issubset(completed):
        return False
    if "s" in requested_stages and not stage_complete(
        run_root / "stage_s", "best_checkpoint.pt", "best_selection.json"
    ):
        return False
    if "id" in requested_stages and not stage_complete(
        run_root / "stage_id",
        "best_checkpoint.pt",
        "best_selection.json",
        "temperature_calibration.json",
    ):
        return False
    if requested_stages == ["a"] and not stage_complete(
        run_root / "stage_a", "last_checkpoint.pt"
    ):
        return False
    return True


def stage_a_retention_pin(config: dict[str, Any], *, variant: str) -> str | None:
    retention = config.get("artifact_retention", {})
    for pin in retention.get("stage_a_pins", []):
        if (
            str(pin.get("protocol_id")) == str(config["protocol_id"])
            and str(pin.get("variant")) == variant
        ):
            return str(pin.get("reason", "registered downstream dependency"))
    return None


@torch.inference_mode()
def replay_nll(
    model: BoichitroForMultiTask,
    dataset: EncodedDataset,
    *,
    device: torch.device,
    pad_token_id: int,
    examples: int,
) -> float:
    indices = list(range(min(examples, len(dataset))))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        collate_fn=lambda rows: collate_examples(rows, pad_token_id),
    )
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention = batch["attention_mask"].to(device, non_blocking=True)
        task_ids = batch["task_ids"].to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(
                input_ids, labels=labels, attention_mask=attention, task_ids=task_ids
            )
        shifted = labels[:, 1:]
        loss = F.cross_entropy(
            output["logits"][:, :-1].float().reshape(-1, output["logits"].size(-1)),
            shifted.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_nll += float(loss)
        total_tokens += int(shifted.ne(-100).sum())
    return total_nll / max(1, total_tokens)


def normalization_validator(
    tokenizer,
    frame: pd.DataFrame,
    *,
    run_dir: Path,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    baseline_replay_nll: float,
    replay_dataset: EncodedDataset,
    replay_examples: int,
    replay_limit: float,
    pad_token_id: int,
):
    def validate(model: BoichitroForMultiTask, epoch: int) -> dict[str, float]:
        predictions = greedy_normalize(
            model,
            tokenizer,
            frame["source_text_model"].astype(str).tolist(),
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
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
        current_replay = replay_nll(
            model,
            replay_dataset,
            device=device,
            pad_token_id=pad_token_id,
            examples=replay_examples,
        )
        degradation = current_replay / max(1e-9, baseline_replay_nll) - 1.0
        guard_pass = degradation <= replay_limit
        prediction_frame.to_parquet(
            run_dir / f"validation_predictions_epoch_{epoch:02d}.parquet", index=False
        )
        by_dialect.to_csv(run_dir / f"validation_by_dialect_epoch_{epoch:02d}.csv", index=False)
        (run_dir / f"validation_metrics_epoch_{epoch:02d}.json").write_text(
            json.dumps(
                {
                    **metrics,
                    "replay_nll": current_replay,
                    "baseline_replay_nll": baseline_replay_nll,
                    "replay_relative_degradation": degradation,
                    "replay_guard_pass": guard_pass,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "objective": float(metrics["macro_chrfpp"]) if guard_pass else -1e9,
            "macro_chrfpp": float(metrics["macro_chrfpp"]),
            "worst_dialect_chrfpp": float(metrics["worst_dialect_chrfpp"]),
            "replay_nll": current_replay,
            "replay_relative_degradation": degradation,
            "replay_guard_pass": float(guard_pass),
        }

    return validate


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


def fit_temperature(predictions: pd.DataFrame) -> dict[str, float]:
    probabilities = np.asarray(predictions["probabilities"].tolist(), dtype=np.float64)
    labels = predictions["label_id"].to_numpy(dtype=np.int64)
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))

    def nll(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        logits = log_probabilities / temperature
        logits -= logits.max(axis=1, keepdims=True)
        normalized = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(-normalized[np.arange(len(labels)), labels].mean())

    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return {
        "temperature": float(math.exp(result.x)),
        "validation_nll": float(result.fun),
        "optimizer_success": bool(result.success),
    }


def trainer_config(values: dict[str, Any], *, seed: int, groupdro: bool, adamw_only: bool):
    return StageTrainerConfig(
        seed=seed,
        use_groupdro=groupdro,
        use_muon=not adamw_only,
        **values,
    )


def validate_stage_s_schedule_contract(config: dict[str, Any]) -> dict[str, Any] | None:
    """Verify that the main Stage-S values exactly match the validation selection."""

    relative = config.get("stage_s_schedule_selection")
    if relative is None:
        return None
    path = PROJECT / str(relative)
    if not path.exists():
        raise FileNotFoundError(f"Stage-S selection report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE_VALIDATION_ONLY":
        raise RuntimeError(f"Stage-S selection is incomplete: {path}")
    if report.get("test_data_access") is not False:
        raise RuntimeError("Stage-S schedule selection was not validation-only")
    selected = report.get("selected_candidate")
    validation = report.get("selected_validation")
    if not isinstance(selected, dict) or not isinstance(validation, dict):
        raise RuntimeError(f"Stage-S selection lacks a selected candidate: {path}")
    if not bool(validation.get("replay_guard_pass", False)):
        raise RuntimeError("Selected Stage-S schedule does not pass the replay guard")
    configured_mixture = {
        str(name): float(value)
        for name, value in config["stage_s"]["mixture"].items()
    }
    selected_mixture = {
        str(name): float(value)
        for name, value in selected["mixture"].items()
    }
    if configured_mixture != selected_mixture:
        raise RuntimeError(
            "Main Stage-S mixture does not match the validation-selected candidate"
        )
    configured_optimizer = {
        name: float(config["stage_s"]["trainer"][name])
        for name in ("muon_lr", "adamw_lr", "router_lr")
    }
    selected_optimizer = {
        name: float(selected["optimizer"][name])
        for name in ("muon_lr", "adamw_lr", "router_lr")
    }
    if configured_optimizer != selected_optimizer:
        raise RuntimeError(
            "Main Stage-S optimizer rates do not match the validation-selected candidate"
        )
    pilot_config_path = PROJECT / str(report["config_path"])
    if sha256_file(pilot_config_path) != str(report["config_sha256"]):
        raise RuntimeError("Stage-S pilot config hash no longer matches its selection report")
    pilot_config = yaml.safe_load(pilot_config_path.read_text(encoding="utf-8"))
    configured_checkpoints = int(
        config["stage_s"]["trainer"]["validation_checkpoints"]
    )
    if configured_checkpoints != int(pilot_config["validation_checkpoints"]):
        raise RuntimeError(
            "Main Stage-S validation frequency does not match the selected pilot"
        )
    return {
        "selection_report": str(path.relative_to(PROJECT)),
        "selection_report_sha256": sha256_file(path),
        "pilot_config_sha256": str(report["config_sha256"]),
        "selected_candidate_id": str(report["selected_candidate_id"]),
        "selected_validation": validation,
        "test_data_access": False,
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.protocol_id:
        config["protocol_id"] = args.protocol_id
    # Validate the registered main schedule before development/ablation CLI
    # overrides alter token budgets, validation subset sizes, or optimizer rates.
    stage_s_schedule_contract = validate_stage_s_schedule_contract(config)
    if args.stage_a_token_budget:
        config["stage_a"]["token_budget"] = args.stage_a_token_budget
    if args.stage_s_token_budget:
        config["stage_s"]["token_budget"] = args.stage_s_token_budget
    if args.normalization_validation_limit:
        config["evaluation"]["normalization_validation_limit"] = int(
            args.normalization_validation_limit
        )
    if args.replay_validation_examples:
        config["evaluation"]["replay_validation_examples"] = int(
            args.replay_validation_examples
        )
    if args.stage_s_validation_checkpoints:
        config["stage_s"]["trainer"]["validation_checkpoints"] = int(
            args.stage_s_validation_checkpoints
        )
    variants = args.variants or list(config["variants"])
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    ablations = set(args.ablation)
    adamw_learning_rate = args.adamw_learning_rate
    optimizer_lr_source = "task_config"
    if "adamw_only" in ablations and adamw_learning_rate is None:
        selection_path = PROJECT / "reports/model/optimizer_pilot_selection.json"
        if not selection_path.exists():
            raise FileNotFoundError(
                f"AdamW-only confirmatory run requires optimizer pilot: {selection_path}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "COMPLETE_VALIDATION_ONLY":
            raise RuntimeError(f"Optimizer pilot is not complete: {selection_path}")
        adamw_learning_rate = float(selection["selected_adamw_learning_rate"])
        optimizer_lr_source = str(selection_path.relative_to(PROJECT))
    elif adamw_learning_rate is not None:
        optimizer_lr_source = "explicit_cli_development_pilot"
    if adamw_learning_rate is not None:
        if adamw_learning_rate <= 0:
            raise ValueError("--adamw-learning-rate must be positive")
        for stage in ("stage_a", "stage_s", "stage_id"):
            config[stage]["trainer"]["adamw_lr"] = float(adamw_learning_rate)
    tokenizer = load_tokenizer(PROJECT / "artifacts/tokenizers/frozen")
    pad_token_id = int(tokenizer.token_to_id("<pad>"))
    maps = json.loads((PROJECT / "cache/tasks/maps.json").read_text(encoding="utf-8"))
    group_count = len(maps["group_to_id"])
    n_sources = len(maps["source_to_id"])
    cache_root = PROJECT / "cache/tasks"
    components = {
        "normalization": load_encoded_cache(cache_root / "normalization_train.pt")[0],
        "identification": load_encoded_cache(cache_root / "identification_train.pt")[0],
        "dialect_clm": load_encoded_cache(cache_root / "dialect_clm_train.pt")[0],
        "romanized": load_encoded_cache(cache_root / "normalization_romanized_train.pt")[0],
        "general_replay": load_encoded_cache(cache_root / "general_replay_train.pt")[0],
    }
    original_component_rows = {name: len(dataset) for name, dataset in components.items()}
    if "no_synthetic" in ablations:
        components["normalization"] = exclude_sources(
            components["normalization"], ["synthetic_robustness_v1"]
        )
    excluded_mixture_components = (
        ["general_replay"] if "no_general_replay" in ablations else []
    )
    stage_a_mixture = renormalize_proportions(
        config["stage_a"]["mixture"], excluded_mixture_components
    )
    stage_s_mixture = renormalize_proportions(
        config["stage_s"]["mixture"], excluded_mixture_components
    )
    id_validation = load_encoded_cache(cache_root / "identification_validation.pt")[0]
    norm_validation = pd.read_parquet(
        PROJECT / "data/final/v1/normalization_validation.parquet"
    )
    evaluation = config["evaluation"]
    if evaluation.get("identification_validation_limit"):
        id_validation = stratified_dataset_subset(
            id_validation, int(evaluation["identification_validation_limit"])
        )
    if evaluation.get("normalization_validation_limit"):
        norm_validation = stratified_frame_subset(
            norm_validation, int(evaluation["normalization_validation_limit"])
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    protocol_suffix = "base" if not ablations else "__".join(sorted(ablations))

    for variant in variants:
        variant_values = config["variants"][variant]
        base_checkpoint = PROJECT / variant_values["base_checkpoint"]
        if not base_checkpoint.exists():
            raise FileNotFoundError(f"Missing base checkpoint for {variant}: {base_checkpoint}")
        for seed in seeds:
            set_seed(seed)
            run_root = (
                PROJECT
                / "runs/task"
                / str(config["protocol_id"])
                / f"{variant}__{protocol_suffix}"
                / str(seed)
            )
            if not args.force and task_run_complete(run_root, args.stages):
                print(
                    f"Skipping complete task run {variant} {protocol_suffix} seed={seed}",
                    flush=True,
                )
                continue
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "data_ablation_report.json").write_text(
                json.dumps(
                    {
                        "ablations": sorted(ablations),
                        "component_rows_before": original_component_rows,
                        "component_rows_after": {
                            name: len(dataset) for name, dataset in components.items()
                        },
                        "stage_a_mixture": stage_a_mixture,
                        "stage_s_mixture": stage_s_mixture,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            initialization_path = run_root / "initialization_report.json"
            stage_a_dir = run_root / "stage_a"
            stage_a_checkpoint = stage_a_dir / "last_checkpoint.pt"
            groupdro = bool(variant_values["groupdro"]) and "no_groupdro" not in ablations

            if "a" in args.stages and (
                args.force or not stage_complete(stage_a_dir, "last_checkpoint.pt")
            ):
                model, initialization = task_model_from_checkpoint(
                    base_checkpoint,
                    variant=variant,
                    n_sources=n_sources,
                    ablations=ablations,
                )
                initialization_path.write_text(
                    json.dumps(initialization, indent=2) + "\n", encoding="utf-8"
                )
                mixture = FixedTokenMixtureDataset(
                    components,
                    stage_a_mixture,
                    token_budget=int(config["stage_a"]["token_budget"]),
                    seed=seed,
                )
                stage_a_dir.mkdir(parents=True, exist_ok=True)
                (stage_a_dir / "mixture_report.json").write_text(
                    json.dumps(mixture.report(), indent=2) + "\n", encoding="utf-8"
                )
                train_stage(
                    model,
                    mixture,
                    pad_token_id=pad_token_id,
                    group_count=group_count,
                    config=trainer_config(
                        config["stage_a"]["trainer"],
                        seed=seed,
                        groupdro=groupdro,
                        adamw_only="adamw_only" in ablations,
                    ),
                    run_dir=stage_a_dir,
                    device=device,
                )
                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None
            if not stage_complete(stage_a_dir, "last_checkpoint.pt"):
                raise RuntimeError(f"Stage A is not complete: {stage_a_dir}")

            if "s" in args.stages:
                stage_s_dir = run_root / "stage_s"
                if args.force or not stage_complete(
                    stage_s_dir, "best_checkpoint.pt", "best_selection.json"
                ):
                    model = model_from_task_checkpoint(stage_a_checkpoint).to(device)
                    baseline_replay = replay_nll(
                        model,
                        components["general_replay"],
                        device=device,
                        pad_token_id=pad_token_id,
                        examples=int(evaluation["replay_validation_examples"]),
                    )
                    mixture = FixedTokenMixtureDataset(
                        components,
                        stage_s_mixture,
                        token_budget=int(config["stage_s"]["token_budget"]),
                        seed=seed + 100_003,
                    )
                    stage_s_dir.mkdir(parents=True, exist_ok=True)
                    (stage_s_dir / "mixture_report.json").write_text(
                        json.dumps(mixture.report(), indent=2) + "\n", encoding="utf-8"
                    )
                    validate = normalization_validator(
                        tokenizer,
                        norm_validation,
                        run_dir=stage_s_dir,
                        device=device,
                        batch_size=int(evaluation["normalization_batch_size"]),
                        max_new_tokens=int(evaluation["normalization_max_new_tokens"]),
                        baseline_replay_nll=baseline_replay,
                        replay_dataset=components["general_replay"],
                        replay_examples=int(evaluation["replay_validation_examples"]),
                        replay_limit=float(evaluation["replay_degradation_limit"]),
                        pad_token_id=pad_token_id,
                    )
                    train_stage(
                        model,
                        mixture,
                        pad_token_id=pad_token_id,
                        group_count=group_count,
                        config=trainer_config(
                            config["stage_s"]["trainer"],
                            seed=seed,
                            groupdro=groupdro,
                            adamw_only="adamw_only" in ablations,
                        ),
                        run_dir=stage_s_dir,
                        validation_fn=validate,
                        device=device,
                    )
                    del model
                    torch.cuda.empty_cache() if device.type == "cuda" else None

            if "id" in args.stages:
                stage_id_dir = run_root / "stage_id"
                if args.force or not stage_complete(
                    stage_id_dir,
                    "best_checkpoint.pt",
                    "best_selection.json",
                    "temperature_calibration.json",
                ):
                    model = model_from_task_checkpoint(stage_a_checkpoint)
                    validate_id = identification_validator(
                        id_validation,
                        run_dir=stage_id_dir,
                        device=device,
                        pad_token_id=pad_token_id,
                        batch_size=int(evaluation["identification_batch_size"]),
                    )
                    train_stage(
                        model,
                        components["identification"],
                        pad_token_id=pad_token_id,
                        group_count=group_count,
                        config=trainer_config(
                            config["stage_id"]["trainer"],
                            seed=seed,
                            groupdro=groupdro,
                            adamw_only="adamw_only" in ablations,
                        ),
                        run_dir=stage_id_dir,
                        validation_fn=validate_id,
                        device=device,
                    )
                    best = torch.load(
                        stage_id_dir / "best_checkpoint.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    epoch = int(best["epoch"])
                    predictions = pd.read_parquet(
                        stage_id_dir / f"validation_predictions_epoch_{epoch:02d}.parquet"
                    )
                    calibration = fit_temperature(predictions)
                    (stage_id_dir / "temperature_calibration.json").write_text(
                        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
                    )
                    del model
                    torch.cuda.empty_cache() if device.type == "cuda" else None

            run_manifest = {
                "status": "COMPLETE_VALIDATION_ONLY",
                "protocol_id": config["protocol_id"],
                "variant": variant,
                "seed": seed,
                "ablations": sorted(ablations),
                "base_checkpoint": str(base_checkpoint.relative_to(PROJECT)),
                "stages_requested": args.stages,
                "stage_token_budgets": {
                    "a": int(config["stage_a"]["token_budget"]),
                    "s": int(config["stage_s"]["token_budget"]),
                },
                "adamw_learning_rate_override": adamw_learning_rate,
                "optimizer_lr_source": optimizer_lr_source,
                "stage_s_schedule_contract": stage_s_schedule_contract,
                "test_data_access": False,
                "trainer_configs": {
                    stage: dataclasses.asdict(
                        trainer_config(
                            config[f"stage_{stage}"]["trainer"],
                            seed=seed,
                            groupdro=groupdro,
                            adamw_only="adamw_only" in ablations,
                        )
                    )
                    for stage in ("a", "s", "id")
                },
                "dialect_labels": list(DIALECTS),
                "tokenizer_sha256": sha256_file(
                    PROJECT / "artifacts/tokenizers/frozen/tokenizer.json"
                ),
            }
            stage_a_retention: dict[str, Any] = {
                "policy": "retain",
                "reason": "Stage A is required by an incomplete or unrequested child branch",
            }
            retention_values = config.get("artifact_retention", {})
            stage_a_path = run_root / "stage_a/last_checkpoint.pt"
            if stage_a_path.exists():
                stage_a_retention.update(
                    checkpoint_sha256=sha256_file(stage_a_path),
                    checkpoint_bytes=stage_a_path.stat().st_size,
                )
            pin_reason = stage_a_retention_pin(config, variant=variant)
            stage_s_guard_pass = False
            if "s" in args.stages and stage_complete(
                run_root / "stage_s", "best_checkpoint.pt", "best_selection.json"
            ):
                stage_s_selection = json.loads(
                    (run_root / "stage_s/best_selection.json").read_text(
                        encoding="utf-8"
                    )
                )
                stage_s_guard_pass = bool(
                    stage_s_selection.get("validation", {}).get(
                        "replay_guard_pass", False
                    )
                )
            requested_children_complete = (
                "s" in args.stages
                and stage_s_guard_pass
                and (
                    "id" not in args.stages
                    or stage_complete(
                        run_root / "stage_id",
                        "best_checkpoint.pt",
                        "best_selection.json",
                        "temperature_calibration.json",
                    )
                )
            )
            discard_stage_a = bool(
                retention_values.get(
                    "discard_stage_a_after_completed_branches", False
                )
                and requested_children_complete
                and pin_reason is None
                and stage_a_path.exists()
            )
            if pin_reason is not None:
                stage_a_retention.update(policy="retain", reason=pin_reason)
            elif not stage_s_guard_pass and "s" in args.stages:
                stage_a_retention.update(
                    policy="retain",
                    reason="Stage-S replay guard failed; Stage A is the registered normalization fallback",
                )
            elif discard_stage_a:
                stage_a_retention.update(
                    policy="scheduled_for_deletion",
                    reason=(
                        "Selected Stage-S and requested ID checkpoints preserve both "
                        "downstream branches; Stage A is deterministically reconstructable"
                    ),
                )
            run_manifest["artifact_retention"] = {"stage_a": stage_a_retention}
            manifest_path = run_root / "run_manifest.json"
            manifest_path.write_text(
                json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
            )
            if discard_stage_a:
                stage_a_path.unlink()
                stage_a_retention["policy"] = "discarded_after_completed_branches"
                stage_a_retention["checkpoint_present"] = False
                manifest_path.write_text(
                    json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
                )
            else:
                stage_a_retention["checkpoint_present"] = stage_a_path.exists()
            (run_root / "stage_a/checkpoint_retention.json").write_text(
                json.dumps(stage_a_retention, indent=2) + "\n", encoding="utf-8"
            )
            # Persist the final presence flag in the canonical run manifest.
            manifest_path.write_text(
                json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
