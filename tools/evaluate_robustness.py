#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import EncodedDataset, encode_identification  # noqa: E402
from boichitro.fusion import (  # noqa: E402
    blend_identification_probabilities,
    classical_identification_probabilities,
    inferred_dialect_rewrite,
    select_normalization_candidates,
)
from boichitro.inference import greedy_normalize, predict_identification  # noqa: E402
from boichitro.metrics import classification_metrics, normalization_metrics  # noqa: E402
from boichitro.modeling import BoichitroConfig, BoichitroForMultiTask  # noqa: E402
from boichitro.protocol import require_protocol_freeze  # noqa: E402
from boichitro.robustness import perturb_text, stable_perturbation_seed  # noqa: E402
from boichitro.tokenization import load_tokenizer, sha256_file  # noqa: E402
from evaluate_locked import load_fusion_resources  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate registered grapheme-level robustness curves.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/robustness_evaluation.yaml"
    )
    parser.add_argument(
        "--task-config", type=Path, default=PROJECT / "configs/task_experiments.yaml"
    )
    parser.add_argument("--variants", nargs="+", default=["M0", "M2", "M3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2903, 4307])
    parser.add_argument("--suffix", default="base")
    parser.add_argument("--max-examples", type=int)
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> BoichitroForMultiTask:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = BoichitroForMultiTask(BoichitroConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def task_checkpoints(task_root: Path) -> tuple[Path, Path]:
    candidate = task_root / "stage_s/best_checkpoint.pt"
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    guard = bool(payload.get("extra", {}).get("validation", {}).get("replay_guard_pass", 0.0))
    normalization = candidate if guard else task_root / "stage_a/last_checkpoint.pt"
    return normalization, task_root / "stage_id/best_checkpoint.pt"


def perturb(frame: pd.DataFrame, column: str, family: str, severity: float, seed: int) -> list[str]:
    return [
        perturb_text(
            str(text),
            family=family,
            severity=severity,
            seed=stable_perturbation_seed(str(row_id), family, severity, seed),
        )
        for row_id, text in zip(frame["row_id"], frame[column])
    ]


def stratified_hash_sample(frame: pd.DataFrame, per_dialect: int, seed: int) -> pd.DataFrame:
    pieces = []
    for dialect, group in frame.groupby("dialect", sort=True, observed=True):
        ranked = sorted(
            group.index,
            key=lambda index: hashlib.sha256(
                f"{seed}|{group.loc[index, 'row_id']}".encode("utf-8")
            ).digest(),
        )
        if len(ranked) < per_dialect:
            raise ValueError(
                f"Robustness stratum {dialect} has {len(ranked)} examples, need {per_dialect}"
            )
        pieces.append(group.loc[ranked[:per_dialect]])
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    task_config = yaml.safe_load(args.task_config.read_text(encoding="utf-8"))
    locked_config_path = PROJECT / config["locked_evaluation_config"]
    locked_config = yaml.safe_load(locked_config_path.read_text(encoding="utf-8"))
    freeze = require_protocol_freeze(
        PROJECT, str(locked_config["protocol_freeze_id"])
    )
    fusion = load_fusion_resources(locked_config, freeze)
    if fusion is None:
        raise RuntimeError("Registered robustness evaluation requires frozen fusion")
    tokenizer = load_tokenizer(PROJECT / "artifacts/tokenizers/frozen")
    pad_token_id = int(tokenizer.token_to_id("<pad>"))
    maps = json.loads((PROJECT / "cache/tasks/maps.json").read_text(encoding="utf-8"))
    norm_frame = pd.read_parquet(PROJECT / config["normalization_frame"])
    id_frame = pd.read_parquet(PROJECT / config["identification_frame"])
    id_frame = id_frame.loc[id_frame["split"].eq("test")].copy()
    if args.max_examples:
        norm_frame = norm_frame.head(args.max_examples).copy()
        id_frame = id_frame.head(args.max_examples).copy()
    else:
        norm_frame = stratified_hash_sample(
            norm_frame,
            int(config["examples_per_dialect"]),
            int(config["sampling_seed"]),
        )
        id_frame = stratified_hash_sample(
            id_frame,
            int(config["examples_per_dialect"]),
            int(config["sampling_seed"]) + 1,
        )
    perturbations = {}
    for family in config["families"]:
        for severity in config["severities"]:
            key = (str(family), float(severity))
            perturbations[key] = {
                "normalization": perturb(
                    norm_frame,
                    "source_text_model",
                    key[0],
                    key[1],
                    int(config["perturbation_seed"]),
                ),
                "identification": perturb(
                    id_frame,
                    "text_model",
                    key[0],
                    key[1],
                    int(config["perturbation_seed"]),
                ),
            }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            task_root = (
                PROJECT
                / "runs/task"
                / str(task_config["protocol_id"])
                / f"{variant}__{args.suffix}"
                / str(seed)
            )
            norm_checkpoint, id_checkpoint = task_checkpoints(task_root)
            base_manifest_path = (
                PROJECT
                / "predictions"
                / str(config["base_protocol_id"])
                / f"{variant}__{args.suffix}"
                / str(seed)
                / "evaluation_manifest.json"
            )
            base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
            expected_hashes = {
                "normalization": sha256_file(norm_checkpoint),
                "identification": sha256_file(id_checkpoint),
            }
            if base["checkpoint_sha256"] != expected_hashes:
                raise RuntimeError("Robustness checkpoint differs from locked base evaluation")
            if base.get("fusion", {}).get("artifact_sha256") != fusion["artifact_sha256"]:
                raise RuntimeError("Robustness fusion artifacts differ from locked evaluation")
            output_dir = (
                PROJECT
                / "predictions"
                / str(config["protocol_id"])
                / f"{variant}__{args.suffix}"
                / str(seed)
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            norm_model = load_model(norm_checkpoint, device)
            clean_norm_predictions = None
            for (family, severity), values in perturbations.items():
                if severity == 0.0 and clean_norm_predictions is not None:
                    predictions = list(clean_norm_predictions)
                else:
                    predictions = greedy_normalize(
                        norm_model,
                        tokenizer,
                        values["normalization"],
                        device=device,
                        batch_size=int(config["normalization_batch_size"]),
                        max_new_tokens=int(config["normalization_max_new_tokens"]),
                    )
                    if severity == 0.0:
                        clean_norm_predictions = list(predictions)
                prediction_frame = pd.DataFrame(
                    {
                        "row_id": norm_frame["row_id"].astype(str),
                        "semantic_group_id": norm_frame["semantic_group_id"].astype(str),
                        "dialect": norm_frame["dialect"].astype(str),
                        "source_id": norm_frame["source_id"].astype(str),
                        "source_clean": norm_frame["source_text_model"].astype(str),
                        "source_perturbed": values["normalization"],
                        "reference": norm_frame["target_text_model"].astype(str),
                        "prediction": predictions,
                        "family": family,
                        "severity": severity,
                    }
                )
                metrics, _ = normalization_metrics(prediction_frame)
                prediction_frame.to_parquet(
                    output_dir / f"normalization_{family}_{severity:.2f}.parquet", index=False
                )
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "task": "normalization",
                        "system_view": "raw_neural",
                        "family": family,
                        "severity": severity,
                        "metric": "macro_chrfpp",
                        "value": float(metrics["macro_chrfpp"]),
                    }
                )
                perturbed_sources = [str(value) for value in values["normalization"]]
                dialect_probabilities = classical_identification_probabilities(
                    perturbed_sources, fusion["svm"]
                )
                rewrites, inferred_dialects = inferred_dialect_rewrite(
                    perturbed_sources,
                    dialect_probabilities,
                    fusion["rewrite_mapping"],
                    allowed_dialects=fusion["normalization_selector"].get(
                        "inferred_dialect_candidates"
                    ),
                )
                rewrite_frame = prediction_frame.copy()
                rewrite_frame["inferred_dialect"] = inferred_dialects
                rewrite_frame["inferred_dialect_probabilities"] = (
                    dialect_probabilities.tolist()
                )
                rewrite_frame["prediction"] = rewrites
                rewrite_metrics, _ = normalization_metrics(rewrite_frame)
                rewrite_frame.to_parquet(
                    output_dir
                    / f"normalization_inferred_rewrite_{family}_{severity:.2f}.parquet",
                    index=False,
                )
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "task": "normalization",
                        "system_view": "source_blind_baseline",
                        "family": family,
                        "severity": severity,
                        "metric": "macro_chrfpp",
                        "value": float(rewrite_metrics["macro_chrfpp"]),
                    }
                )
                fused_values, selector_margins, selected_neural = (
                    select_normalization_candidates(
                        perturbed_sources,
                        predictions,
                        rewrites,
                        dialect_probabilities,
                        fusion["normalization_selector"],
                    )
                )
                fused_frame = prediction_frame.copy()
                fused_frame["neural_prediction"] = predictions
                fused_frame["rewrite_prediction"] = rewrites
                fused_frame["inferred_dialect"] = inferred_dialects
                fused_frame["selector_predicted_neural_margin"] = selector_margins
                fused_frame["selected_neural"] = selected_neural
                fused_frame["prediction"] = fused_values
                fused_metrics, _ = normalization_metrics(fused_frame)
                fused_frame.to_parquet(
                    output_dir
                    / f"normalization_fused_{family}_{severity:.2f}.parquet",
                    index=False,
                )
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "task": "normalization",
                        "system_view": "source_blind_fused",
                        "family": family,
                        "severity": severity,
                        "metric": "macro_chrfpp",
                        "value": float(fused_metrics["macro_chrfpp"]),
                    }
                )
            del norm_model
            torch.cuda.empty_cache() if device.type == "cuda" else None

            id_model = load_model(id_checkpoint, device)
            temperature = float(
                json.loads(
                    (task_root / "stage_id/temperature_calibration.json").read_text(
                        encoding="utf-8"
                    )
                )["temperature"]
            )
            clean_id_predictions = None
            for (family, severity), values in perturbations.items():
                altered = id_frame.copy()
                altered["text_model"] = values["identification"]
                encoded = EncodedDataset(
                    [
                        encode_identification(
                            row,
                            tokenizer,
                            max_length=256,
                            source_to_id=maps["source_to_id"],
                            group_to_id=maps["group_to_id"],
                        )
                        for row in altered.to_dict("records")
                    ]
                )
                if severity == 0.0 and clean_id_predictions is not None:
                    predictions = clean_id_predictions.copy(deep=True)
                else:
                    predictions, _ = predict_identification(
                        id_model,
                        encoded,
                        device=device,
                        pad_token_id=pad_token_id,
                        batch_size=int(config["identification_batch_size"]),
                        temperature=temperature,
                    )
                    if severity == 0.0:
                        clean_id_predictions = predictions.copy(deep=True)
                probabilities = np.asarray(predictions["probabilities"].tolist())
                metrics, _, _ = classification_metrics(
                    predictions["label_id"], predictions["prediction_id"], probabilities
                )
                predictions["family"] = family
                predictions["severity"] = severity
                predictions["text_clean"] = id_frame["text_model"].astype(str).tolist()
                predictions["text_perturbed"] = values["identification"]
                predictions.to_parquet(
                    output_dir / f"identification_{family}_{severity:.2f}.parquet", index=False
                )
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "task": "identification",
                        "system_view": "raw_neural",
                        "family": family,
                        "severity": severity,
                        "metric": "regional_macro_f1",
                        "value": float(metrics["regional_macro_f1"]),
                    }
                )
                svm_probabilities = classical_identification_probabilities(
                    [str(value) for value in values["identification"]],
                    fusion["svm"],
                )
                svm_prediction_ids = svm_probabilities.argmax(axis=1)
                svm_metrics, _, _ = classification_metrics(
                    predictions["label_id"], svm_prediction_ids, svm_probabilities
                )
                svm_output = predictions.copy()
                svm_output["probabilities"] = svm_probabilities.tolist()
                svm_output["prediction_id"] = svm_prediction_ids
                svm_output.to_parquet(
                    output_dir
                    / f"identification_char_svm_{family}_{severity:.2f}.parquet",
                    index=False,
                )
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "task": "identification",
                        "system_view": "source_blind_baseline",
                        "family": family,
                        "severity": severity,
                        "metric": "regional_macro_f1",
                        "value": float(svm_metrics["regional_macro_f1"]),
                    }
                )
                fused_probabilities = blend_identification_probabilities(
                    probabilities,
                    svm_probabilities,
                    neural_weight=float(
                        fusion["identification_report"]["selected_alpha_neural"]
                    ),
                )
                fused_prediction_ids = fused_probabilities.argmax(axis=1)
                fused_metrics, _, _ = classification_metrics(
                    predictions["label_id"],
                    fused_prediction_ids,
                    fused_probabilities,
                )
                fused_output = predictions.copy()
                fused_output["neural_probabilities"] = probabilities.tolist()
                fused_output["svm_probabilities"] = svm_probabilities.tolist()
                fused_output["probabilities"] = fused_probabilities.tolist()
                fused_output["prediction_id"] = fused_prediction_ids
                fused_output.to_parquet(
                    output_dir
                    / f"identification_fused_{family}_{severity:.2f}.parquet",
                    index=False,
                )
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "task": "identification",
                        "system_view": "source_blind_fused",
                        "family": family,
                        "severity": severity,
                        "metric": "regional_macro_f1",
                        "value": float(fused_metrics["regional_macro_f1"]),
                    }
                )
            del id_model
            torch.cuda.empty_cache() if device.type == "cuda" else None
            (output_dir / "robustness_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "variant": variant,
                        "seed": seed,
                        "checkpoint_sha256": expected_hashes,
                        "perturbation_seed": config["perturbation_seed"],
                        "sampling_seed": config["sampling_seed"],
                        "sampling_policy": config["sampling_policy"],
                        "families": config["families"],
                        "severities": config["severities"],
                        "examples": {
                            "normalization": len(norm_frame),
                            "identification": len(id_frame),
                        },
                        "system_views": [
                            "raw_neural",
                            "source_blind_baseline",
                            "source_blind_fused",
                        ],
                        "fusion_artifact_sha256": fusion["artifact_sha256"],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    results = pd.DataFrame(rows)
    baseline = results.loc[results["severity"].eq(0.0), [
        "variant", "seed", "task", "system_view", "family", "value"
    ]].rename(columns={"value": "clean_value"})
    results = results.merge(
        baseline,
        on=["variant", "seed", "task", "system_view", "family"],
        validate="many_to_one",
    )
    results["absolute_drop_from_clean"] = results["clean_value"] - results["value"]
    results["relative_drop_from_clean"] = (
        results["absolute_drop_from_clean"] / results["clean_value"].clip(lower=1e-9)
    )
    summary_rows = []
    for keys, group in results.groupby(
        ["variant", "seed", "task", "system_view", "family"],
        sort=True,
        observed=True,
    ):
        ordered = group.sort_values("severity")
        maximum = float(ordered["severity"].max())
        auc = float(np.trapezoid(ordered["value"], ordered["severity"])) / max(maximum, 1e-9)
        summary_rows.append(
            {
                "variant": keys[0],
                "seed": keys[1],
                "task": keys[2],
                "system_view": keys[3],
                "family": keys[4],
                "normalized_robustness_auc": auc,
                "clean_value": float(ordered["clean_value"].iloc[0]),
                "worst_value": float(ordered["value"].min()),
                "maximum_absolute_drop": float(ordered["absolute_drop_from_clean"].max()),
                "maximum_relative_drop": float(ordered["relative_drop_from_clean"].max()),
            }
        )
    summaries = pd.DataFrame(summary_rows)
    report_dir = PROJECT / "reports/robustness" / str(config["protocol_id"])
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "robustness_curves.csv", index=False)
    summaries.to_csv(report_dir / "robustness_summary.csv", index=False)
    print(f"Robustness evaluation complete: {report_dir}")


if __name__ == "__main__":
    main()
