#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from boichitro.data import load_encoded_cache  # noqa: E402
from boichitro.fusion import (  # noqa: E402
    blend_identification_probabilities,
    classical_identification_probabilities,
)
from boichitro.inference import predict_identification  # noqa: E402
from boichitro.metrics import classification_metrics  # noqa: E402
from boichitro.modeling import BoichitroConfig, BoichitroForMultiTask  # noqa: E402
from boichitro.protocol import require_frozen_artifact, require_protocol_freeze  # noqa: E402
from boichitro.tokenization import load_tokenizer, sha256_file  # noqa: E402
from evaluate_locked import load_fusion_resources  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-time locked evaluation of bidirectional ID specialization."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/bidirectional_identification.yaml",
    )
    parser.add_argument(
        "--locked-config",
        type=Path,
        default=PROJECT / "configs/locked_evaluation.yaml",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--allow-identical-rerun", action="store_true")
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> BoichitroForMultiTask:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = BoichitroForMultiTask(BoichitroConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    if not model.config.bidirectional_attention:
        raise RuntimeError("Locked M3B checkpoint is not bidirectional")
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    locked = yaml.safe_load(args.locked_config.read_text(encoding="utf-8"))
    freeze = (
        require_protocol_freeze(PROJECT, str(locked["protocol_freeze_id"]))
        if locked.get("protocol_freeze_id")
        else None
    )
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    tokenizer = load_tokenizer(PROJECT / "artifacts/tokenizers/frozen")
    pad_token_id = int(tokenizer.token_to_id("<pad>"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    fusion = load_fusion_resources(locked, freeze)

    for seed in seeds:
        run_root = (
            PROJECT
            / "runs/task"
            / str(config["protocol_id"])
            / str(config["variant_id"])
            / str(seed)
        )
        checkpoint = run_root / "stage_id/best_checkpoint.pt"
        calibration_path = run_root / "stage_id/temperature_calibration.json"
        if not checkpoint.exists() or not calibration_path.exists():
            raise FileNotFoundError(f"Frozen M3B artifacts missing under {run_root}")
        checkpoint_hash = (
            require_frozen_artifact(PROJECT, freeze, checkpoint)
            if freeze is not None
            else sha256_file(checkpoint)
        )
        if freeze is not None:
            require_frozen_artifact(PROJECT, freeze, calibration_path)
        output_dir = (
            PROJECT
            / "predictions"
            / str(locked["protocol_id"])
            / f"{config['variant_id']}__base"
            / str(seed)
        )
        manifest_path = output_dir / "evaluation_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("checkpoint_sha256") != checkpoint_hash:
                raise RuntimeError(
                    f"Refusing to overwrite outputs from a different checkpoint: {output_dir}"
                )
            if not args.allow_identical_rerun:
                print(f"Locked M3B evaluation already complete, skipping seed={seed}")
                continue
        output_dir.mkdir(parents=True, exist_ok=True)
        model = load_model(checkpoint, device)
        temperature = float(
            json.loads(calibration_path.read_text(encoding="utf-8"))["temperature"]
        )
        results = {}
        fused_results = {}
        classical_results = {}
        for track, details in locked["identification"]["tracks"].items():
            relative_path = details["cache"] if isinstance(details, dict) else details
            dataset = load_encoded_cache(PROJECT / relative_path)[0]
            predictions, _ = predict_identification(
                model,
                dataset,
                device=device,
                pad_token_id=pad_token_id,
                batch_size=int(locked["identification"]["batch_size"]),
                temperature=temperature,
            )
            probabilities = np.asarray(predictions["probabilities"].tolist(), dtype=np.float64)
            metrics, by_class, matrix = classification_metrics(
                predictions["label_id"].to_numpy(),
                predictions["prediction_id"].to_numpy(),
                probabilities,
            )
            results[track] = metrics
            predictions.to_parquet(
                output_dir / f"identification_{track}.parquet", index=False
            )
            by_class.to_csv(
                output_dir / f"identification_{track}_by_class.csv", index=False
            )
            np.save(output_dir / f"identification_{track}_confusion.npy", matrix)
            (output_dir / f"identification_{track}_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            if fusion is not None:
                if not isinstance(details, dict):
                    raise TypeError("Fused M3B evaluation requires frame and split metadata")
                text_frame = pd.read_parquet(
                    PROJECT / details["frame"],
                    columns=["row_id", "split", "text_model"],
                )
                text_frame = text_frame.loc[
                    text_frame["split"].eq(str(details["split"]))
                ]
                text_by_id = dict(
                    zip(
                        text_frame["row_id"].astype(str),
                        text_frame["text_model"].astype(str),
                        strict=True,
                    )
                )
                texts = [text_by_id[str(row_id)] for row_id in predictions["row_id"]]
                svm = classical_identification_probabilities(texts, fusion["svm"])
                svm_ids = svm.argmax(axis=1)
                svm_metrics, svm_by_class, svm_matrix = classification_metrics(
                    predictions["label_id"].to_numpy(), svm_ids, svm
                )
                classical_results[track] = svm_metrics
                svm_output = predictions.copy()
                svm_output["probabilities"] = svm.tolist()
                svm_output["prediction_id"] = svm_ids
                svm_output.to_parquet(
                    output_dir / f"identification_char_svm_{track}.parquet",
                    index=False,
                )
                svm_by_class.to_csv(
                    output_dir / f"identification_char_svm_{track}_by_class.csv",
                    index=False,
                )
                np.save(
                    output_dir / f"identification_char_svm_{track}_confusion.npy",
                    svm_matrix,
                )
                (
                    output_dir / f"identification_char_svm_{track}_metrics.json"
                ).write_text(
                    json.dumps(svm_metrics, indent=2) + "\n", encoding="utf-8"
                )
                neural = np.asarray(
                    predictions["probabilities"].tolist(), dtype=np.float64
                )
                fused = blend_identification_probabilities(
                    neural,
                    svm,
                    neural_weight=float(
                        fusion["identification_report"]["selected_alpha_neural"]
                    ),
                )
                fused_ids = fused.argmax(axis=1)
                fused_metrics, fused_by_class, fused_matrix = classification_metrics(
                    predictions["label_id"].to_numpy(), fused_ids, fused
                )
                fused_results[track] = fused_metrics
                fused_output = predictions.copy()
                fused_output["neural_probabilities"] = neural.tolist()
                fused_output["svm_probabilities"] = svm.tolist()
                fused_output["probabilities"] = fused.tolist()
                fused_output["prediction_id"] = fused_ids
                fused_output.to_parquet(
                    output_dir / f"identification_fused_{track}.parquet",
                    index=False,
                )
                fused_by_class.to_csv(
                    output_dir / f"identification_fused_{track}_by_class.csv",
                    index=False,
                )
                np.save(
                    output_dir / f"identification_fused_{track}_confusion.npy",
                    fused_matrix,
                )
                (output_dir / f"identification_fused_{track}_metrics.json").write_text(
                    json.dumps(fused_metrics, indent=2) + "\n", encoding="utf-8"
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        manifest = {
            "status": "COMPLETE_LOCKED_ID_EVALUATION",
            "protocol_id": locked["protocol_id"],
            "task_protocol_id": config["protocol_id"],
            "variant": config["variant_id"],
            "seed": seed,
            "checkpoint": str(checkpoint.relative_to(PROJECT)),
            "checkpoint_sha256": checkpoint_hash,
            "temperature": temperature,
            "identification": results,
            "identification_fused": fused_results,
            "identification_classical_baseline": classical_results,
            "fusion": {
                "enabled": fusion is not None,
                "artifact_sha256": (
                    fusion["artifact_sha256"] if fusion is not None else {}
                ),
            },
            "tokenizer_sha256": sha256_file(
                PROJECT / "artifacts/tokenizers/frozen/tokenizer.json"
            ),
            "evaluation_config_sha256": sha256_file(args.locked_config),
            "protocol_freeze": {
                "protocol_id": freeze["protocol_id"],
                "code_sha256": freeze["code_sha256"],
                "config_sha256": freeze["config_sha256"],
                "selected_artifacts_sha256": freeze["selected_artifacts_sha256"],
            }
            if freeze is not None
            else None,
            "test_policy": locked["test_policy"],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Locked M3B identification evaluation complete: seed={seed}", flush=True)


if __name__ == "__main__":
    main()
