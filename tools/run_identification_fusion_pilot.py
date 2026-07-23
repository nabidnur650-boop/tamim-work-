#!/usr/bin/env python3
"""Select a source-blind neural/SVM probability blend on validation only."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.metrics import classification_metrics  # noqa: E402
from boichitro.tokenization import DIALECTS, nfc  # noqa: E402


SELECTION_VARIANTS = ("M0", "M1")
SELECTION_SEEDS = (1701, 2903, 4307)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def svm_probabilities(frame: pd.DataFrame) -> np.ndarray:
    path = PROJECT / "artifacts/baselines/id_char_tfidf_svm.pkl"
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    matrix = artifact["vectorizer"].transform(frame["text_model"].map(nfc))
    logits = np.asarray(artifact["model"].decision_function(matrix), dtype=np.float64)
    if logits.ndim == 1:
        logits = np.column_stack((-logits, logits))
    if logits.shape[1] != len(DIALECTS):
        expanded = np.full((len(logits), len(DIALECTS)), -1e9, dtype=np.float64)
        expanded[:, np.asarray(artifact["model"].classes_, dtype=int)] = logits
        logits = expanded
    return softmax(logits / float(artifact["temperature"]), axis=1)


def calibrated_neural(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    return softmax(logits, axis=1)


def main() -> None:
    evaluation = pd.read_parquet(PROJECT / "data/final/v1/identification_evaluation.parquet")
    validation = evaluation.loc[evaluation["split"].eq("validation")].copy()
    identification_train = pd.read_parquet(
        PROJECT / "data/final/v1/identification_train.parquet",
        columns=["source_id"],
    )
    validation_ids = set(validation["row_id"].astype(str))
    svm = svm_probabilities(validation)
    svm_by_id = {
        str(row_id): probability
        for row_id, probability in zip(validation["row_id"], svm, strict=True)
    }

    runs: list[dict] = []
    for variant in SELECTION_VARIANTS:
        for seed in SELECTION_SEEDS:
            stage = PROJECT / f"runs/task/boichitro_q1_v1/{variant}__base/{seed}/stage_id"
            selection = json.loads((stage / "best_selection.json").read_text(encoding="utf-8"))
            validation_id = int(selection["validation_id"])
            predictions_path = stage / f"validation_predictions_epoch_{validation_id:02d}.parquet"
            predictions = pd.read_parquet(predictions_path)
            if set(predictions["row_id"].astype(str)) != validation_ids:
                raise RuntimeError(f"Validation row mismatch: {predictions_path}")
            temperature = float(
                json.loads((stage / "temperature_calibration.json").read_text(encoding="utf-8"))[
                    "temperature"
                ]
            )
            neural = calibrated_neural(
                np.asarray(predictions["probabilities"].tolist(), dtype=np.float64),
                temperature,
            )
            classical = np.stack(
                [svm_by_id[str(row_id)] for row_id in predictions["row_id"]]
            )
            runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "predictions": predictions,
                    "neural": neural,
                    "svm": classical,
                    "labels": predictions["label_id"].to_numpy(dtype=np.int64),
                    "temperature": temperature,
                    "predictions_path": str(predictions_path.relative_to(PROJECT)),
                }
            )

    alpha_grid = np.linspace(0.0, 1.0, 41)
    grid_rows: list[dict] = []
    run_rows: list[dict] = []
    for alpha in alpha_grid:
        per_run = []
        for run in runs:
            probabilities = alpha * run["neural"] + (1.0 - alpha) * run["svm"]
            metrics, _, _ = classification_metrics(
                run["labels"], probabilities.argmax(axis=1), probabilities
            )
            row = {
                "alpha_neural": float(alpha),
                "variant": run["variant"],
                "seed": run["seed"],
                **{key: float(value) for key, value in metrics.items()},
            }
            per_run.append(row)
            run_rows.append(row)
        grid_rows.append(
            {
                "alpha_neural": float(alpha),
                "mean_regional_macro_f1": float(
                    np.mean([row["regional_macro_f1"] for row in per_run])
                ),
                "mean_macro_f1_13": float(np.mean([row["macro_f1_13"] for row in per_run])),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in per_run])),
                "mean_worst_present_dialect_f1": float(
                    np.mean([row["worst_present_dialect_f1"] for row in per_run])
                ),
                "mean_ece_15": float(np.mean([row["ece_15"] for row in per_run])),
            }
        )

    selected = max(
        grid_rows,
        key=lambda row: (
            row["mean_regional_macro_f1"],
            row["mean_worst_present_dialect_f1"],
            -row["mean_ece_15"],
            -abs(row["alpha_neural"] - 0.5),
        ),
    )
    alpha = float(selected["alpha_neural"])
    output_root = PROJECT / "predictions/development_fusion/id_probability_blend_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    selected_runs = []
    for run in runs:
        probabilities = alpha * run["neural"] + (1.0 - alpha) * run["svm"]
        metrics, by_class, confusion = classification_metrics(
            run["labels"], probabilities.argmax(axis=1), probabilities
        )
        destination = output_root / f"{run['variant']}__base/{run['seed']}"
        destination.mkdir(parents=True, exist_ok=True)
        predictions = run["predictions"].copy()
        predictions["neural_probabilities_calibrated"] = run["neural"].tolist()
        predictions["svm_probabilities"] = run["svm"].tolist()
        predictions["probabilities"] = probabilities.tolist()
        predictions["prediction_id"] = probabilities.argmax(axis=1)
        predictions.to_parquet(destination / "validation_predictions.parquet", index=False)
        by_class.to_csv(destination / "validation_by_class.csv", index=False)
        np.save(destination / "validation_confusion.npy", confusion)
        (destination / "validation_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        selected_runs.append(
            {
                "variant": run["variant"],
                "seed": run["seed"],
                "neural_temperature": run["temperature"],
                "source_predictions": run["predictions_path"],
                **{key: float(value) for key, value in metrics.items()},
            }
        )

    report_root = PROJECT / "reports/model"
    pd.DataFrame(grid_rows).to_csv(report_root / "id_fusion_alpha_grid.csv", index=False)
    pd.DataFrame(run_rows).to_csv(report_root / "id_fusion_per_run_grid.csv", index=False)
    artifact_path = PROJECT / "artifacts/baselines/id_char_tfidf_svm.pkl"
    report = {
        "status": "COMPLETE_VALIDATION_ONLY",
        "protocol_id": "boichitro_id_probability_blend_v1",
        "test_data_access": False,
        "source_blind": True,
        "development_score_scope": "selection_conditioned_exploratory",
        "confirmatory_inference": False,
        "selection_reuse_disclosure": (
            "The blend weight and its reported development score use the same "
            "validation rows; locked evaluation is required for confirmation."
        ),
        "selection_scope": "in-domain development validation",
        "source_ood_examples_used_for_selection": 0,
        "validation_sources": sorted(validation["source_id"].astype(str).unique()),
        "validation_sources_seen_in_training": sorted(
            set(validation["source_id"].astype(str))
            & set(identification_train["source_id"].astype(str))
        ),
        "reporting_role": (
            "separate frozen system view; raw neural outputs remain primary for "
            "source-OOD architectural inference"
        ),
        "selection_variants": list(SELECTION_VARIANTS),
        "selection_seeds": list(SELECTION_SEEDS),
        "selection_rule": (
            "maximize mean regional macro-F1 across six completed M0/M1 development runs; "
            "tie-break by worst-class F1, ECE, then proximity to equal weighting"
        ),
        "selected_alpha_neural": alpha,
        "selected_alpha_svm": 1.0 - alpha,
        "selected_summary": selected,
        "svm_artifact": str(artifact_path.relative_to(PROJECT)),
        "svm_artifact_sha256": sha256_file(artifact_path),
        "runs": selected_runs,
    }
    (report_root / "id_fusion_selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selected_summary"], indent=2))


if __name__ == "__main__":
    main()
