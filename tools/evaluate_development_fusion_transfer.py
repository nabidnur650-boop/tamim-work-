#!/usr/bin/env python3
"""Apply already-fixed fusion to later M2/M3 development runs without tuning."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
for path in (PROJECT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from boichitro.fusion import (  # noqa: E402
    blend_identification_probabilities,
    classical_identification_probabilities,
    inferred_dialect_rewrite,
    select_normalization_candidates,
    temperature_scale_probabilities,
)
from boichitro.metrics import classification_metrics, normalization_metrics  # noqa: E402
from boichitro.tokenization import sha256_file  # noqa: E402
from tools.evaluate_locked import load_fusion_resources  # noqa: E402


VARIANTS = ("M2", "M3")
SEEDS = (1701, 2903, 4307)


def complete(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "COMPLETE"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def main() -> None:
    config = yaml.safe_load(
        (PROJECT / "configs/locked_evaluation.yaml").read_text(encoding="utf-8")
    )
    fusion = load_fusion_resources(config, None)
    if fusion is None:
        raise RuntimeError("Fusion resources are disabled")
    id_evaluation = pd.read_parquet(
        PROJECT / "data/final/v1/identification_evaluation.parquet"
    )
    id_validation = id_evaluation.loc[
        id_evaluation["split"].eq("validation")
    ].copy()
    id_text_by_row = dict(
        zip(
            id_validation["row_id"].astype(str),
            id_validation["text_model"].astype(str),
            strict=True,
        )
    )
    output_root = PROJECT / "predictions/development_fusion/architecture_transfer_v1"
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            root = PROJECT / f"runs/task/boichitro_q1_v1/{variant}__base/{seed}"
            if not (
                complete(root / "stage_s/training_report.json")
                and complete(root / "stage_id/training_report.json")
            ):
                continue
            destination = output_root / f"{variant}__base/{seed}"
            destination.mkdir(parents=True, exist_ok=True)

            norm_selection = json.loads(
                (root / "stage_s/best_selection.json").read_text(encoding="utf-8")
            )
            norm = pd.read_parquet(
                root
                / "stage_s"
                / f"validation_predictions_epoch_{int(norm_selection['validation_id']):02d}.parquet"
            )
            sources = norm["source"].astype(str).tolist()
            dialect_probabilities = classical_identification_probabilities(
                sources, fusion["svm"]
            )
            rewrite, inferred = inferred_dialect_rewrite(
                sources,
                dialect_probabilities,
                fusion["rewrite_mapping"],
                allowed_dialects=fusion["normalization_selector"].get(
                    "inferred_dialect_candidates"
                ),
            )
            fused, margins, selected_neural = select_normalization_candidates(
                sources,
                norm["prediction"].astype(str).tolist(),
                rewrite,
                dialect_probabilities,
                fusion["normalization_selector"],
            )
            raw_norm_metrics, _ = normalization_metrics(norm)
            rewrite_frame = norm.copy()
            rewrite_frame["prediction"] = rewrite
            rewrite_metrics, _ = normalization_metrics(rewrite_frame)
            fused_frame = norm.copy()
            fused_frame["neural_prediction"] = norm["prediction"].astype(str)
            fused_frame["rewrite_prediction"] = rewrite
            fused_frame["inferred_dialect"] = inferred
            fused_frame["selector_predicted_neural_margin"] = margins
            fused_frame["selected_neural"] = selected_neural
            fused_frame["prediction"] = fused
            fused_norm_metrics, _ = normalization_metrics(fused_frame)
            fused_frame.to_parquet(
                destination / "normalization_validation_predictions.parquet",
                index=False,
            )

            id_selection = json.loads(
                (root / "stage_id/best_selection.json").read_text(encoding="utf-8")
            )
            ident = pd.read_parquet(
                root
                / "stage_id"
                / f"validation_predictions_epoch_{int(id_selection['validation_id']):02d}.parquet"
            )
            temperature = float(
                json.loads(
                    (root / "stage_id/temperature_calibration.json").read_text(
                        encoding="utf-8"
                    )
                )["temperature"]
            )
            neural = temperature_scale_probabilities(
                np.asarray(ident["probabilities"].tolist(), dtype=np.float64),
                temperature=temperature,
            )
            texts = [id_text_by_row[str(row_id)] for row_id in ident["row_id"]]
            svm = classical_identification_probabilities(texts, fusion["svm"])
            fused_probabilities = blend_identification_probabilities(
                neural,
                svm,
                neural_weight=float(
                    fusion["identification_report"]["selected_alpha_neural"]
                ),
            )
            labels = ident["label_id"].to_numpy(dtype=np.int64)
            raw_id_metrics, _, _ = classification_metrics(
                labels, neural.argmax(axis=1), neural
            )
            fused_id_metrics, _, _ = classification_metrics(
                labels, fused_probabilities.argmax(axis=1), fused_probabilities
            )
            id_output = ident.copy()
            id_output["neural_probabilities_calibrated"] = neural.tolist()
            id_output["svm_probabilities"] = svm.tolist()
            id_output["probabilities"] = fused_probabilities.tolist()
            id_output["prediction_id"] = fused_probabilities.argmax(axis=1)
            id_output.to_parquet(
                destination / "identification_validation_predictions.parquet",
                index=False,
            )
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "normalization_raw_macro_chrfpp": raw_norm_metrics["macro_chrfpp"],
                    "normalization_rewrite_macro_chrfpp": rewrite_metrics["macro_chrfpp"],
                    "normalization_fused_macro_chrfpp": fused_norm_metrics["macro_chrfpp"],
                    "normalization_fused_worst_dialect_chrfpp": fused_norm_metrics[
                        "worst_dialect_chrfpp"
                    ],
                    "normalization_selected_neural_fraction": float(
                        selected_neural.mean()
                    ),
                    "identification_raw_regional_macro_f1": raw_id_metrics[
                        "regional_macro_f1"
                    ],
                    "identification_fused_regional_macro_f1": fused_id_metrics[
                        "regional_macro_f1"
                    ],
                    "identification_fused_ece_15": fused_id_metrics["ece_15"],
                }
            )

    expected = len(VARIANTS) * len(SEEDS)
    status = (
        "COMPLETE_VALIDATION_ONLY" if len(rows) == expected else "PARTIAL_VALIDATION_ONLY"
    )
    report = {
        "status": status,
        "protocol_id": "boichitro_fixed_fusion_architecture_transfer_v1",
        "test_data_access": False,
        "source_blind": True,
        "selection_artifacts_modified": False,
        "selection_variants": ["M0", "M1"],
        "transfer_variants": list(VARIANTS),
        "completed_runs": len(rows),
        "expected_runs": expected,
        "interpretation": (
            "fixed-selector transfer to later architectures; validation rows are shared, "
            "so this is not an independent-row generalization estimate"
        ),
        "normalization_selection_report_sha256": sha256_file(
            PROJECT / "reports/model/normalization_fusion_selection_v2.json"
        ),
        "identification_selection_report_sha256": sha256_file(
            PROJECT / "reports/model/id_fusion_selection.json"
        ),
        "runs": rows,
    }
    report_root = PROJECT / "reports/model"
    (report_root / "development_fusion_architecture_transfer.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(
        report_root / "development_fusion_architecture_transfer.csv", index=False
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
