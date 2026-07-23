#!/usr/bin/env python3
"""Quantify selection-conditioned development fusion gains with paired resampling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
for path in (PROJECT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.run_statistics import (  # noqa: E402
    align_pair,
    paired_bootstrap,
    paired_randomization,
)


VARIANTS = ("M0", "M1")
SEEDS = (1701, 2903, 4307)
BOOTSTRAP_REPLICATES = 5000
RANDOMIZATION_REPLICATES = 5000


def infer(
    frames: dict[int, pd.DataFrame], *, task: str, bootstrap_seed: int
) -> dict:
    draws, points = paired_bootstrap(
        frames,
        task=task,
        replicates=BOOTSTRAP_REPLICATES,
        seed=bootstrap_seed,
    )
    randomization, observed = paired_randomization(
        frames,
        task=task,
        replicates=RANDOMIZATION_REPLICATES,
        seed=bootstrap_seed + 100_000,
    )
    lower, upper = np.quantile(draws["delta"], [0.025, 0.975])
    p_two_sided = (
        int(randomization["null_delta"].abs().ge(abs(observed)).sum()) + 1
    ) / (len(randomization) + 1)
    return {
        "task": task,
        "run_points": points,
        "mean_delta": float(np.mean([point["delta"] for point in points])),
        "bootstrap_mean_delta": float(draws["delta"].mean()),
        "confidence_lower_95": float(lower),
        "confidence_upper_95": float(upper),
        "bootstrap_probability_positive": float(draws["delta"].gt(0).mean()),
        "paired_randomization_observed_delta": float(observed),
        "paired_randomization_p_two_sided": float(p_two_sided),
        "bootstrap_replicates": len(draws),
        "randomization_replicates": len(randomization),
    }


def normalization_frames() -> dict[int, pd.DataFrame]:
    control = pd.read_parquet(
        PROJECT
        / "predictions/development_source_blind/"
        "N_WORD_REWRITE_INFERRED_SUPPORTED_DIALECT_validation.parquet"
    )
    frames = {}
    index = 0
    for variant in VARIANTS:
        for seed in SEEDS:
            treatment = pd.read_parquet(
                PROJECT
                / "predictions/development_fusion/normalization_selector_v2"
                / f"{variant}__base/{seed}/validation_predictions_oof.parquet"
            )
            frames[index] = align_pair(treatment, control, "normalization")
            index += 1
    return frames


def identification_frames() -> dict[int, pd.DataFrame]:
    frames = {}
    index = 0
    for variant in VARIANTS:
        for seed in SEEDS:
            root = PROJECT / f"runs/task/boichitro_q1_v1/{variant}__base/{seed}/stage_id"
            selection = json.loads(
                (root / "best_selection.json").read_text(encoding="utf-8")
            )
            control = pd.read_parquet(
                root
                / f"validation_predictions_epoch_{int(selection['validation_id']):02d}.parquet"
            )
            treatment = pd.read_parquet(
                PROJECT
                / "predictions/development_fusion/id_probability_blend_v1"
                / f"{variant}__base/{seed}/validation_predictions.parquet"
            )
            frames[index] = align_pair(treatment, control, "identification")
            index += 1
    return frames


def main() -> None:
    normalization = infer(
        normalization_frames(), task="normalization", bootstrap_seed=87123
    )
    identification = infer(
        identification_frames(), task="identification", bootstrap_seed=97123
    )
    report = {
        "status": "COMPLETE_VALIDATION_ONLY",
        "protocol_id": "boichitro_development_fusion_audit_v1",
        "test_data_access": False,
        "source_blind": True,
        "selection_variants": list(VARIANTS),
        "selection_seeds": list(SEEDS),
        "normalization_comparison": (
            "semantic-group-held-out OOF selector V2 minus inferred-supported-dialect rewrite"
        ),
        "identification_comparison": (
            "calibrated neural/SVM blend minus raw neural classifier"
        ),
        "interval_method": (
            "paired hierarchical bootstrap over six model runs with semantic-group "
            "draws synchronized across repeated evaluation rows and cross-dialect "
            "realizations"
        ),
        "p_value_method": (
            "paired randomization at the global semantic-group level with swaps synchronized "
            "across model runs and cross-dialect realizations"
        ),
        "inference_scope": "selection_conditioned_exploratory_development_diagnostic",
        "selection_reuse_disclosure": (
            "The selector family/threshold and identification blend weight were chosen "
            "using these development rows. Intervals and randomization p-values condition "
            "on those selected settings and do not include selection uncertainty."
        ),
        "confirmatory_inference": False,
        "normalization": normalization,
        "identification": identification,
        "reporting_constraint": (
            "development evidence only; raw locked neural outputs remain primary for "
            "architectural and source-OOD claims"
        ),
    }
    root = PROJECT / "reports/model"
    (root / "development_fusion_uncertainty.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for task in ("normalization", "identification"):
        summary = report[task]
        rows.append(
            {
                "task": task,
                **{key: value for key, value in summary.items() if key != "run_points"},
            }
        )
    pd.DataFrame(rows).to_csv(
        root / "development_fusion_uncertainty.csv", index=False
    )
    lines = [
        "# Development-only source-blind fusion audit",
        "",
        "No test split was accessed. Each global semantic group receives one bootstrap multiplicity or randomization swap reused across model runs and cross-dialect realizations.",
        "",
        "**Selection-reuse warning:** fusion settings were chosen on these development rows. The intervals and p-values condition on the selected settings, do not include selection uncertainty, and are exploratory rather than confirmatory.",
        "",
        "| Task | Paired gain | 95% hierarchical CI | Randomization p (two-sided) |",
        "|---|---:|---:|---:|",
    ]
    for task, summary in (("Normalization chrF++", normalization), ("Identification regional macro-F1", identification)):
        lines.append(
            f"| {task} | {summary['mean_delta']:.4f} | "
            f"[{summary['confidence_lower_95']:.4f}, {summary['confidence_upper_95']:.4f}] | "
            f"{summary['paired_randomization_p_two_sided']:.6f} |"
        )
    lines.extend(
        [
            "",
            "These are selection-conditioned development diagnostics, not confirmatory or locked source-OOD claims. The raw neural view is retained separately.",
            "",
        ]
    )
    (root / "DEVELOPMENT_FUSION_UNCERTAINTY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({"normalization": normalization, "identification": identification}, indent=2))


if __name__ == "__main__":
    main()
