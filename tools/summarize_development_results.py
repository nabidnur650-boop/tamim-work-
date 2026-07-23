#!/usr/bin/env python3
"""Build reproducible development-only summaries from selected checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
VARIANTS = ("M0", "M1", "M2", "M3")
SEEDS = (1701, 2903, 4307)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_development_rows(
    project: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main_rows: list[dict[str, Any]] = []
    dialect_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    task_root = project / "runs/task/boichitro_q1_v1"

    for variant in VARIANTS:
        for seed in SEEDS:
            run_root = task_root / f"{variant}__base" / str(seed)
            norm = load_json(run_root / "stage_s/best_selection.json")
            ident = load_json(run_root / "stage_id/best_selection.json")
            if not norm or not ident:
                continue
            if norm.get("status") != "SELECTED_ON_VALIDATION":
                continue
            if ident.get("status") != "SELECTED_ON_VALIDATION":
                continue

            norm_values = norm["validation"]
            id_values = ident["validation"]
            norm_step = int(norm["global_step"])
            id_epoch = int(ident["epoch"])
            main_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "status": "COMPLETE_VALIDATION_ONLY",
                    "norm_macro_chrfpp": float(norm_values["macro_chrfpp"]),
                    "norm_worst_dialect_chrfpp": float(
                        norm_values["worst_dialect_chrfpp"]
                    ),
                    "replay_nll": float(norm_values["replay_nll"]),
                    "replay_degradation_percent": 100.0
                    * float(norm_values["replay_relative_degradation"]),
                    "norm_selected_step": norm_step,
                    "id_accuracy": float(id_values["accuracy"]),
                    "id_balanced_accuracy": float(id_values["balanced_accuracy"]),
                    "id_macro_f1_13": float(id_values["macro_f1_13"]),
                    "id_regional_macro_f1": float(id_values["regional_macro_f1"]),
                    "id_mcc": float(id_values["mcc"]),
                    "id_ece_15": float(id_values["ece_15"]),
                    "id_brier": float(id_values["brier"]),
                    "id_worst_present_dialect_f1": float(
                        id_values["worst_present_dialect_f1"]
                    ),
                    "id_selected_epoch": id_epoch,
                }
            )

            dialect_path = (
                run_root
                / "stage_s"
                / f"validation_by_dialect_epoch_{norm_step}.csv"
            )
            class_path = (
                run_root
                / "stage_id"
                / f"validation_by_class_epoch_{id_epoch:02d}.csv"
            )
            if not dialect_path.exists() or not class_path.exists():
                raise FileNotFoundError(
                    f"Selected validation detail is missing for {variant} seed {seed}"
                )
            for row in pd.read_csv(dialect_path).to_dict("records"):
                dialect_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "selected_step": norm_step,
                        **row,
                    }
                )
            for row in pd.read_csv(class_path).to_dict("records"):
                class_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "selected_epoch": id_epoch,
                        **row,
                    }
                )

    main = pd.DataFrame(main_rows)
    dialect = pd.DataFrame(dialect_rows)
    classes = pd.DataFrame(class_rows)
    if not main.empty:
        main = main.sort_values(["variant", "seed"], kind="stable").reset_index(
            drop=True
        )
        dialect = dialect.sort_values(
            ["variant", "seed", "dialect"], kind="stable"
        ).reset_index(drop=True)
        classes = classes.sort_values(
            ["variant", "seed", "label_id"], kind="stable"
        ).reset_index(drop=True)
    return main, dialect, classes


def write_m0_m1_comparisons(
    project: Path, dialect: pd.DataFrame, classes: pd.DataFrame, output: Path
) -> list[Path]:
    written: list[Path] = []
    available = set(dialect["variant"].unique()) if not dialect.empty else set()
    if {"M0", "M1"}.issubset(available):
        baseline_path = (
            project
            / "metrics/normalization/N_WORD_REWRITE_validation_by_dialect.csv"
        )
        baseline = pd.read_csv(baseline_path).set_index("dialect")
        aggregate = (
            dialect.loc[dialect.variant.isin(["M0", "M1"])]
            .groupby(["dialect", "variant"], observed=True)
            .agg(
                validation_rows=("rows", "max"),
                chrfpp_mean=("chrfpp", "mean"),
                chrfpp_sd=("chrfpp", "std"),
            )
            .reset_index()
        )
        rows = []
        for name in sorted(aggregate.dialect.unique()):
            m0 = aggregate.loc[
                (aggregate.dialect == name) & (aggregate.variant == "M0")
            ].iloc[0]
            m1 = aggregate.loc[
                (aggregate.dialect == name) & (aggregate.variant == "M1")
            ].iloc[0]
            rewrite = float(baseline.loc[name, "chrfpp"])
            rows.append(
                {
                    "dialect": name,
                    "validation_rows": int(m0.validation_rows),
                    "word_rewrite_chrfpp": rewrite,
                    "m0_chrfpp_mean": float(m0.chrfpp_mean),
                    "m0_chrfpp_sd": float(m0.chrfpp_sd),
                    "m1_chrfpp_mean": float(m1.chrfpp_mean),
                    "m1_chrfpp_sd": float(m1.chrfpp_sd),
                    "m1_minus_word_rewrite": float(m1.chrfpp_mean) - rewrite,
                }
            )
        path = output / "main_validation_by_dialect_current.csv"
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
        written.append(path)

    available = set(classes["variant"].unique()) if not classes.empty else set()
    if {"M0", "M1"}.issubset(available):
        aggregate = (
            classes.loc[classes.variant.isin(["M0", "M1"])]
            .groupby(["dialect", "variant"], observed=True)
            .agg(
                validation_support=("support", "max"),
                f1_mean=("f1", "mean"),
                f1_sd=("f1", "std"),
            )
            .reset_index()
        )
        rows = []
        for name in sorted(aggregate.dialect.unique()):
            m0 = aggregate.loc[
                (aggregate.dialect == name) & (aggregate.variant == "M0")
            ].iloc[0]
            m1 = aggregate.loc[
                (aggregate.dialect == name) & (aggregate.variant == "M1")
            ].iloc[0]
            rows.append(
                {
                    "dialect": name,
                    "validation_support": int(m0.validation_support),
                    "m0_f1_mean": float(m0.f1_mean),
                    "m0_f1_sd": float(m0.f1_sd),
                    "m1_f1_mean": float(m1.f1_mean),
                    "m1_f1_sd": float(m1.f1_sd),
                    "m1_minus_m0": float(m1.f1_mean) - float(m0.f1_mean),
                }
            )
        path = output / "main_identification_by_class_current.csv"
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
        written.append(path)
    return written


def main() -> None:
    output = PROJECT / "reports/model"
    output.mkdir(parents=True, exist_ok=True)
    main_frame, dialect_frame, class_frame = collect_development_rows(PROJECT)
    if main_frame.empty:
        raise RuntimeError("No complete main development runs were found")

    paths = [
        output / "main_validation_results_current.csv",
        output / "main_validation_by_dialect_all_current.csv",
        output / "main_identification_by_class_all_current.csv",
    ]
    main_frame.to_csv(paths[0], index=False, float_format="%.6f")
    dialect_frame.to_csv(paths[1], index=False, float_format="%.12f")
    class_frame.to_csv(paths[2], index=False, float_format="%.12f")
    paths.extend(
        write_m0_m1_comparisons(PROJECT, dialect_frame, class_frame, output)
    )
    counts = {
        variant: int((main_frame.variant == variant).sum()) for variant in VARIANTS
    }
    report = {
        "status": "PARTIAL_VALIDATION_ONLY"
        if len(main_frame) < len(VARIANTS) * len(SEEDS)
        else "COMPLETE_VALIDATION_ONLY",
        "protocol_id": "boichitro_q1_v1",
        "test_data_access": False,
        "completed_runs": int(len(main_frame)),
        "expected_runs": len(VARIANTS) * len(SEEDS),
        "completed_by_variant": counts,
        "outputs": {
            str(path.relative_to(PROJECT)): sha256_file(path) for path in paths
        },
    }
    report_path = output / "development_results_snapshot.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
