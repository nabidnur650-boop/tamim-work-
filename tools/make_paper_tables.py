#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready tables and source CSVs.")
    parser.add_argument("--protocol", default="locked_test_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2903, 4307])
    return parser.parse_args()


def write_table(frame: pd.DataFrame, root: Path, name: str) -> str:
    frame.to_csv(root / f"{name}.csv", index=False)
    (root / f"{name}.tex").write_text(
        frame.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.4f}"),
        encoding="utf-8",
    )
    return f"## {name.replace('_', ' ').title()}\n\n{frame.to_markdown(index=False)}\n"


def main_results(protocol: str, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for variant in ("M0", "M1", "M2", "M3"):
        for seed in seeds:
            path = (
                PROJECT
                / "predictions"
                / protocol
                / f"{variant}__base"
                / str(seed)
                / "evaluation_manifest.json"
            )
            if not path.exists():
                return pd.DataFrame()
            payload = json.loads(path.read_text(encoding="utf-8"))
            for task, manifest_key, view, metrics_to_report in (
                (
                    "normalization",
                    "normalization",
                    "raw_neural",
                    ("macro_chrfpp", "worst_dialect_chrfpp", "corpus_sacrebleu", "cer", "wer"),
                ),
                (
                    "normalization",
                    "normalization_fused",
                    "source_blind_fused",
                    ("macro_chrfpp", "worst_dialect_chrfpp", "corpus_sacrebleu", "cer", "wer"),
                ),
                (
                    "identification",
                    "identification",
                    "raw_neural",
                    ("regional_macro_f1", "macro_f1_13", "balanced_accuracy", "mcc", "ece_15", "brier"),
                ),
                (
                    "identification",
                    "identification_fused",
                    "source_blind_fused",
                    ("regional_macro_f1", "macro_f1_13", "balanced_accuracy", "mcc", "ece_15", "brier"),
                ),
            ):
                for track, metrics in payload.get(manifest_key, {}).items():
                    for metric in metrics_to_report:
                        rows.append(
                            {
                                "variant": variant,
                                "seed": seed,
                                "task": task,
                                "system_view": view,
                                "track": track,
                                "metric": metric,
                                "value": metrics[metric],
                            }
                        )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["variant", "task", "system_view", "track", "metric"])["value"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    summary["mean_sd"] = summary.apply(lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1)
    return summary


def external_results(seeds: list[int]) -> pd.DataFrame:
    rows = []
    mapping = {
        "normalization": ("BANGLAT5_SMALL", "MT5_SMALL"),
        "identification": ("BANGLABERT_MIT", "XLMR_BASE"),
    }
    for task, models in mapping.items():
        for model in models:
            for seed in seeds:
                path = PROJECT / "predictions/locked_external_test_v1" / task / model / str(seed) / "evaluation_manifest.json"
                if not path.exists():
                    return pd.DataFrame()
                payload = json.loads(path.read_text(encoding="utf-8"))
                metric = "macro_chrfpp" if task == "normalization" else "regional_macro_f1"
                for track, values in payload["results"].items():
                    rows.append(
                        {"task": task, "model": model, "seed": seed, "track": track, "metric": metric, "value": values[metric]}
                    )
    frame = pd.DataFrame(rows)
    result = (
        frame.groupby(["task", "model", "track", "metric"])["value"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    result["mean_sd"] = result.apply(lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1)
    return result


def training_compute_summary(seeds: list[int]) -> pd.DataFrame:
    """Aggregate only complete, registered foundation and main task runs."""

    rows = []
    foundation_runs = (
        "F_DENSE_300M",
        "U_M0_DENSE_200M",
        "U_M1_SWITCH_200M",
        "U_M2_STANDARD_MOE_200M",
        "P_M2_BANKED_20M",
        "P_M2_UNBANKED_20M",
        "P_M2_SCRATCH_20M",
        "P_M2_ANNEALED_20M",
        "P_M2_PAIRED_20M",
        "P_M1_LOSS_FREE_ROUTER_10M",
        "P_M1_AUX_ROUTER_10M",
    )
    required = [
        PROJECT / "runs" / run_id / "1701" / "training_report.json"
        for run_id in foundation_runs
    ]
    required.extend(
        PROJECT
        / "runs/task/boichitro_q1_v1"
        / f"{variant}__base"
        / str(seed)
        / f"stage_{stage}/training_report.json"
        for variant in ("M0", "M1", "M2", "M3")
        for seed in seeds
        for stage in ("a", "s", "id")
    )
    if not all(path.exists() for path in required):
        return pd.DataFrame()

    for run_id, path in zip(foundation_runs, required[: len(foundation_runs)]):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE_FIXED_BUDGET":
            return pd.DataFrame()
        parameters = payload["parameter_report"]
        rows.append(
            {
                "system": run_id,
                "stage": "foundation_or_continuation",
                "seed": 1701,
                "tokens_seen": int(payload["tokens_seen"]),
                "elapsed_hours": float(payload["elapsed_seconds"]) / 3600.0,
                "tokens_per_second": float(payload["tokens_per_second"]),
                "peak_memory_gib": float(payload["peak_memory_gib"]),
                "total_parameters": int(parameters["total_parameters"]),
                "active_parameters_per_token": int(
                    parameters["active_parameters_per_token"]
                ),
            }
        )

    task_paths = required[len(foundation_runs) :]
    path_index = 0
    for variant in ("M0", "M1", "M2", "M3"):
        for seed in seeds:
            for stage in ("a", "s", "id"):
                path = task_paths[path_index]
                path_index += 1
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "COMPLETE":
                    return pd.DataFrame()
                parameters = payload["parameter_report"]
                rows.append(
                    {
                        "system": variant,
                        "stage": f"task_{stage}",
                        "seed": seed,
                        "tokens_seen": int(payload["tokens_seen"]),
                        "elapsed_hours": float(payload["elapsed_seconds"]) / 3600.0,
                        "tokens_per_second": float(payload["tokens_per_second"]),
                        "peak_memory_gib": float(payload["peak_memory_gib"]),
                        "total_parameters": int(parameters["total_parameters"]),
                        "active_parameters_per_token": int(
                            parameters["active_parameters_per_token"]
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(PROJECT / "tables/paper/training_compute_runs.csv", index=False)
    return (
        frame.groupby(["system", "stage"], sort=False)
        .agg(
            runs=("seed", "size"),
            tokens_seen_total=("tokens_seen", "sum"),
            gpu_hours_total=("elapsed_hours", "sum"),
            tokens_per_second_mean=("tokens_per_second", "mean"),
            tokens_per_second_std=("tokens_per_second", "std"),
            peak_memory_gib_max=("peak_memory_gib", "max"),
            total_parameters=("total_parameters", "max"),
            active_parameters_per_token=("active_parameters_per_token", "max"),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    root = PROJECT / "tables/paper"
    root.mkdir(parents=True, exist_ok=True)
    sections = []
    created = []

    dataset = json.loads((PROJECT / "reports/final_dataset_report.json").read_text(encoding="utf-8"))
    counts = pd.DataFrame(
        [{"quantity": key, "count": value} for key, value in dataset["counts"].items()]
    )
    sections.append(write_table(counts, root, "dataset_counts"))
    created.append("dataset_counts")

    normalization_all = pd.read_parquet(
        PROJECT / "data/final/v1/normalization_all.parquet",
        columns=["dialect", "split", "is_synthetic", "source_id"],
    )
    normalization_composition = (
        normalization_all.groupby(["dialect", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "validation", "test", "test_ood"], fill_value=0)
        .reset_index()
    )
    sections.append(
        write_table(normalization_composition, root, "normalization_dialect_splits")
    )
    created.append("normalization_dialect_splits")

    identification_all = pd.read_parquet(
        PROJECT / "data/final/v1/identification_all.parquet",
        columns=["dialect", "split", "source_id"],
    )
    identification_composition = (
        identification_all.groupby(["dialect", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=["train", "validation", "test", "test_ood", "test_external"],
            fill_value=0,
        )
        .reset_index()
    )
    sections.append(
        write_table(identification_composition, root, "identification_dialect_splits")
    )
    created.append("identification_dialect_splits")

    tokenizer = pd.read_csv(PROJECT / "reports/tokenizer/tokenizer_proxy_summary.csv")
    tokenizer = tokenizer[[
        "candidate_id", "mean_bpc", "std_bpc", "mean_worst_dialect_bpc", "tokens_per_character", "mean_tokens_per_second", "selected"
    ]]
    sections.append(write_table(tokenizer, root, "tokenizer_selection"))
    created.append("tokenizer_selection")

    systems = pd.read_csv(PROJECT / "reports/model/gb10_model_benchmark.csv")
    systems = systems[[
        "model_id", "total_parameters", "active_parameters_per_token", "tokens_per_second", "peak_memory_gib"
    ]]
    sections.append(write_table(systems, root, "model_systems"))
    created.append("model_systems")

    compute = training_compute_summary(args.seeds)
    if not compute.empty:
        sections.append(write_table(compute, root, "training_compute_summary"))
        created.append("training_compute_summary")

    optimizer_pilot_path = PROJECT / "reports/model/optimizer_pilot_selection.json"
    if optimizer_pilot_path.exists():
        optimizer_pilot = json.loads(optimizer_pilot_path.read_text(encoding="utf-8"))
        if optimizer_pilot.get("status") == "COMPLETE_VALIDATION_ONLY":
            optimizer_rows = pd.DataFrame(optimizer_pilot["candidates"])
            optimizer_rows["selected"] = optimizer_rows["adamw_learning_rate"].eq(
                float(optimizer_pilot["selected_adamw_learning_rate"])
            )
            sections.append(
                write_table(optimizer_rows, root, "adamw_learning_rate_pilot")
            )
            created.append("adamw_learning_rate_pilot")

    continuation_pilot_path = (
        PROJECT / "reports/model/continuation_lr_pilot_selection.json"
    )
    if continuation_pilot_path.exists():
        continuation_pilot = json.loads(
            continuation_pilot_path.read_text(encoding="utf-8")
        )
        if continuation_pilot.get("status") == "COMPLETE_VALIDATION_ONLY":
            continuation_rows = pd.DataFrame(continuation_pilot["candidates"])
            continuation_rows["selected"] = continuation_rows["id"].eq(
                continuation_pilot["selected_candidate"]
            )
            sections.append(
                write_table(
                    continuation_rows,
                    root,
                    "continuation_learning_rate_pilot",
                )
            )
            created.append("continuation_learning_rate_pilot")

    retention_pilot_path = (
        PROJECT / "reports/model/stage_s_retention_pilot_selection.json"
    )
    if retention_pilot_path.exists():
        retention = json.loads(retention_pilot_path.read_text(encoding="utf-8"))
        if retention.get("status") == "COMPLETE_VALIDATION_ONLY":
            rows = []
            for candidate in retention["candidates"]:
                registered = candidate["candidate"] or {}
                curve = candidate["validation_curve"]
                selected_validation = candidate["selected_validation"]
                display_validation = selected_validation or sorted(
                    curve, key=lambda row: -float(row["macro_chrfpp"])
                )[0]
                mixture = registered.get("mixture", {})
                optimizer = registered.get("optimizer", {})
                rows.append(
                    {
                        "candidate": candidate["candidate_id"],
                        "normalization_fraction": mixture.get("normalization", 0.55),
                        "replay_fraction": mixture.get("general_replay", 0.10),
                        "muon_lr": optimizer.get("muon_lr", 0.006),
                        "adamw_lr": optimizer.get("adamw_lr", 0.0001),
                        "selected_step": int(display_validation["optimizer_step"]),
                        "macro_chrfpp": float(display_validation["macro_chrfpp"]),
                        "worst_dialect_chrfpp": float(
                            display_validation["worst_dialect_chrfpp"]
                        ),
                        "replay_degradation_percent": 100.0
                        * float(display_validation["replay_relative_degradation"]),
                        "eligible_checkpoints": int(
                            candidate["eligible_checkpoint_count"]
                        ),
                        "selected": candidate["candidate_id"]
                        == retention["selected_candidate_id"],
                    }
                )
            retention_rows = pd.DataFrame(rows)
            sections.append(
                write_table(retention_rows, root, "stage_s_retention_pilot")
            )
            created.append("stage_s_retention_pilot")

    upcycling_path = PROJECT / "reports/model/upcycling_strategy_selection.json"
    if upcycling_path.exists():
        upcycling = json.loads(upcycling_path.read_text(encoding="utf-8"))
        if upcycling.get("status") == "COMPLETE_VALIDATION_ONLY":
            upcycling_rows = pd.DataFrame(
                [
                    {
                        "strategy": strategy,
                        "run_id": values["run_id"],
                        "initial_validation_bpc": values["initial_validation_bpc"],
                        "final_validation_bpc": values["final_validation_bpc"],
                        "maximum_regression_percent": values[
                            "maximum_transient_change_from_own_initial_percent"
                        ],
                        "eligible": values["eligible"],
                        "selected": strategy == upcycling["selected_strategy"],
                    }
                    for strategy, values in upcycling["conditions"].items()
                ]
            )
            sections.append(
                write_table(upcycling_rows, root, "upcycling_strategy_pilot")
            )
            created.append("upcycling_strategy_pilot")

    switch_path = PROJECT / "reports/model/switch_router_selection.json"
    if switch_path.exists():
        switch = json.loads(switch_path.read_text(encoding="utf-8"))
        if switch.get("status") == "COMPLETE_VALIDATION_ONLY":
            switch_rows = pd.DataFrame(
                [
                    {
                        "strategy": strategy,
                        "run_id": values["run_id"],
                        "initial_validation_bpc": values["initial_validation_bpc"],
                        "final_validation_bpc": values["final_validation_bpc"],
                        "maximum_router_load_cv": values[
                            "maximum_logged_router_load_cv"
                        ],
                        "final_router_load_cv": values[
                            "final_logged_router_load_cv"
                        ],
                        "eligible": values["eligible"],
                        "selected": strategy == switch["selected_strategy"],
                    }
                    for strategy, values in switch["candidates"].items()
                ]
            )
            sections.append(write_table(switch_rows, root, "switch_router_pilot"))
            created.append("switch_router_pilot")

    fusion_uncertainty_path = (
        PROJECT / "reports/model/development_fusion_uncertainty.json"
    )
    if fusion_uncertainty_path.exists():
        fusion = json.loads(fusion_uncertainty_path.read_text(encoding="utf-8"))
        if fusion.get("status") == "COMPLETE_VALIDATION_ONLY":
            fusion_rows = pd.DataFrame(
                [
                    {
                        "task": task,
                        "paired_mean_gain": values["mean_delta"],
                        "confidence_lower_95": values["confidence_lower_95"],
                        "confidence_upper_95": values["confidence_upper_95"],
                        "randomization_p_two_sided": values[
                            "paired_randomization_p_two_sided"
                        ],
                        "bootstrap_replicates": values["bootstrap_replicates"],
                        "randomization_replicates": values[
                            "randomization_replicates"
                        ],
                    }
                    for task, values in (
                        ("normalization", fusion["normalization"]),
                        ("identification", fusion["identification"]),
                    )
                ]
            )
            sections.append(
                write_table(fusion_rows, root, "development_fusion_uncertainty")
            )
            created.append("development_fusion_uncertainty")

    main = main_results(args.protocol, args.seeds)
    if not main.empty:
        sections.append(write_table(main, root, "main_locked_results"))
        created.append("main_locked_results")
    external = external_results(args.seeds)
    if not external.empty:
        sections.append(write_table(external, root, "external_baselines"))
        created.append("external_baselines")
    ablation_path = PROJECT / "figures/data/fig_ablation_deltas.csv"
    if ablation_path.exists():
        ablation = (
            pd.read_csv(ablation_path)
            .groupby(["label", "task"])["delta_from_m3"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        sections.append(write_table(ablation, root, "ablation_deltas"))
        created.append("ablation_deltas")
    robustness_path = PROJECT / "reports/robustness/locked_robustness_v1/robustness_summary.csv"
    if robustness_path.exists():
        robustness = (
            pd.read_csv(robustness_path)
            .groupby(["variant", "task", "system_view", "family"])
            .agg(
                robustness_auc_mean=("normalized_robustness_auc", "mean"),
                robustness_auc_std=("normalized_robustness_auc", "std"),
                maximum_relative_drop_mean=("maximum_relative_drop", "mean"),
                maximum_relative_drop_std=("maximum_relative_drop", "std"),
            )
            .reset_index()
        )
        sections.append(write_table(robustness, root, "robustness_summary"))
        created.append("robustness_summary")
    inference_path = PROJECT / "reports/model/task_inference_benchmark.csv"
    if inference_path.exists():
        inference = pd.read_csv(inference_path)
        sections.append(write_table(inference, root, "task_inference_efficiency"))
        created.append("task_inference_efficiency")

    inference_sources = (
        (
            "main",
            PROJECT / "reports/statistics" / args.protocol / "confirmatory_statistics.csv",
        ),
        (
            "bidirectional",
            PROJECT
            / "reports/statistics"
            / args.protocol
            / "bidirectional_specialization/confirmatory_statistics.csv",
        ),
        (
            "ablations",
            PROJECT
            / "reports/statistics"
            / args.protocol
            / "ablations/confirmatory_statistics.csv",
        ),
        (
            "system_fusion",
            PROJECT
            / "reports/statistics"
            / args.protocol
            / "source_blind_system_fusion/confirmatory_statistics.csv",
        ),
    )
    inference_frames = []
    for analysis_family, path in inference_sources:
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "analysis_family", analysis_family)
            inference_frames.append(frame)
    if len(inference_frames) == len(inference_sources):
        inference = pd.concat(inference_frames, ignore_index=True)
        inference = inference[
            [
                "analysis_family",
                "id",
                "task",
                "track",
                "multiplicity_family",
                "bootstrap_mean_delta",
                "confidence_lower",
                "confidence_upper",
                "p_two_sided",
                "p_holm_two_sided",
            ]
        ]
        sections.append(write_table(inference, root, "confirmatory_inference"))
        created.append("confirmatory_inference")

    human_root = PROJECT / "human_evaluation/blind_native_normalization_v1"
    human_summary_path = human_root / "human_evaluation_summary.csv"
    human_comparisons_path = human_root / "human_evaluation_comparisons.csv"
    if human_summary_path.exists() and human_comparisons_path.exists():
        sections.append(
            write_table(pd.read_csv(human_summary_path), root, "human_evaluation_summary")
        )
        sections.append(
            write_table(
                pd.read_csv(human_comparisons_path),
                root,
                "human_evaluation_comparisons",
            )
        )
        created.extend(("human_evaluation_summary", "human_evaluation_comparisons"))

    (root / "TABLES.md").write_text("# Boichitro paper tables\n\n" + "\n".join(sections), encoding="utf-8")
    manifest = {"status": "COMPLETE", "created": created, "protocol": args.protocol}
    (root / "table_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
