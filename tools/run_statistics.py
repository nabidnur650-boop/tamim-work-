#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sacrebleu.metrics import CHRF
from sklearn.metrics import f1_score

PROJECT = Path(__file__).resolve().parents[1]
CHRFPP = CHRF(char_order=6, word_order=2, beta=2)
STD_ID = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run registered paired hierarchical bootstraps.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/statistical_analysis.yaml"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2903, 4307])
    parser.add_argument("--suffix", default="base")
    return parser.parse_args()


def prediction_path(
    protocol: str,
    variant: str,
    suffix: str,
    seed: int,
    task: str,
    track: str,
    prediction_prefix: str | None = None,
) -> Path:
    prefix = prediction_prefix or task
    return (
        PROJECT
        / "predictions"
        / protocol
        / f"{variant}__{suffix}"
        / str(seed)
        / f"{prefix}_{track}.parquet"
    )


def macro_chrf(frame: pd.DataFrame, prediction_column: str) -> float:
    values = []
    statistics_column = f"_chrf_stats_{prediction_column.removeprefix('prediction_')}"
    for _, group in frame.groupby("dialect", sort=True, observed=True):
        if statistics_column in group:
            statistics = np.stack(group[statistics_column].to_numpy()).sum(axis=0)
            values.append(CHRFPP._compute_score_from_stats(statistics.tolist()).score)
        else:
            values.append(
                CHRFPP.corpus_score(
                    group[prediction_column].astype(str).tolist(),
                    [group["reference"].astype(str).tolist()],
                ).score
            )
    return float(np.mean(values))


def chrf_segment_statistics(hypothesis: str, reference: str) -> np.ndarray:
    processed_hypothesis = CHRFPP._preprocess_segment(str(hypothesis))
    processed_reference = CHRFPP._preprocess_segment(str(reference))
    reference_info = CHRFPP._extract_reference_info([processed_reference])
    return np.asarray(
        CHRFPP._compute_segment_statistics(processed_hypothesis, reference_info),
        dtype=np.int64,
    )


def regional_f1(frame: pd.DataFrame, prediction_column: str) -> float:
    labels = frame["label_id"].to_numpy(dtype=np.int64)
    predictions = frame[prediction_column].to_numpy(dtype=np.int64)
    present = sorted(set(labels.tolist()) - {STD_ID})
    return float(
        f1_score(labels, predictions, labels=present, average="macro", zero_division=0)
    )


def align_pair(treatment: pd.DataFrame, control: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "normalization":
        columns = ["row_id", "semantic_group_id", "dialect", "reference", "prediction"]
        merged = treatment[columns].merge(
            control[columns], on="row_id", suffixes=("_treatment", "_control"), validate="one_to_one"
        )
        for column in ("semantic_group_id", "dialect", "reference"):
            if not merged[f"{column}_treatment"].equals(merged[f"{column}_control"]):
                raise ValueError(f"Paired normalization mismatch in {column}")
        aligned = pd.DataFrame(
            {
                "row_id": merged["row_id"],
                "semantic_group_id": merged["semantic_group_id_treatment"],
                "dialect": merged["dialect_treatment"],
                "reference": merged["reference_treatment"],
                "prediction_treatment": merged["prediction_treatment"],
                "prediction_control": merged["prediction_control"],
            }
        )
        aligned["_chrf_stats_treatment"] = [
            chrf_segment_statistics(prediction, reference)
            for prediction, reference in zip(
                aligned["prediction_treatment"], aligned["reference"]
            )
        ]
        aligned["_chrf_stats_control"] = [
            chrf_segment_statistics(prediction, reference)
            for prediction, reference in zip(
                aligned["prediction_control"], aligned["reference"]
            )
        ]
        return aligned
    columns = ["row_id", "semantic_group_id", "dialect", "label_id", "prediction_id"]
    merged = treatment[columns].merge(
        control[columns], on="row_id", suffixes=("_treatment", "_control"), validate="one_to_one"
    )
    for column in ("semantic_group_id", "dialect", "label_id"):
        if not merged[f"{column}_treatment"].equals(merged[f"{column}_control"]):
            raise ValueError(f"Paired identification mismatch in {column}")
    return pd.DataFrame(
        {
            "row_id": merged["row_id"],
            "semantic_group_id": merged["semantic_group_id_treatment"],
            "dialect": merged["dialect_treatment"],
            "label_id": merged["label_id_treatment"],
            "prediction_treatment": merged["prediction_id_treatment"],
            "prediction_control": merged["prediction_id_control"],
        }
    )


def resample_stratified_groups(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pieces = []
    for _, dialect_frame in frame.groupby("dialect", sort=True, observed=True):
        groups = dialect_frame["semantic_group_id"].astype(str).unique()
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indexed = {key: group for key, group in dialect_frame.groupby("semantic_group_id")}
        pieces.extend(indexed[group] for group in sampled)
    return pd.concat(pieces, ignore_index=True)


def _confusion_counts(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((13, 13), dtype=np.int64)
    np.add.at(matrix, (labels.astype(np.int64), predictions.astype(np.int64)), 1)
    return matrix


def _regional_f1_from_confusion(matrix: np.ndarray) -> float:
    values = []
    support = matrix.sum(axis=1)
    for label in range(matrix.shape[0]):
        if label == STD_ID or support[label] == 0:
            continue
        true_positive = matrix[label, label]
        false_positive = matrix[:, label].sum() - true_positive
        false_negative = matrix[label, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(values))


def prepare_group_statistics(frame: pd.DataFrame, task: str) -> dict[str, Any]:
    dialect_strata = []
    for dialect, dialect_frame in frame.groupby("dialect", sort=True, observed=True):
        group_ids = []
        treatment_groups = []
        control_groups = []
        for group_id, group in dialect_frame.groupby(
            "semantic_group_id", sort=True, observed=True
        ):
            group_ids.append(str(group_id))
            if task == "normalization":
                treatment_groups.append(
                    np.stack(group["_chrf_stats_treatment"].to_numpy()).sum(axis=0)
                )
                control_groups.append(
                    np.stack(group["_chrf_stats_control"].to_numpy()).sum(axis=0)
                )
            else:
                labels = group["label_id"].to_numpy(dtype=np.int64)
                treatment_groups.append(
                    _confusion_counts(
                        labels, group["prediction_treatment"].to_numpy(dtype=np.int64)
                    )
                )
                control_groups.append(
                    _confusion_counts(
                        labels, group["prediction_control"].to_numpy(dtype=np.int64)
                    )
                )
        dialect_strata.append(
            {
                "dialect": str(dialect),
                "group_ids": np.asarray(group_ids, dtype=object),
                "group_index": {group_id: index for index, group_id in enumerate(group_ids)},
                "treatment": np.stack(treatment_groups),
                "control": np.stack(control_groups),
            }
        )
    return {"task": task, "dialect_strata": dialect_strata}


def shared_group_contract(
    prepared_by_run: dict[int, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Require identical semantic-group strata for repeated model runs."""

    if not prepared_by_run:
        raise ValueError("At least one paired model run is required")
    first_key = sorted(prepared_by_run)[0]
    first = prepared_by_run[first_key]
    shared = {
        str(stratum["dialect"]): np.asarray(stratum["group_ids"], dtype=object)
        for stratum in first["dialect_strata"]
    }
    for run_key, prepared in prepared_by_run.items():
        if prepared["task"] != first["task"]:
            raise ValueError("Paired model runs disagree on task")
        observed = {
            str(stratum["dialect"]): np.asarray(stratum["group_ids"], dtype=object)
            for stratum in prepared["dialect_strata"]
        }
        if set(observed) != set(shared):
            raise ValueError(f"Paired model run {run_key} has different dialect strata")
        for dialect, group_ids in shared.items():
            if not np.array_equal(observed[dialect], group_ids):
                raise ValueError(
                    f"Paired model run {run_key} has different semantic groups for {dialect}"
                )
    return shared


def synchronized_bootstrap_groups(
    shared_groups: dict[str, np.ndarray], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Sample global semantic groups once and reuse multiplicities everywhere."""

    indices = synchronized_bootstrap_indices(shared_groups, rng)
    return {
        dialect: groups[indices[dialect]]
        for dialect, groups in shared_groups.items()
    }


def synchronized_bootstrap_indices(
    shared_groups: dict[str, np.ndarray], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Return local indices carrying synchronized global-group multiplicities."""

    global_groups = np.unique(np.concatenate(list(shared_groups.values())))
    sampled = rng.choice(global_groups, size=len(global_groups), replace=True)
    values, counts = np.unique(sampled, return_counts=True)
    multiplicity = {
        str(group): int(count) for group, count in zip(values, counts, strict=True)
    }
    return {
        dialect: np.repeat(
            np.arange(len(groups), dtype=np.int64),
            np.asarray([multiplicity.get(str(group), 0) for group in groups]),
        )
        for dialect, groups in shared_groups.items()
    }


def synchronized_randomization_swaps(
    shared_groups: dict[str, np.ndarray], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Draw one treatment/control swap per global group across runs/dialects."""

    global_groups = np.unique(np.concatenate(list(shared_groups.values())))
    values = rng.integers(0, 2, size=len(global_groups), dtype=np.int8).astype(bool)
    swap_by_group = {
        str(group): bool(swap)
        for group, swap in zip(global_groups, values, strict=True)
    }
    return {
        dialect: np.asarray(
            [swap_by_group[str(group)] for group in groups], dtype=bool
        )
        for dialect, groups in shared_groups.items()
    }


def score_group_statistics(
    prepared: dict[str, Any],
    rng: np.random.Generator | None = None,
    sampled_groups: dict[str, np.ndarray] | None = None,
    sampled_indices: dict[str, np.ndarray] | None = None,
) -> tuple[float, float]:
    if sum(value is not None for value in (rng, sampled_groups, sampled_indices)) > 1:
        raise ValueError("Specify only one group-resampling input")
    treatment_aggregates = []
    control_aggregates = []
    for stratum in prepared["dialect_strata"]:
        treatment_groups = stratum["treatment"]
        control_groups = stratum["control"]
        if sampled_indices is not None:
            dialect = str(stratum["dialect"])
            if dialect not in sampled_indices:
                raise ValueError(f"Missing synchronized bootstrap indices for {dialect}")
            indices = np.asarray(sampled_indices[dialect], dtype=np.int64)
        elif sampled_groups is not None:
            dialect = str(stratum["dialect"])
            if dialect not in sampled_groups:
                raise ValueError(f"Missing synchronized bootstrap groups for {dialect}")
            indices = np.asarray(
                [stratum["group_index"][str(group)] for group in sampled_groups[dialect]],
                dtype=np.int64,
            )
        elif rng is None:
            indices = np.arange(len(treatment_groups))
        else:
            indices = rng.integers(0, len(treatment_groups), size=len(treatment_groups))
        treatment_aggregates.append(treatment_groups[indices].sum(axis=0))
        control_aggregates.append(control_groups[indices].sum(axis=0))
    return score_aggregates(
        prepared["task"], treatment_aggregates, control_aggregates
    )


def score_aggregates(
    task: str,
    treatment_aggregates: list[np.ndarray],
    control_aggregates: list[np.ndarray],
) -> tuple[float, float]:
    if task == "normalization":
        treatment = float(
            np.mean(
                [
                    CHRFPP._compute_score_from_stats(value.tolist()).score
                    for value in treatment_aggregates
                ]
            )
        )
        control = float(
            np.mean(
                [
                    CHRFPP._compute_score_from_stats(value.tolist()).score
                    for value in control_aggregates
                ]
            )
        )
    else:
        treatment = _regional_f1_from_confusion(np.stack(treatment_aggregates).sum(axis=0))
        control = _regional_f1_from_confusion(np.stack(control_aggregates).sum(axis=0))
    return treatment, control


def score_randomized_group_statistics(
    prepared: dict[str, Any],
    rng: np.random.Generator | None = None,
    swaps_by_dialect: dict[str, np.ndarray] | None = None,
) -> tuple[float, float]:
    """Exchange paired systems at each semantic group, optionally across runs."""

    if rng is not None and swaps_by_dialect is not None:
        raise ValueError("Specify either rng or swaps_by_dialect, not both")
    if rng is None and swaps_by_dialect is None:
        raise ValueError("Randomized scoring requires swaps")

    treatment_aggregates = []
    control_aggregates = []
    for stratum in prepared["dialect_strata"]:
        treatment_groups = stratum["treatment"]
        control_groups = stratum["control"]
        dialect = str(stratum["dialect"])
        swap = (
            np.asarray(swaps_by_dialect[dialect], dtype=bool)
            if swaps_by_dialect is not None
            else rng.integers(  # type: ignore[union-attr]
                0, 2, size=len(treatment_groups), dtype=np.int8
            ).astype(bool)
        )
        if swap.shape != (len(treatment_groups),):
            raise ValueError(f"Invalid synchronized randomization swaps for {dialect}")
        mask = swap.reshape((len(swap),) + (1,) * (treatment_groups.ndim - 1))
        treatment_aggregates.append(
            np.where(mask, control_groups, treatment_groups).sum(axis=0)
        )
        control_aggregates.append(
            np.where(mask, treatment_groups, control_groups).sum(axis=0)
        )
    return score_aggregates(
        prepared["task"], treatment_aggregates, control_aggregates
    )


def paired_bootstrap(
    frames: dict[int, pd.DataFrame],
    *,
    task: str,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    seeds = np.asarray(sorted(frames), dtype=np.int64)
    prepared = {
        run_seed: prepare_group_statistics(frame, task)
        for run_seed, frame in frames.items()
    }
    shared_groups = shared_group_contract(prepared)
    seed_points = []
    for run_seed in seeds:
        treatment, control = score_group_statistics(prepared[int(run_seed)])
        seed_points.append(
            {
                "seed": int(run_seed),
                "treatment": treatment,
                "control": control,
                "delta": treatment - control,
            }
        )
    draws = []
    for replicate in range(replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_indices = synchronized_bootstrap_indices(shared_groups, rng)
        deltas = []
        treatment_values = []
        control_values = []
        for run_seed in sampled_seeds:
            treatment, control = score_group_statistics(
                prepared[int(run_seed)], sampled_indices=sampled_indices
            )
            treatment_values.append(treatment)
            control_values.append(control)
            deltas.append(treatment - control)
        draws.append(
            {
                "replicate": replicate,
                "treatment": float(np.mean(treatment_values)),
                "control": float(np.mean(control_values)),
                "delta": float(np.mean(deltas)),
            }
        )
    return pd.DataFrame(draws), seed_points


def paired_randomization(
    frames: dict[int, pd.DataFrame],
    *,
    task: str,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    """Build a paired null with one synchronized swap per global semantic group."""

    rng = np.random.default_rng(seed)
    prepared = {
        run_seed: prepare_group_statistics(frame, task)
        for run_seed, frame in frames.items()
    }
    shared_groups = shared_group_contract(prepared)
    observed_deltas = []
    for run_seed in sorted(prepared):
        treatment, control = score_group_statistics(prepared[run_seed])
        observed_deltas.append(treatment - control)
    observed = float(np.mean(observed_deltas))
    draws = []
    for replicate in range(replicates):
        synchronized_swaps = synchronized_randomization_swaps(shared_groups, rng)
        deltas = []
        for run_seed in sorted(prepared):
            treatment, control = score_randomized_group_statistics(
                prepared[run_seed], swaps_by_dialect=synchronized_swaps
            )
            deltas.append(treatment - control)
        draws.append(
            {
                "replicate": replicate,
                "null_delta": float(np.mean(deltas)),
            }
        )
    return pd.DataFrame(draws), observed


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("resampling_unit") != "global_semantic_group":
        raise ValueError("Statistics config must use global semantic-group resampling")
    if config.get("repeated_row_dependence") != (
        "synchronized_across_seeds_architectures_and_dialects"
    ):
        raise ValueError("Statistics config must preserve repeated-row dependence")
    output_dir = PROJECT / "reports/statistics" / str(config["protocol_id"])
    if config.get("output_subdir"):
        output_dir = output_dir / str(config["output_subdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for comparison_index, comparison in enumerate(config["confirmatory_comparisons"]):
        frames = {}
        for seed in args.seeds:
            treatment = pd.read_parquet(
                prediction_path(
                    config["protocol_id"],
                    comparison["treatment"],
                    comparison.get("treatment_suffix", args.suffix),
                    seed,
                    comparison["task"],
                    comparison["track"],
                    comparison.get("treatment_prediction_prefix"),
                )
            )
            control = pd.read_parquet(
                prediction_path(
                    config["protocol_id"],
                    comparison["control"],
                    comparison.get("control_suffix", args.suffix),
                    seed,
                    comparison["task"],
                    comparison["track"],
                    comparison.get("control_prediction_prefix"),
                )
            )
            frames[seed] = align_pair(treatment, control, comparison["task"])
        draws, seed_points = paired_bootstrap(
            frames,
            task=comparison["task"],
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["bootstrap_seed"]) + comparison_index,
        )
        randomization, observed_delta = paired_randomization(
            frames,
            task=comparison["task"],
            replicates=int(config["randomization_replicates"]),
            seed=int(config["randomization_seed"]) + comparison_index,
        )
        alpha = 1.0 - float(config["confidence_level"])
        lower, upper = np.quantile(draws["delta"], [alpha / 2.0, 1.0 - alpha / 2.0])
        p_one_sided = (
            int(randomization["null_delta"].ge(observed_delta).sum()) + 1
        ) / (len(randomization) + 1)
        p_two_sided = (
            int(
                randomization["null_delta"]
                .abs()
                .ge(abs(observed_delta))
                .sum()
            )
            + 1
        ) / (len(randomization) + 1)
        seed_deltas = np.asarray([row["delta"] for row in seed_points])
        summary = {
            **comparison,
            "multiplicity_family": comparison.get("family", "confirmatory"),
            "seeds": args.seeds,
            "seed_points": seed_points,
            "mean_delta_across_seeds": float(seed_deltas.mean()),
            "seed_standard_deviation": float(seed_deltas.std(ddof=1)),
            "bootstrap_mean_delta": float(draws["delta"].mean()),
            "confidence_lower": float(lower),
            "confidence_upper": float(upper),
            "bootstrap_probability_delta_positive": float(draws["delta"].gt(0).mean()),
            "randomization_observed_delta": observed_delta,
            "randomization_null_mean": float(randomization["null_delta"].mean()),
            "randomization_null_standard_deviation": float(
                randomization["null_delta"].std(ddof=1)
            ),
            "p_one_sided": float(p_one_sided),
            "p_two_sided": float(p_two_sided),
            "bootstrap_replicates": len(draws),
            "randomization_replicates": len(randomization),
        }
        summaries.append(summary)
        draws.to_parquet(output_dir / f"{comparison['id']}_bootstrap.parquet", index=False)
        randomization.to_parquet(
            output_dir / f"{comparison['id']}_randomization.parquet", index=False
        )
    families = sorted({row["multiplicity_family"] for row in summaries})
    for family in families:
        indices = [
            index
            for index, row in enumerate(summaries)
            if row["multiplicity_family"] == family
        ]
        adjusted_one_sided = holm_adjust(
            [summaries[index]["p_one_sided"] for index in indices]
        )
        adjusted_two_sided = holm_adjust(
            [summaries[index]["p_two_sided"] for index in indices]
        )
        for index, one_sided, two_sided in zip(
            indices, adjusted_one_sided, adjusted_two_sided
        ):
            summaries[index]["p_holm_one_sided"] = one_sided
            summaries[index]["p_holm_two_sided"] = two_sided
    (output_dir / "confirmatory_statistics.json").write_text(
        json.dumps(
            {
                "protocol_id": config["protocol_id"],
                "confidence_level": config["confidence_level"],
                "multiple_testing": config["multiple_testing"],
                "primary_inference": config.get("primary_inference"),
                "interval_method": (
                    "paired hierarchical bootstrap over model runs with semantic-group "
                    "draws synchronized across repeated evaluation rows and cross-dialect "
                    "realizations"
                ),
                "p_value_method": (
                    "paired randomization at the global semantic-group level with swaps "
                    "synchronized across model runs and cross-dialect realizations"
                ),
                "repeated_row_dependence_preserved": True,
                "comparisons": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "seed_points"}
            for row in summaries
        ]
    ).to_csv(output_dir / "confirmatory_statistics.csv", index=False)
    print(f"Statistical report written to {output_dir}")


if __name__ == "__main__":
    main()
