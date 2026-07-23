#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed blinded native ratings.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/human_evaluation.yaml"
    )
    return parser.parse_args()


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_sign_randomization(
    paired_delta: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
    higher_is_better: bool,
) -> tuple[np.ndarray, float, float]:
    values = np.asarray(paired_delta, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Paired randomization requires a non-empty 1-D delta array")
    observed_delta = float(values.mean())
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(replicates, len(values)),
        replace=True,
    )
    null_delta = (signs * values[None, :]).mean(axis=1)
    p_one_sided = (
        (int(np.count_nonzero(null_delta >= observed_delta)) + 1)
        if higher_is_better
        else (int(np.count_nonzero(null_delta <= observed_delta)) + 1)
    ) / (len(null_delta) + 1)
    p_two_sided = (
        int(np.count_nonzero(np.abs(null_delta) >= abs(observed_delta))) + 1
    ) / (len(null_delta) + 1)
    return null_delta, float(p_one_sided), float(p_two_sided)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = PROJECT / "human_evaluation" / str(config["protocol_id"])
    master = pd.read_csv(root / "BLINDING_KEY_DO_NOT_SHARE_WITH_RATERS.csv")
    packets = []
    for path in sorted(root.glob("rater_packet_*.csv")):
        frame = pd.read_csv(path)
        packets.append(frame)
    if len(packets) != int(config["raters"]):
        raise ValueError(f"Expected {config['raters']} rater packets, found {len(packets)}")
    ratings = pd.concat(packets, ignore_index=True)
    dimensions = list(config["rating_dimensions"])
    binary = list(config["binary_dimensions"])
    analysis_dimensions = dimensions + binary
    for dimension, limits in config["rating_dimensions"].items():
        ratings[dimension] = pd.to_numeric(ratings[dimension], errors="raise")
        if not ratings[dimension].between(limits["minimum"], limits["maximum"]).all():
            raise ValueError(f"Ratings outside bounds for {dimension}")
    for dimension in binary:
        ratings[dimension] = pd.to_numeric(ratings[dimension], errors="raise").astype(int)
        if not ratings[dimension].isin([0, 1]).all():
            raise ValueError(f"Binary ratings outside 0/1 for {dimension}")
    counts = ratings["blind_item_id"].value_counts()
    if not counts.eq(int(config["ratings_per_item"])).all():
        raise ValueError("Every blind item must have the registered number of ratings")
    merged = ratings.merge(master, on="blind_item_id", validate="many_to_one")
    merged.to_parquet(root / "completed_ratings_unblinded.parquet", index=False)

    agreement = []
    for dimension in analysis_dimensions:
        pair_values = []
        for first, second in combinations(sorted(merged["rater_id"].unique()), 2):
            a = merged.loc[merged["rater_id"].eq(first), ["blind_item_id", dimension]]
            b = merged.loc[merged["rater_id"].eq(second), ["blind_item_id", dimension]]
            pair = a.merge(b, on="blind_item_id", suffixes=("_a", "_b"))
            if len(pair) >= 2:
                pair_values.append(
                    cohen_kappa_score(
                        pair[f"{dimension}_a"],
                        pair[f"{dimension}_b"],
                        weights="quadratic" if dimension in dimensions else None,
                    )
                )
        agreement.append(
            {
                "dimension": dimension,
                "kappa_weighting": "quadratic" if dimension in dimensions else "unweighted",
                "mean_pairwise_kappa": float(np.nanmean(pair_values)),
                "rater_pairs": len(pair_values),
            }
        )

    rng = np.random.default_rng(int(config["sampling_seed"]) + 9_001)
    row_ids = merged["row_id"].astype(str).unique()
    systems = list(config["systems"])
    draws = []
    for replicate in range(int(config["bootstrap_replicates"])):
        sampled_ids = rng.choice(row_ids, size=len(row_ids), replace=True)
        sampled = pd.concat(
            [merged.loc[merged["row_id"].astype(str).eq(row_id)] for row_id in sampled_ids],
            ignore_index=True,
        )
        for dimension in analysis_dimensions:
            means = sampled.groupby("system")[dimension].mean()
            for system in systems:
                draws.append(
                    {
                        "replicate": replicate,
                        "dimension": dimension,
                        "system": system,
                        "mean": float(means[system]),
                    }
                )
    draw_frame = pd.DataFrame(draws)
    draw_frame.to_parquet(root / "human_rating_bootstrap.parquet", index=False)
    point = merged.groupby("system", as_index=False)[analysis_dimensions].mean()
    summaries = []
    for dimension in analysis_dimensions:
        for system in systems:
            values = draw_frame.loc[
                draw_frame["dimension"].eq(dimension) & draw_frame["system"].eq(system),
                "mean",
            ]
            summaries.append(
                {
                    "dimension": dimension,
                    "system": system,
                    "mean": float(point.loc[point["system"].eq(system), dimension].iloc[0]),
                    "confidence_lower": float(values.quantile(0.025)),
                    "confidence_upper": float(values.quantile(0.975)),
                }
            )
    comparisons = []
    randomization_rows = []
    randomization_rng = np.random.default_rng(int(config["randomization_seed"]))
    item_means = merged.groupby(["row_id", "system"], as_index=False)[
        analysis_dimensions
    ].mean()
    for dimension in analysis_dimensions:
        pivot = draw_frame.loc[draw_frame["dimension"].eq(dimension)].pivot(
            index="replicate", columns="system", values="mean"
        )
        paired_items = item_means.pivot(index="row_id", columns="system", values=dimension)
        for control in ("M2", "BANGLAT5_SMALL"):
            delta = pivot["M3"] - pivot[control]
            paired_delta = (paired_items["M3"] - paired_items[control]).dropna().to_numpy()
            observed_delta = float(paired_delta.mean())
            higher_is_better = dimension not in binary
            null_delta, p_one_sided, p_two_sided = paired_sign_randomization(
                paired_delta,
                replicates=int(config["randomization_replicates"]),
                rng=randomization_rng,
                higher_is_better=higher_is_better,
            )
            comparison_id = f"M3_vs_{control}_{dimension}"
            randomization_rows.extend(
                {
                    "comparison_id": comparison_id,
                    "replicate": replicate,
                    "null_delta": float(value),
                }
                for replicate, value in enumerate(null_delta)
            )
            comparisons.append(
                {
                    "id": comparison_id,
                    "dimension": dimension,
                    "treatment": "M3",
                    "control": control,
                    "preferred_direction": "higher" if higher_is_better else "lower",
                    "observed_paired_item_delta": observed_delta,
                    "bootstrap_mean_delta": float(delta.mean()),
                    "confidence_lower": float(delta.quantile(0.025)),
                    "confidence_upper": float(delta.quantile(0.975)),
                    "p_one_sided": float(p_one_sided),
                    "p_two_sided": float(p_two_sided),
                }
            )
    pd.DataFrame(randomization_rows).to_parquet(
        root / "human_rating_randomization.parquet", index=False
    )
    adjusted_one = holm_adjust([row["p_one_sided"] for row in comparisons])
    adjusted_two = holm_adjust([row["p_two_sided"] for row in comparisons])
    for row, one_sided, two_sided in zip(comparisons, adjusted_one, adjusted_two):
        row["p_holm_one_sided"] = one_sided
        row["p_holm_two_sided"] = two_sided
    report = {
        "status": "COMPLETE",
        "protocol_id": config["protocol_id"],
        "ratings": len(merged),
        "unique_blind_items": merged["blind_item_id"].nunique(),
        "interval_method": "paired item-cluster bootstrap",
        "p_value_method": "paired item-level sign randomization",
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "randomization_replicates": int(config["randomization_replicates"]),
        "agreement": agreement,
        "system_summaries": summaries,
        "confirmatory_comparisons": comparisons,
    }
    (root / "human_evaluation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(root / "human_evaluation_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(root / "human_evaluation_comparisons.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
