#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import jensenshannon

PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify permutation-invariant expert specialization from locked traces."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/routing_analysis.yaml"
    )
    return parser.parse_args()


def contingency(labels: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = sorted(set(labels.tolist()))
    mapping = {value: index for index, value in enumerate(values)}
    table = np.zeros((len(values), counts.shape[1]), dtype=np.float64)
    for row, label in zip(counts, labels):
        table[mapping[str(label)]] += row
    return table, values


def normalized_mutual_information(table: np.ndarray) -> tuple[float, float, float, float]:
    smoothed = table + 1e-12
    joint = smoothed / smoothed.sum()
    dialect = joint.sum(axis=1, keepdims=True)
    expert = joint.sum(axis=0, keepdims=True)
    mutual_information = float((joint * np.log(joint / (dialect @ expert))).sum())
    dialect_entropy = float(-(dialect * np.log(dialect)).sum())
    expert_entropy = float(-(expert * np.log(expert)).sum())
    denominator = math.sqrt(max(1e-12, dialect_entropy * expert_entropy))
    return mutual_information / denominator, mutual_information, dialect_entropy, expert_entropy


def holm_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted


def specialization_metrics(table: np.ndarray) -> dict[str, float]:
    normalized_mi, mutual_information, _, expert_entropy = normalized_mutual_information(table)
    smoothed = table + 1e-12
    distributions = smoothed / smoothed.sum(axis=1, keepdims=True)
    divergences = [
        float(jensenshannon(distributions[first], distributions[second], base=2.0) ** 2)
        for first, second in combinations(range(len(distributions)), 2)
    ]
    return {
        "mutual_information_nats": mutual_information,
        "normalized_mutual_information": normalized_mi,
        "mean_pairwise_dialect_js_divergence": float(np.mean(divergences))
        if divergences
        else 0.0,
        "expert_purity": float(table.max(axis=0).sum() / max(1e-12, table.sum())),
        "expert_load_cv": float(table.sum(axis=0).std() / max(1e-12, table.sum(axis=0).mean())),
        "normalized_expert_load_entropy": expert_entropy / math.log(table.shape[1]),
        "dialects_present": float(table.shape[0]),
        "assignments": float(table.sum()),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    task_distributions: dict[tuple[str, int, int, str], np.ndarray] = {}
    replicate_count = int(config["permutation_replicates"])
    for variant_index, variant in enumerate(config["variants"]):
        for seed_index, seed in enumerate(config["seeds"]):
            for task_index, task in enumerate(config["tasks"]):
                path = (
                    PROJECT
                    / "predictions"
                    / str(config["protocol_id"])
                    / f"{variant}__{config['suffix']}"
                    / str(seed)
                    / f"routing_{task}_{config['track']}.parquet"
                )
                frame = pd.read_parquet(path)
                for layer, layer_frame in frame.groupby("layer_index", sort=True):
                    labels = layer_frame["dialect"].astype(str).to_numpy()
                    source_labels = layer_frame["source_id"].astype(str).to_numpy()
                    counts = np.asarray(layer_frame["expert_counts"].tolist(), dtype=np.float64)
                    table, dialects = contingency(labels, counts)
                    observed = specialization_metrics(table)
                    source_table, sources = contingency(source_labels, counts)
                    source_nmi = normalized_mutual_information(source_table)[0]
                    conditional_values = []
                    conditional_weights = []
                    for source in sources:
                        mask = source_labels == source
                        if len(set(labels[mask].tolist())) < 2:
                            continue
                        conditional_table, _ = contingency(labels[mask], counts[mask])
                        conditional_values.append(
                            normalized_mutual_information(conditional_table)[0]
                        )
                        conditional_weights.append(float(counts[mask].sum()))
                    conditional_dialect_nmi = (
                        float(np.average(conditional_values, weights=conditional_weights))
                        if conditional_values
                        else float("nan")
                    )
                    rng = np.random.default_rng(
                        int(config["permutation_seed"])
                        + 10_000 * variant_index
                        + 1_000 * seed_index
                        + 100 * task_index
                        + int(layer)
                    )
                    null = np.empty(replicate_count, dtype=np.float64)
                    for replicate in range(replicate_count):
                        permuted, _ = contingency(rng.permutation(labels), counts)
                        null[replicate] = normalized_mutual_information(permuted)[0]
                    rows.append(
                        {
                            "variant": variant,
                            "seed": int(seed),
                            "task": task,
                            "layer": int(layer),
                            **observed,
                            "source_expert_normalized_mutual_information": source_nmi,
                            "dialect_minus_source_nmi": observed[
                                "normalized_mutual_information"
                            ]
                            - source_nmi,
                            "conditional_dialect_expert_nmi_within_source": conditional_dialect_nmi,
                            "sources_present": float(len(sources)),
                            "nmi_permutation_mean": float(null.mean()),
                            "nmi_permutation_std": float(null.std(ddof=1)),
                            "nmi_permutation_p_greater": float(
                                (int(np.count_nonzero(null >= observed["normalized_mutual_information"])) + 1)
                                / (replicate_count + 1)
                            ),
                            "dialect_labels": dialects,
                        }
                    )
                    task_distributions[(variant, int(seed), int(layer), task)] = (
                        counts.sum(axis=0) / max(1e-12, counts.sum())
                    )
    metrics = pd.DataFrame(rows)
    metrics["nmi_permutation_p_holm_within_variant_task"] = np.nan
    for _, indices in metrics.groupby(["variant", "task"]).groups.items():
        values = metrics.loc[indices, "nmi_permutation_p_greater"].to_numpy()
        metrics.loc[indices, "nmi_permutation_p_holm_within_variant_task"] = holm_adjust(values)
    task_rows = []
    for variant in config["variants"]:
        for seed in config["seeds"]:
            layers = sorted(
                key[2]
                for key in task_distributions
                if key[0] == variant and key[1] == int(seed) and key[3] == config["tasks"][0]
            )
            for layer in layers:
                first = task_distributions[(variant, int(seed), layer, config["tasks"][0])]
                second = task_distributions[(variant, int(seed), layer, config["tasks"][1])]
                task_rows.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "layer": layer,
                        "task_pair": f"{config['tasks'][0]}_vs_{config['tasks'][1]}",
                        "task_js_divergence": float(jensenshannon(first, second, base=2.0) ** 2),
                    }
                )
    task_metrics = pd.DataFrame(task_rows)
    output = PROJECT / "reports/routing" / str(config["protocol_id"])
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output / "expert_specialization_metrics.parquet", index=False)
    metrics.drop(columns=["dialect_labels"]).to_csv(
        output / "expert_specialization_metrics.csv", index=False
    )
    task_metrics.to_csv(output / "task_routing_divergence.csv", index=False)
    summary = (
        metrics.groupby(["variant", "task"], as_index=False)[
            [
                "normalized_mutual_information",
                "mean_pairwise_dialect_js_divergence",
                "expert_purity",
                "expert_load_cv",
                "normalized_expert_load_entropy",
                "source_expert_normalized_mutual_information",
                "dialect_minus_source_nmi",
                "nmi_permutation_p_greater",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(output / "expert_specialization_summary.csv", index=False)
    manifest = {
        "status": "COMPLETE",
        "protocol_id": config["protocol_id"],
        "variants": config["variants"],
        "seeds": config["seeds"],
        "tasks": config["tasks"],
        "permutation_replicates_per_layer": replicate_count,
        "rows": len(metrics),
        "expert_identity_alignment_across_seeds": "not_required_for_permutation_invariant_metrics",
    }
    (output / "routing_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
