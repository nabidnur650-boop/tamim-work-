#!/usr/bin/env python3
"""Fit a reference-free normalization candidate selector on development data.

Whole semantic groups, including every dialect realization and repeated model
output belonging to a group, are held out together in every fold.  The final
selector sees only source text, two candidate strings, and dialect probabilities
inferred from source text.  Gold dialect, source identity, and reference text
are not inference features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import CHRF
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.fusion import FUSION_FEATURE_NAMES, normalization_fusion_features  # noqa: E402
from boichitro.metrics import normalization_metrics  # noqa: E402
from tools.audit_source_blind_baselines import predict_dialects  # noqa: E402


SELECTION_VARIANTS = ("M0", "M1")
SELECTION_SEEDS = (1701, 2903, 4307)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    return parser.parse_args()


def model_specs(version: str):
    specifications = {
        "extra_leaf8": lambda: ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=-1,
            random_state=72119,
        ),
    }
    if version == "v1":
        specifications = {
            "hist_leaf15_l2": lambda: HistGradientBoostingRegressor(
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=5.0,
                learning_rate=0.05,
                random_state=72119,
            ),
            **specifications,
            "extra_leaf20": lambda: ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=20,
                max_features=0.8,
                n_jobs=-1,
                random_state=72119,
            ),
        }
    return specifications


def semantic_group_folds(
    records: pd.DataFrame, *, n_splits: int = 5, random_state: int = 72119
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assign repeated rows to folds while keeping semantic groups indivisible."""

    required = {"row_id", "semantic_group_id", "dialect"}
    if not required.issubset(records.columns):
        raise ValueError(f"Missing selector-fold columns: {sorted(required - set(records))}")
    consistency = records.groupby("row_id", observed=True).agg(
        semantic_groups=("semantic_group_id", "nunique"),
        dialects=("dialect", "nunique"),
    )
    if consistency.gt(1).any(axis=None):
        raise ValueError("Repeated selector rows disagree on semantic group or dialect")
    unique_rows = records[
        ["row_id", "semantic_group_id", "dialect"]
    ].drop_duplicates("row_id")
    if unique_rows["semantic_group_id"].astype(str).eq("").any():
        raise ValueError("Semantic-group selector folds require non-empty group IDs")
    fold_by_id: dict[str, int] = {}
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    for fold, (_, validation_indices) in enumerate(
        splitter.split(
            unique_rows["row_id"],
            unique_rows["dialect"],
            groups=unique_rows["semantic_group_id"],
        )
    ):
        for row_id in unique_rows.iloc[validation_indices]["row_id"]:
            fold_by_id[str(row_id)] = fold
    folds = records["row_id"].astype(str).map(fold_by_id)
    if folds.isna().any():
        raise RuntimeError("At least one selector row was not assigned to a fold")
    audit = unique_rows.assign(
        fold=unique_rows["row_id"].astype(str).map(fold_by_id)
    )
    if audit.groupby("semantic_group_id", observed=True)["fold"].nunique().gt(1).any():
        raise RuntimeError("A semantic group was assigned to more than one selector fold")
    return folds.to_numpy(dtype=np.int64), unique_rows


def main() -> None:
    args = parse_args()
    version = str(args.version)
    predecessor_path = PROJECT / "reports/model/normalization_fusion_selection.json"
    predecessor = None
    specifications = model_specs(version)
    if version == "v2":
        predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
        preselected_model = str(predecessor.get("selected", {}).get("model", ""))
        v1_specifications = model_specs("v1")
        if preselected_model not in v1_specifications:
            raise RuntimeError(
                "V2 selector family must be a model selected by the V1 protocol"
            )
        specifications = {preselected_model: v1_specifications[preselected_model]}
    rewrite_model_id = (
        "N_WORD_REWRITE_INFERRED_DIALECT"
        if version == "v1"
        else "N_WORD_REWRITE_INFERRED_SUPPORTED_DIALECT"
    )
    rewrite_path = (
        PROJECT
        / "predictions/development_source_blind/"
        f"{rewrite_model_id}_validation.parquet"
    )
    if not rewrite_path.exists():
        raise FileNotFoundError(
            f"Run tools/audit_source_blind_baselines.py first: {rewrite_path}"
        )
    rewrite = pd.read_parquet(rewrite_path).rename(columns={"prediction": "rewrite"})
    inferred_labels, inferred_probabilities = predict_dialects(rewrite["source"])
    probability_by_id = {
        str(row_id): probabilities
        for row_id, probabilities in zip(
            rewrite["row_id"], inferred_probabilities, strict=True
        )
    }
    rewrite_by_id = {
        str(row_id): value for row_id, value in zip(rewrite["row_id"], rewrite["rewrite"])
    }

    metric = CHRF(word_order=2)
    feature_rows: list[np.ndarray] = []
    targets: list[float] = []
    records: list[dict] = []
    run_frames: dict[tuple[str, int], pd.DataFrame] = {}
    print("Extracting source-blind selector features", flush=True)
    for variant in SELECTION_VARIANTS:
        for seed in SELECTION_SEEDS:
            stage = PROJECT / f"runs/task/boichitro_q1_v1/{variant}__base/{seed}/stage_s"
            selection = json.loads((stage / "best_selection.json").read_text(encoding="utf-8"))
            validation_id = int(selection["validation_id"])
            predictions_path = stage / f"validation_predictions_epoch_{validation_id:02d}.parquet"
            frame = pd.read_parquet(predictions_path).copy()
            frame["rewrite"] = [rewrite_by_id[str(row_id)] for row_id in frame["row_id"]]
            run_frames[(variant, seed)] = frame
            for local_index, row in enumerate(frame.itertuples(index=False)):
                probabilities = probability_by_id[str(row.row_id)]
                feature_rows.append(
                    normalization_fusion_features(
                        row.source,
                        row.prediction,
                        row.rewrite,
                        probabilities,
                    )
                )
                neural_score = metric.sentence_score(
                    str(row.prediction), [str(row.reference)]
                ).score
                rewrite_score = metric.sentence_score(
                    str(row.rewrite), [str(row.reference)]
                ).score
                targets.append(float(neural_score - rewrite_score))
                records.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "local_index": local_index,
                        "row_id": str(row.row_id),
                        "semantic_group_id": str(row.semantic_group_id),
                        "dialect": str(row.dialect),
                        "neural_sentence_chrfpp": float(neural_score),
                        "rewrite_sentence_chrfpp": float(rewrite_score),
                    }
                )

    features = np.stack(feature_rows)
    target = np.asarray(targets, dtype=np.float64)
    record_frame = pd.DataFrame(records)
    folds, unique_rows = semantic_group_folds(record_frame)
    if len(unique_rows) != len(rewrite):
        raise RuntimeError("Row-level selector folds do not match validation rows")

    threshold_grid = np.arange(-4.0, 16.1, 1.0)
    selection_rows: list[dict] = []
    oof_predictions: dict[str, np.ndarray] = {}
    for model_name, factory in specifications.items():
        print(f"Cross-validating {model_name}", flush=True)
        predicted_margin = np.zeros(len(features), dtype=np.float64)
        for fold in range(5):
            training = folds != fold
            validation = folds == fold
            model = factory()
            model.fit(features[training], target[training])
            predicted_margin[validation] = model.predict(features[validation])
        oof_predictions[model_name] = predicted_margin
        correlation = float(np.corrcoef(predicted_margin, target)[0, 1])
        for threshold in threshold_grid:
            use_neural = predicted_margin > threshold
            per_run = []
            for (variant, seed), frame in run_frames.items():
                indices = record_frame.index[
                    record_frame["variant"].eq(variant)
                    & record_frame["seed"].eq(seed)
                ].to_numpy()
                selected = frame.copy()
                selected["prediction"] = np.where(
                    use_neural[indices],
                    frame["prediction"].astype(str),
                    frame["rewrite"].astype(str),
                )
                metrics, _ = normalization_metrics(selected)
                per_run.append(metrics)
            selection_rows.append(
                {
                    "model": model_name,
                    "threshold": float(threshold),
                    "oof_margin_correlation": correlation,
                    "neural_selection_fraction": float(use_neural.mean()),
                    "selected_neural_win_rate": float(
                        np.mean(target[use_neural] > 0) if use_neural.any() else 0.0
                    ),
                    "mean_macro_chrfpp": float(
                        np.mean([row["macro_chrfpp"] for row in per_run])
                    ),
                    "mean_worst_dialect_chrfpp": float(
                        np.mean([row["worst_dialect_chrfpp"] for row in per_run])
                    ),
                    "mean_corpus_chrfpp": float(
                        np.mean([row["corpus_chrfpp"] for row in per_run])
                    ),
                    "mean_exact_match": float(
                        np.mean([row["exact_match"] for row in per_run])
                    ),
                }
            )

    selected = max(
        selection_rows,
        key=lambda row: (
            row["mean_macro_chrfpp"],
            row["mean_worst_dialect_chrfpp"],
            row["mean_corpus_chrfpp"],
            -abs(row["threshold"]),
        ),
    )
    model_name = str(selected["model"])
    threshold = float(selected["threshold"])
    predicted_margin = oof_predictions[model_name]
    use_neural = predicted_margin > threshold
    output_root = (
        PROJECT / f"predictions/development_fusion/normalization_selector_{version}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    selected_runs = []
    for (variant, seed), frame in run_frames.items():
        indices = record_frame.index[
            record_frame["variant"].eq(variant) & record_frame["seed"].eq(seed)
        ].to_numpy()
        output = frame.copy()
        output["neural_prediction"] = frame["prediction"].astype(str)
        output["rewrite_prediction"] = frame["rewrite"].astype(str)
        output["selector_predicted_margin"] = predicted_margin[indices]
        output["selected_neural"] = use_neural[indices]
        output["prediction"] = np.where(
            output["selected_neural"],
            output["neural_prediction"],
            output["rewrite_prediction"],
        )
        metrics, by_dialect = normalization_metrics(output)
        destination = output_root / f"{variant}__base/{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        output.to_parquet(destination / "validation_predictions_oof.parquet", index=False)
        by_dialect.to_csv(destination / "validation_by_dialect_oof.csv", index=False)
        (destination / "validation_metrics_oof.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        selected_runs.append(
            {
                "variant": variant,
                "seed": seed,
                "neural_selection_fraction": float(output["selected_neural"].mean()),
                **metrics,
            }
        )

    final_model = specifications[model_name]()
    final_model.fit(features, target)
    artifact_root = PROJECT / "artifacts/fusion"
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / f"normalization_selector_{version}.pkl"
    source_blind_audit = json.loads(
        (
            PROJECT / "reports/model/source_blind_normalization_baseline_audit.json"
        ).read_text(encoding="utf-8")
    )
    inferred_dialect_candidates = (
        list(source_blind_audit["normalization_supported_dialects"])
        if version == "v2"
        else None
    )
    artifact = {
        "model": final_model,
        "model_name": model_name,
        "threshold": threshold,
        "feature_names": list(FUSION_FEATURE_NAMES),
        "source_blind": True,
        "rewrite_candidate_id": rewrite_model_id,
        "inference_inputs": [
            "source_text",
            "neural_candidate",
            "inferred_dialect_rewrite_candidate",
            "inferred_dialect_probabilities",
        ],
    }
    if inferred_dialect_candidates is not None:
        artifact["inferred_dialect_candidates"] = inferred_dialect_candidates
    with artifact_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    report_root = PROJECT / "reports/model"
    grid_name = (
        "normalization_fusion_selection_grid.csv"
        if version == "v1"
        else "normalization_fusion_selection_v2_grid.csv"
    )
    oof_name = (
        "normalization_fusion_oof_rows.parquet"
        if version == "v1"
        else "normalization_fusion_v2_oof_rows.parquet"
    )
    report_name = (
        "normalization_fusion_selection.json"
        if version == "v1"
        else "normalization_fusion_selection_v2.json"
    )
    pd.DataFrame(selection_rows).sort_values(
        ["mean_macro_chrfpp", "mean_worst_dialect_chrfpp"], ascending=False
    ).to_csv(report_root / grid_name, index=False)
    record_frame.assign(
        fold=folds,
        target_margin=target,
        selected_model_oof_margin=predicted_margin,
        selected_neural=use_neural,
    ).to_parquet(report_root / oof_name, index=False)
    baseline_metrics = json.loads(
        (
            PROJECT
            / "metrics/development_source_blind/"
            f"{rewrite_model_id}_validation.json"
        ).read_text(encoding="utf-8")
    )
    report = {
        "status": "COMPLETE_VALIDATION_ONLY",
        "protocol_id": f"boichitro_normalization_selector_{version}",
        "test_data_access": False,
        "source_blind": True,
        "development_score_scope": "selection_conditioned_exploratory",
        "confirmatory_inference": False,
        "selection_reuse_disclosure": (
            "Model family and/or threshold selection uses these development folds; "
            "the reported OOF score is for system selection and is not an unbiased "
            "locked confirmation."
        ),
        "semantic_group_cross_validation_folds": 5,
        "cross_validation_group_unit": "semantic_group_id",
        "unique_validation_rows": int(len(unique_rows)),
        "unique_semantic_groups": int(unique_rows["semantic_group_id"].nunique()),
        "semantic_groups_spanning_multiple_dialects": int(
            unique_rows.groupby("semantic_group_id")["dialect"].nunique().gt(1).sum()
        ),
        "selection_variants": list(SELECTION_VARIANTS),
        "selection_seeds": list(SELECTION_SEEDS),
        "feature_names": list(FUSION_FEATURE_NAMES),
        "forbidden_inference_features": [
            "reference",
            "gold_dialect",
            "source_id",
            "evaluation_track",
        ],
        "baseline": {
            "model_id": baseline_metrics["model_id"],
            "macro_chrfpp": baseline_metrics["macro_chrfpp"],
            "worst_dialect_chrfpp": baseline_metrics["worst_dialect_chrfpp"],
        },
        "selected": selected,
        "artifact": str(artifact_path.relative_to(PROJECT)),
        "artifact_sha256": sha256_file(artifact_path),
        "runs": selected_runs,
    }
    if version == "v2":
        assert predecessor is not None and inferred_dialect_candidates is not None
        report.update(
            {
                "predecessor_protocol_id": predecessor["protocol_id"],
                "predecessor_selection_report": str(
                    predecessor_path.relative_to(PROJECT)
                ),
                "preselected_model_family": str(
                    predecessor["selected"]["model"]
                ),
                "v2_change": (
                    "restrict dialect inference to labels represented in the "
                    "training-only normalization rewrite inventory"
                ),
                "inferred_dialect_candidates": inferred_dialect_candidates,
            }
        )
    (report_root / report_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
