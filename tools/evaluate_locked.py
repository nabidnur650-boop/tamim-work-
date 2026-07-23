#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import EncodedDataset, load_encoded_cache  # noqa: E402
from boichitro.inference import (  # noqa: E402
    greedy_normalize,
    predict_identification,
    trace_routing,
)
from boichitro.fusion import (  # noqa: E402
    blend_identification_probabilities,
    classical_identification_probabilities,
    inferred_dialect_rewrite,
    select_normalization_candidates,
)
from boichitro.metrics import classification_metrics, normalization_metrics  # noqa: E402
from boichitro.modeling import BoichitroConfig, BoichitroForMultiTask  # noqa: E402
from boichitro.protocol import require_frozen_artifact, require_protocol_freeze  # noqa: E402
from boichitro.tokenization import load_tokenizer, sha256_file  # noqa: E402


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_fusion_resources(config: dict, freeze: dict | None) -> dict[str, Any] | None:
    details = config.get("fusion", {})
    if not details.get("enabled", False):
        return None
    identification_report_path = PROJECT / details["identification_selection_report"]
    normalization_report_path = PROJECT / details["normalization_selection_report"]
    identification_report = json.loads(
        identification_report_path.read_text(encoding="utf-8")
    )
    normalization_report = json.loads(
        normalization_report_path.read_text(encoding="utf-8")
    )
    for name, report in (
        ("identification", identification_report),
        ("normalization", normalization_report),
    ):
        if report.get("status") != "COMPLETE_VALIDATION_ONLY":
            raise RuntimeError(f"{name} fusion selection is not complete")
        if report.get("test_data_access") is not False:
            raise RuntimeError(f"{name} fusion selection was not validation-only")
        if report.get("source_blind") is not True:
            raise RuntimeError(f"{name} fusion selection is not source-blind")

    svm_path = PROJECT / identification_report["svm_artifact"]
    rewrite_path = PROJECT / details["normalization_rewrite_artifact"]
    selector_path = PROJECT / normalization_report["artifact"]
    paths = (
        identification_report_path,
        normalization_report_path,
        svm_path,
        rewrite_path,
        selector_path,
    )
    hashes = {
        str(path.relative_to(PROJECT)): (
            require_frozen_artifact(PROJECT, freeze, path)
            if freeze is not None
            else sha256_file(path)
        )
        for path in paths
    }
    if hashes[str(svm_path.relative_to(PROJECT))] != identification_report.get(
        "svm_artifact_sha256"
    ):
        raise RuntimeError("Identification fusion SVM hash does not match selection report")
    if hashes[str(selector_path.relative_to(PROJECT))] != normalization_report.get(
        "artifact_sha256"
    ):
        raise RuntimeError(
            "Normalization selector hash does not match selection report"
        )
    return {
        "identification_report": identification_report,
        "normalization_report": normalization_report,
        "svm": load_pickle(svm_path),
        "rewrite_mapping": load_pickle(rewrite_path),
        "normalization_selector": load_pickle(selector_path),
        "artifact_sha256": hashes,
        "contract": str(details["contract"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen checkpoints once on locked tests.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/locked_evaluation.yaml"
    )
    parser.add_argument(
        "--task-config", type=Path, default=PROJECT / "configs/task_experiments.yaml"
    )
    parser.add_argument("--variants", nargs="+", choices=("M0", "M1", "M2", "M3"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--ablation", action="append", default=[])
    parser.add_argument(
        "--task-protocol-id",
        help="Read frozen task checkpoints from this protocol (for registered ablations).",
    )
    parser.add_argument("--allow-identical-rerun", action="store_true")
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> BoichitroForMultiTask:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = BoichitroForMultiTask(BoichitroConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def selected_normalization_checkpoint(run_root: Path) -> tuple[Path, dict[str, Any]]:
    stage_s = run_root / "stage_s/best_checkpoint.pt"
    stage_a = run_root / "stage_a/last_checkpoint.pt"
    if not stage_s.exists():
        raise FileNotFoundError(f"Missing normalization checkpoint: {stage_s}")
    payload = torch.load(stage_s, map_location="cpu", weights_only=False)
    validation = payload.get("extra", {}).get("validation", {})
    guard_pass = bool(validation.get("replay_guard_pass", 0.0))
    selected = stage_s if guard_pass else stage_a
    return selected, {
        "stage_s_candidate": str(stage_s.relative_to(PROJECT)),
        "replay_guard_pass": guard_pass,
        "validation": validation,
        "selected": str(selected.relative_to(PROJECT)),
    }


def stratified_hash_subset(dataset: EncodedDataset, count: int) -> EncodedDataset:
    by_dialect: dict[str, list[int]] = {}
    for index, example in enumerate(dataset.examples):
        by_dialect.setdefault(str(example["dialect"]), []).append(index)
    dialects = sorted(by_dialect)
    base, remainder = divmod(count, max(1, len(dialects)))
    selected: list[int] = []
    unused: list[int] = []
    for dialect_index, dialect in enumerate(dialects):
        ranked = sorted(
            by_dialect[dialect],
            key=lambda index: hashlib.sha256(
                str(dataset.examples[index]["row_id"]).encode("utf-8")
            ).digest(),
        )
        quota = base + int(dialect_index < remainder)
        selected.extend(ranked[:quota])
        unused.extend(ranked[quota:])
    if len(selected) < min(count, len(dataset)):
        unused.sort(
            key=lambda index: hashlib.sha256(
                str(dataset.examples[index]["row_id"]).encode("utf-8")
            ).digest()
        )
        selected.extend(unused[: min(count, len(dataset)) - len(selected)])
    return EncodedDataset([dataset.examples[index] for index in selected])


def prefix_only(dataset: EncodedDataset) -> EncodedDataset:
    examples = []
    for original in dataset.examples:
        example = dict(original)
        supervised = [index for index, label in enumerate(example["labels"]) if label != -100]
        stop = supervised[0] if supervised else len(example["input_ids"])
        example["input_ids"] = list(example["input_ids"][:stop])
        example["labels"] = [-100] * stop
        examples.append(example)
    return EncodedDataset(examples)


def filter_dialects(frame, details):
    if details.get("include_dialects"):
        frame = frame.loc[
            frame["dialect"].isin([str(value) for value in details["include_dialects"]])
        ]
    if details.get("exclude_dialects"):
        frame = frame.loc[
            ~frame["dialect"].isin([str(value) for value in details["exclude_dialects"]])
        ]
    if frame.empty:
        raise ValueError(f"Dialect filter produced an empty evaluation track: {details}")
    return frame.copy()


def filter_encoded_dialects(dataset: EncodedDataset, details: dict) -> EncodedDataset:
    included = frozenset(str(value) for value in details.get("include_dialects", []))
    excluded = frozenset(str(value) for value in details.get("exclude_dialects", []))
    examples = [
        example
        for example in dataset.examples
        if (not included or str(example["dialect"]) in included)
        and str(example["dialect"]) not in excluded
    ]
    if not examples:
        raise ValueError(f"Dialect filter produced an empty routing track: {details}")
    return EncodedDataset(examples)


def save_normalization_track(
    model,
    tokenizer,
    frame: pd.DataFrame,
    *,
    track: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    fusion: dict[str, Any] | None,
) -> tuple[
    dict[str, Any], dict[str, Any] | None, dict[str, Any] | None
]:
    source_column = "romanized_input_model" if "romanized_input_model" in frame else "source_text_model"
    predictions = greedy_normalize(
        model,
        tokenizer,
        frame[source_column].astype(str).tolist(),
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    prediction_frame = pd.DataFrame(
        {
            "row_id": frame["row_id"].astype(str),
            "semantic_group_id": frame.get("semantic_group_id", frame["row_id"]).astype(str),
            "dialect": frame["dialect"].astype(str),
            "source_id": frame["source_id"].astype(str),
            "source": frame[source_column].astype(str),
            "reference": frame["target_text_model"].astype(str),
            "prediction": predictions,
        }
    )
    metrics, by_dialect = normalization_metrics(prediction_frame)
    prediction_frame.to_parquet(output_dir / f"normalization_{track}.parquet", index=False)
    by_dialect.to_csv(output_dir / f"normalization_{track}_by_dialect.csv", index=False)
    source_rows = []
    for source_id, group in prediction_frame.groupby("source_id", sort=True):
        source_metrics, _ = normalization_metrics(group)
        source_rows.append({"source_id": source_id, **source_metrics})
    pd.DataFrame(source_rows).to_csv(
        output_dir / f"normalization_{track}_by_source.csv", index=False
    )
    (output_dir / f"normalization_{track}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if fusion is None:
        return metrics, None, None

    source_texts = prediction_frame["source"].astype(str).tolist()
    neural_candidates = prediction_frame["prediction"].astype(str).tolist()
    svm_probabilities = classical_identification_probabilities(
        source_texts, fusion["svm"]
    )
    rewrite_candidates, inferred_dialects = inferred_dialect_rewrite(
        source_texts,
        svm_probabilities,
        fusion["rewrite_mapping"],
        allowed_dialects=fusion["normalization_selector"].get(
            "inferred_dialect_candidates"
        ),
    )
    rewrite_only = prediction_frame.copy()
    rewrite_only["inferred_dialect"] = inferred_dialects
    rewrite_only["inferred_dialect_probabilities"] = svm_probabilities.tolist()
    rewrite_only["prediction"] = rewrite_candidates
    rewrite_metrics, rewrite_by_dialect = normalization_metrics(rewrite_only)
    rewrite_only.to_parquet(
        output_dir / f"normalization_inferred_rewrite_{track}.parquet", index=False
    )
    rewrite_by_dialect.to_csv(
        output_dir / f"normalization_inferred_rewrite_{track}_by_dialect.csv",
        index=False,
    )
    (output_dir / f"normalization_inferred_rewrite_{track}_metrics.json").write_text(
        json.dumps(rewrite_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fused_predictions, selector_margins, selected_neural = (
        select_normalization_candidates(
            source_texts,
            neural_candidates,
            rewrite_candidates,
            svm_probabilities,
            fusion["normalization_selector"],
        )
    )
    fused = prediction_frame.copy()
    fused["neural_prediction"] = neural_candidates
    fused["rewrite_prediction"] = rewrite_candidates
    fused["inferred_dialect"] = inferred_dialects
    fused["inferred_dialect_probabilities"] = svm_probabilities.tolist()
    fused["selector_predicted_neural_margin"] = selector_margins
    fused["selected_neural"] = selected_neural
    fused["prediction"] = fused_predictions
    fused_metrics, fused_by_dialect = normalization_metrics(fused)
    fused.to_parquet(
        output_dir / f"normalization_fused_{track}.parquet", index=False
    )
    fused_by_dialect.to_csv(
        output_dir / f"normalization_fused_{track}_by_dialect.csv", index=False
    )
    fused_source_rows = []
    for source_id, group in fused.groupby("source_id", sort=True):
        source_metrics, _ = normalization_metrics(group)
        fused_source_rows.append({"source_id": source_id, **source_metrics})
    pd.DataFrame(fused_source_rows).to_csv(
        output_dir / f"normalization_fused_{track}_by_source.csv", index=False
    )
    fused_metrics = {
        **fused_metrics,
        "selected_neural_fraction": float(selected_neural.mean()),
    }
    (output_dir / f"normalization_fused_{track}_metrics.json").write_text(
        json.dumps(fused_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics, fused_metrics, rewrite_metrics


def save_identification_track(
    model,
    dataset: EncodedDataset,
    *,
    track: str,
    output_dir: Path,
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
    temperature: float,
    texts: list[str],
    fusion: dict[str, Any] | None,
) -> tuple[
    dict[str, float], dict[str, float] | None, dict[str, float] | None
]:
    predictions, _ = predict_identification(
        model,
        dataset,
        device=device,
        pad_token_id=pad_token_id,
        batch_size=batch_size,
        temperature=temperature,
    )
    probabilities = np.asarray(predictions["probabilities"].tolist(), dtype=np.float64)
    metrics, by_class, matrix = classification_metrics(
        predictions["label_id"].to_numpy(),
        predictions["prediction_id"].to_numpy(),
        probabilities,
    )
    predictions.to_parquet(output_dir / f"identification_{track}.parquet", index=False)
    by_class.to_csv(output_dir / f"identification_{track}_by_class.csv", index=False)
    np.save(output_dir / f"identification_{track}_confusion.npy", matrix)
    (output_dir / f"identification_{track}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    if fusion is None:
        return metrics, None, None
    if len(texts) != len(predictions):
        raise ValueError(
            f"Identification text/prediction count mismatch: {len(texts)} != {len(predictions)}"
        )
    svm_probabilities = classical_identification_probabilities(texts, fusion["svm"])
    svm_prediction_ids = svm_probabilities.argmax(axis=1)
    svm_metrics, svm_by_class, svm_matrix = classification_metrics(
        predictions["label_id"].to_numpy(),
        svm_prediction_ids,
        svm_probabilities,
    )
    svm_predictions = predictions.copy()
    svm_predictions["probabilities"] = svm_probabilities.tolist()
    svm_predictions["prediction_id"] = svm_prediction_ids
    svm_predictions.to_parquet(
        output_dir / f"identification_char_svm_{track}.parquet", index=False
    )
    svm_by_class.to_csv(
        output_dir / f"identification_char_svm_{track}_by_class.csv", index=False
    )
    np.save(output_dir / f"identification_char_svm_{track}_confusion.npy", svm_matrix)
    (output_dir / f"identification_char_svm_{track}_metrics.json").write_text(
        json.dumps(svm_metrics, indent=2) + "\n", encoding="utf-8"
    )
    neural_weight = float(
        fusion["identification_report"]["selected_alpha_neural"]
    )
    fused_probabilities = blend_identification_probabilities(
        probabilities,
        svm_probabilities,
        neural_weight=neural_weight,
    )
    fused_prediction_ids = fused_probabilities.argmax(axis=1)
    fused_metrics, fused_by_class, fused_matrix = classification_metrics(
        predictions["label_id"].to_numpy(),
        fused_prediction_ids,
        fused_probabilities,
    )
    fused = predictions.copy()
    fused["neural_probabilities"] = probabilities.tolist()
    fused["svm_probabilities"] = svm_probabilities.tolist()
    fused["probabilities"] = fused_probabilities.tolist()
    fused["prediction_id"] = fused_prediction_ids
    fused.to_parquet(
        output_dir / f"identification_fused_{track}.parquet", index=False
    )
    fused_by_class.to_csv(
        output_dir / f"identification_fused_{track}_by_class.csv", index=False
    )
    np.save(
        output_dir / f"identification_fused_{track}_confusion.npy", fused_matrix
    )
    fused_metrics = {
        **fused_metrics,
        "neural_probability_weight": neural_weight,
        "svm_probability_weight": 1.0 - neural_weight,
    }
    (output_dir / f"identification_fused_{track}_metrics.json").write_text(
        json.dumps(fused_metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics, fused_metrics, svm_metrics


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    freeze = (
        require_protocol_freeze(PROJECT, str(config["protocol_freeze_id"]))
        if config.get("protocol_freeze_id")
        else None
    )
    task_config = yaml.safe_load(args.task_config.read_text(encoding="utf-8"))
    variants = args.variants or list(task_config["variants"])
    seeds = args.seeds or [int(seed) for seed in task_config["seeds"]]
    task_protocol_id = str(
        args.task_protocol_id
        or config.get("task_protocol_id")
        or task_config["protocol_id"]
    )
    suffix = "base" if not args.ablation else "__".join(sorted(args.ablation))
    tokenizer = load_tokenizer(PROJECT / "artifacts/tokenizers/frozen")
    pad_token_id = int(tokenizer.token_to_id("<pad>"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    fusion = load_fusion_resources(config, freeze)

    for variant in variants:
        for seed in seeds:
            task_root = (
                PROJECT
                / "runs/task"
                / task_protocol_id
                / f"{variant}__{suffix}"
                / str(seed)
            )
            norm_checkpoint, norm_selection = selected_normalization_checkpoint(task_root)
            id_checkpoint = task_root / "stage_id/best_checkpoint.pt"
            calibration_path = task_root / "stage_id/temperature_calibration.json"
            if not id_checkpoint.exists() or not calibration_path.exists():
                raise FileNotFoundError(f"Frozen ID artifacts missing under {task_root}")
            if freeze is not None:
                checkpoint_hashes = {
                    "normalization": require_frozen_artifact(
                        PROJECT, freeze, norm_checkpoint
                    ),
                    "identification": require_frozen_artifact(
                        PROJECT, freeze, id_checkpoint
                    ),
                }
                require_frozen_artifact(PROJECT, freeze, calibration_path)
            else:
                checkpoint_hashes = {
                    "normalization": sha256_file(norm_checkpoint),
                    "identification": sha256_file(id_checkpoint),
                }
            output_dir = (
                PROJECT
                / "predictions"
                / str(config["protocol_id"])
                / f"{variant}__{suffix}"
                / str(seed)
            )
            manifest_path = output_dir / "evaluation_manifest.json"
            if manifest_path.exists():
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                if previous.get("checkpoint_sha256") != checkpoint_hashes:
                    raise RuntimeError(
                        f"Refusing to overwrite test outputs from different checkpoints: {output_dir}"
                    )
                if not args.allow_identical_rerun:
                    print(f"Locked evaluation already complete, skipping {variant} seed {seed}")
                    continue
            output_dir.mkdir(parents=True, exist_ok=True)

            norm_model = load_model(norm_checkpoint, device)
            normalization_results = {}
            normalization_fused_results = {}
            normalization_source_blind_baseline_results = {}
            for track, details in config["normalization"]["tracks"].items():
                frame = pd.read_parquet(PROJECT / details["frame"])
                frame = filter_dialects(frame, details)
                if config["normalization"].get("max_examples"):
                    frame = frame.head(int(config["normalization"]["max_examples"])).copy()
                raw_metrics, fused_metrics, rewrite_metrics = save_normalization_track(
                    norm_model,
                    tokenizer,
                    frame,
                    track=track,
                    output_dir=output_dir,
                    device=device,
                    batch_size=int(config["normalization"]["batch_size"]),
                    max_new_tokens=int(config["normalization"]["max_new_tokens"]),
                    fusion=fusion,
                )
                normalization_results[track] = raw_metrics
                if fused_metrics is not None:
                    normalization_fused_results[track] = fused_metrics
                if rewrite_metrics is not None:
                    normalization_source_blind_baseline_results[track] = rewrite_metrics

            norm_trace_cache = load_encoded_cache(
                PROJECT / config["normalization"]["tracks"]["source_ood"]["cache"]
            )[0]
            norm_trace_cache = filter_encoded_dialects(
                norm_trace_cache,
                config["normalization"]["tracks"]["source_ood"],
            )
            norm_trace = stratified_hash_subset(
                prefix_only(norm_trace_cache), int(config["routing"]["examples_per_track"])
            )
            if norm_model.config.architecture != "dense":
                trace_routing(
                    norm_model,
                    norm_trace,
                    device=device,
                    pad_token_id=pad_token_id,
                    batch_size=int(config["routing"]["batch_size"]),
                ).to_parquet(output_dir / "routing_normalization_source_ood.parquet", index=False)
            del norm_model
            torch.cuda.empty_cache() if device.type == "cuda" else None

            id_model = load_model(id_checkpoint, device)
            temperature = float(
                json.loads(calibration_path.read_text(encoding="utf-8"))["temperature"]
            )
            identification_results = {}
            identification_fused_results = {}
            identification_classical_baseline_results = {}
            id_datasets = {}
            for track, details in config["identification"]["tracks"].items():
                if not isinstance(details, dict):
                    raise TypeError(
                        "Locked identification tracks must declare cache, frame, and split"
                    )
                dataset = load_encoded_cache(PROJECT / details["cache"])[0]
                if config["identification"].get("max_examples"):
                    dataset = EncodedDataset(
                        dataset.examples[: int(config["identification"]["max_examples"])]
                    )
                id_datasets[track] = dataset
                text_frame = pd.read_parquet(
                    PROJECT / details["frame"], columns=["row_id", "split", "text_model"]
                )
                text_frame = text_frame.loc[
                    text_frame["split"].eq(str(details["split"]))
                ]
                if text_frame["row_id"].astype(str).duplicated().any():
                    raise RuntimeError(
                        f"Duplicate identification row IDs in locked track {track}"
                    )
                text_by_id = dict(
                    zip(
                        text_frame["row_id"].astype(str),
                        text_frame["text_model"].astype(str),
                        strict=True,
                    )
                )
                missing_text = [
                    str(example["row_id"])
                    for example in dataset.examples
                    if str(example["row_id"]) not in text_by_id
                ]
                if missing_text:
                    raise RuntimeError(
                        f"Missing source text for {len(missing_text)} rows in {track}"
                    )
                texts = [
                    text_by_id[str(example["row_id"])] for example in dataset.examples
                ]
                raw_metrics, fused_metrics, svm_metrics = save_identification_track(
                    id_model,
                    dataset,
                    track=track,
                    output_dir=output_dir,
                    device=device,
                    pad_token_id=pad_token_id,
                    batch_size=int(config["identification"]["batch_size"]),
                    temperature=temperature,
                    texts=texts,
                    fusion=fusion,
                )
                identification_results[track] = raw_metrics
                if fused_metrics is not None:
                    identification_fused_results[track] = fused_metrics
                if svm_metrics is not None:
                    identification_classical_baseline_results[track] = svm_metrics
            if id_model.config.architecture != "dense":
                id_trace = stratified_hash_subset(
                    id_datasets["source_ood"], int(config["routing"]["examples_per_track"])
                )
                trace_routing(
                    id_model,
                    id_trace,
                    device=device,
                    pad_token_id=pad_token_id,
                    batch_size=int(config["routing"]["batch_size"]),
                ).to_parquet(
                    output_dir / "routing_identification_source_ood.parquet", index=False
                )
            del id_model
            torch.cuda.empty_cache() if device.type == "cuda" else None

            manifest = {
                "status": "COMPLETE_LOCKED_EVALUATION",
                "protocol_id": config["protocol_id"],
                "task_protocol_id": task_protocol_id,
                "variant": variant,
                "seed": seed,
                "ablations": sorted(args.ablation),
                "checkpoint_sha256": checkpoint_hashes,
                "normalization_checkpoint_selection": norm_selection,
                "identification_checkpoint": str(id_checkpoint.relative_to(PROJECT)),
                "temperature": temperature,
                "normalization": normalization_results,
                "normalization_fused": normalization_fused_results,
                "normalization_source_blind_baseline": (
                    normalization_source_blind_baseline_results
                ),
                "identification": identification_results,
                "identification_fused": identification_fused_results,
                "identification_classical_baseline": (
                    identification_classical_baseline_results
                ),
                "fusion": {
                    "enabled": fusion is not None,
                    "contract": fusion["contract"] if fusion is not None else None,
                    "artifact_sha256": (
                        fusion["artifact_sha256"] if fusion is not None else {}
                    ),
                    "identification_protocol_id": (
                        fusion["identification_report"]["protocol_id"]
                        if fusion is not None
                        else None
                    ),
                    "normalization_protocol_id": (
                        fusion["normalization_report"]["protocol_id"]
                        if fusion is not None
                        else None
                    ),
                },
                "tokenizer_sha256": sha256_file(
                    PROJECT / "artifacts/tokenizers/frozen/tokenizer.json"
                ),
                "evaluation_config_sha256": sha256_file(args.config),
                "protocol_freeze": {
                    "protocol_id": freeze["protocol_id"],
                    "code_sha256": freeze["code_sha256"],
                    "config_sha256": freeze["config_sha256"],
                    "selected_artifacts_sha256": freeze["selected_artifacts_sha256"],
                }
                if freeze is not None
                else None,
                "test_policy": config["test_policy"],
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Locked evaluation complete: {variant} seed {seed}", flush=True)


if __name__ == "__main__":
    main()
