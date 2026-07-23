#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.protocol import (  # noqa: E402
    file_manifest,
    freeze_manifest_path,
    manifest_sha256,
    protocol_fingerprints,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Irreversibly fingerprint all development choices before locked-test access."
    )
    parser.add_argument("--protocol-id", default="locked_test_v1")
    return parser.parse_args()


def require_complete(path: Path, expected: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required development artifact missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != expected:
        raise RuntimeError(
            f"Development artifact is not complete ({payload.get('status')}): {path}"
        )


def selected_validation_paths(stage_dir: Path) -> list[Path]:
    selection_path = stage_dir / "best_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"Validation selection manifest missing: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "SELECTED_ON_VALIDATION":
        raise RuntimeError(f"Invalid validation selection manifest: {selection_path}")
    validation_id = int(selection["validation_id"])
    predictions = stage_dir / f"validation_predictions_epoch_{validation_id:02d}.parquet"
    if not predictions.exists():
        raise FileNotFoundError(f"Selected validation predictions missing: {predictions}")
    return [selection_path, predictions]


def task_paths(
    *, protocol: str, variant: str, suffix: str, seed: int, identification: bool
) -> list[Path]:
    root = PROJECT / "runs/task" / protocol / f"{variant}__{suffix}" / str(seed)
    require_complete(root / "stage_a/training_report.json", "COMPLETE")
    require_complete(root / "stage_s/training_report.json", "COMPLETE")
    require_complete(root / "run_manifest.json", "COMPLETE_VALIDATION_ONLY")
    required = [
        root / "stage_a/mixture_report.json",
        root / "stage_a/training_report.json",
        root / "stage_a/train_log.jsonl",
        root / "stage_a/checkpoint_retention.json",
        root / "stage_s/best_checkpoint.pt",
        root / "stage_s/mixture_report.json",
        root / "stage_s/training_report.json",
        root / "data_ablation_report.json",
        root / "initialization_report.json",
        root / "run_manifest.json",
    ]
    adoption_report = root / "pilot_adoption_report.json"
    if adoption_report.exists():
        required.append(adoption_report)
    required.extend(selected_validation_paths(root / "stage_s"))
    stage_s_selection = json.loads(
        (root / "stage_s/best_selection.json").read_text(encoding="utf-8")
    )
    stage_a_checkpoint = root / "stage_a/last_checkpoint.pt"
    if not bool(
        stage_s_selection.get("validation", {}).get("replay_guard_pass", False)
    ):
        if not stage_a_checkpoint.exists():
            raise FileNotFoundError(
                f"Replay-guard fallback Stage-A checkpoint missing: {stage_a_checkpoint}"
            )
        required.append(stage_a_checkpoint)
    elif stage_a_checkpoint.exists():
        # A registered downstream pin may intentionally keep the branch point.
        required.append(stage_a_checkpoint)
    if identification:
        require_complete(root / "stage_id/training_report.json", "COMPLETE")
        required.extend(
            (
                root / "stage_id/best_checkpoint.pt",
                root / "stage_id/temperature_calibration.json",
                root / "stage_id/training_report.json",
            )
        )
        required.extend(selected_validation_paths(root / "stage_id"))
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required selected artifact missing: {path}")
    return required


def selected_artifacts() -> list[Path]:
    task = yaml.safe_load((PROJECT / "configs/task_experiments.yaml").read_text())
    ablations = yaml.safe_load((PROJECT / "configs/ablation_registry.yaml").read_text())
    bidirectional = yaml.safe_load(
        (PROJECT / "configs/bidirectional_identification.yaml").read_text()
    )
    external = yaml.safe_load(
        (PROJECT / "configs/external_model_baselines.yaml").read_text()
    )
    optimizer_pilot = yaml.safe_load(
        (PROJECT / "configs/optimizer_pilot.yaml").read_text()
    )
    continuation_lr_pilot = yaml.safe_load(
        (PROJECT / "configs/continuation_lr_pilot.yaml").read_text()
    )
    paths: list[Path] = []

    for run_id in (
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
    ):
        root = PROJECT / "runs" / run_id / "1701"
        require_complete(root / "training_report.json", "COMPLETE_FIXED_BUDGET")
        paths.extend(
            (
                root / "final_checkpoint.pt",
                root / "training_report.json",
                root / "train_log.jsonl",
                root / "validation_log.jsonl",
            )
        )
        endpoint = root / "endpoint_validation.json"
        if endpoint.exists():
            paths.append(endpoint)

    dense_provenance = PROJECT / "reports/model/f_dense_300m_provenance_supplement.json"
    require_complete(dense_provenance, "COMPLETE_PROVENANCE_SUPPLEMENT")
    dense_payload = json.loads(dense_provenance.read_text(encoding="utf-8"))
    dense_hash_targets = {
        "final_checkpoint.pt": PROJECT / "runs/F_DENSE_300M/1701/final_checkpoint.pt",
        "last_checkpoint.pt": PROJECT / "runs/F_DENSE_300M/1701/last_checkpoint.pt",
    }
    for name, target in dense_hash_targets.items():
        if sha256_file(target) != dense_payload["checkpoint_sha256"][name]:
            raise RuntimeError(f"Dense provenance checkpoint hash mismatch: {target}")
    reconstructed_targets = {
        "training_config_sha256": PROJECT / "configs/foundation_300m.yaml",
        "packed_metadata_sha256": PROJECT / "data/pretraining/packed_1024/metadata.json",
        "tokenizer_sha256": PROJECT / "artifacts/tokenizers/frozen/tokenizer.json",
    }
    for field, target in reconstructed_targets.items():
        if sha256_file(target) != dense_payload["reconstructed_from_frozen_inputs"][field]:
            raise RuntimeError(f"Dense provenance input hash mismatch: {target}")
    paths.append(dense_provenance)

    continuation_selection = (
        PROJECT / "reports/model/continuation_lr_pilot_selection.json"
    )
    require_complete(continuation_selection, "COMPLETE_VALIDATION_ONLY")
    continuation_payload = json.loads(
        continuation_selection.read_text(encoding="utf-8")
    )
    paths.extend(
        (
            continuation_selection,
            PROJECT / "configs/continuation_lr_pilot.yaml",
            PROJECT
            / str(continuation_lr_pilot["rejected_trigger_run"]["config_snapshot"]),
            PROJECT
            / str(continuation_lr_pilot["rejected_trigger_run"]["path"])
            / "validation_log.jsonl",
        )
    )

    router_diagnostics = PROJECT / "reports/model/foundation_router_diagnostics.json"
    require_complete(router_diagnostics, "COMPLETE_VALIDATION_ONLY")
    paths.extend(
        (
            router_diagnostics,
            PROJECT / "reports/model/foundation_router_diagnostics.csv",
        )
    )
    task_preflight = PROJECT / "reports/model/task_model_preflight.json"
    require_complete(task_preflight, "PASS_VALIDATION_ONLY")
    paths.append(task_preflight)

    source_blind_audit = (
        PROJECT / "reports/model/source_blind_normalization_baseline_audit.json"
    )
    cross_task_firewall = (
        PROJECT / "reports/model/cross_task_input_firewall.json"
    )
    require_complete(cross_task_firewall, "PASS")
    cross_task_payload = json.loads(
        cross_task_firewall.read_text(encoding="utf-8")
    )
    if (
        cross_task_payload.get("protected_text_emitted") is not False
        or cross_task_payload.get("model_or_hyperparameter_selection_use") is not False
        or not cross_task_payload.get("tracks")
        or any(
            int(row.get("exact_overlap_rows_with_id_train", 1)) != 0
            or int(row.get("compact_overlap_rows_with_id_train", 1)) != 0
            or int(row.get("near_overlap_rows_with_id_train", 1)) != 0
            for row in cross_task_payload.get("tracks", [])
        )
    ):
        raise RuntimeError("Cross-task source-blind input firewall did not pass")
    identification_fusion = PROJECT / "reports/model/id_fusion_selection.json"
    normalization_fusion = (
        PROJECT / "reports/model/normalization_fusion_selection.json"
    )
    normalization_fusion_v2 = (
        PROJECT / "reports/model/normalization_fusion_selection_v2.json"
    )
    fusion_uncertainty = (
        PROJECT / "reports/model/development_fusion_uncertainty.json"
    )
    fusion_transfer = (
        PROJECT / "reports/model/development_fusion_architecture_transfer.json"
    )
    for report in (
        source_blind_audit,
        identification_fusion,
        normalization_fusion,
        normalization_fusion_v2,
        fusion_uncertainty,
        fusion_transfer,
    ):
        require_complete(report, "COMPLETE_VALIDATION_ONLY")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("test_data_access") is not False:
            raise RuntimeError(f"Development system selection accessed test data: {report}")
        if report != source_blind_audit and payload.get("source_blind") is not True:
            raise RuntimeError(f"Fusion selection is not source-blind: {report}")
        if report in (normalization_fusion, normalization_fusion_v2) and payload.get(
            "cross_validation_group_unit"
        ) != "semantic_group_id":
            raise RuntimeError(
                f"Normalization fusion did not hold out whole semantic groups: {report}"
            )
        if report in (
            identification_fusion,
            normalization_fusion,
            normalization_fusion_v2,
            fusion_uncertainty,
        ) and payload.get("confirmatory_inference") is not False:
            raise RuntimeError(
                f"Development fusion evidence must be marked exploratory: {report}"
            )
        if report == fusion_transfer and (
            int(payload.get("completed_runs", 0)) != 6
            or int(payload.get("expected_runs", 0)) != 6
        ):
            raise RuntimeError("Fixed fusion architecture transfer is incomplete")
    fusion_artifacts = (
        PROJECT / "artifacts/baselines/id_char_tfidf_svm.pkl",
        PROJECT / "artifacts/baselines/normalization_word_rewrite.pkl",
        PROJECT / "artifacts/fusion/normalization_selector_v1.pkl",
        PROJECT / "artifacts/fusion/normalization_selector_v2.pkl",
    )
    paths.extend(
        (
            cross_task_firewall,
            source_blind_audit,
            PROJECT / "reports/model/source_blind_normalization_baselines.csv",
            identification_fusion,
            PROJECT / "reports/model/id_fusion_alpha_grid.csv",
            PROJECT / "reports/model/id_fusion_per_run_grid.csv",
            normalization_fusion,
            PROJECT / "reports/model/normalization_fusion_selection_grid.csv",
            PROJECT / "reports/model/normalization_fusion_oof_rows.parquet",
            normalization_fusion_v2,
            PROJECT / "reports/model/normalization_fusion_selection_v2_grid.csv",
            PROJECT / "reports/model/normalization_fusion_v2_oof_rows.parquet",
            fusion_uncertainty,
            PROJECT / "reports/model/development_fusion_uncertainty.csv",
            PROJECT / "reports/model/DEVELOPMENT_FUSION_UNCERTAINTY.md",
            fusion_transfer,
            PROJECT / "reports/model/development_fusion_architecture_transfer.csv",
            *fusion_artifacts,
        )
    )
    paths.extend(
        sorted(
            path
            for root in (
                PROJECT / "predictions/development_source_blind",
                PROJECT / "metrics/development_source_blind",
                PROJECT / "predictions/development_fusion",
            )
            for path in root.rglob("*")
            if path.is_file()
        )
    )
    retention_selection = (
        PROJECT / "reports/model/stage_s_retention_pilot_selection.json"
    )
    require_complete(retention_selection, "COMPLETE_VALIDATION_ONLY")
    retention_payload = json.loads(
        retention_selection.read_text(encoding="utf-8")
    )
    if retention_payload.get("test_data_access") is not False:
        raise RuntimeError("Stage-S retention selection was not validation-only")
    retention_config = PROJECT / "configs/stage_s_retention_pilot.yaml"
    rejected_task_config = (
        PROJECT / "configs/task_experiments_rejected_stage_s_default.yaml"
    )
    retention_root = PROJECT / "runs/pilots" / str(
        retention_payload["protocol_id"]
    )
    paths.extend(
        (
            retention_selection,
            retention_config,
            rejected_task_config,
            retention_root / "baseline_replay.json",
        )
    )
    paths.extend(
        task_paths(
            protocol="boichitro_stage_s_default_pilot_v1",
            variant="M0",
            suffix="base",
            seed=1701,
            identification=False,
        )
    )
    for candidate in retention_payload["candidates"]:
        root = PROJECT / str(candidate["run_dir"])
        if candidate["source"] != "registered_candidate":
            for validation in candidate["validation_curve"]:
                metrics_path = PROJECT / str(validation["metrics_path"])
                step = int(validation["optimizer_step"])
                paths.extend(
                    (
                        metrics_path,
                        root / f"validation_predictions_epoch_{step:02d}.parquet",
                        root / f"validation_by_dialect_epoch_{step:02d}.csv",
                    )
                )
            continue
        require_complete(root / "training_report.json", "COMPLETE")
        paths.extend(
            (
                root / "candidate_contract.json",
                root / "trainer_config.json",
                root / "mixture_report.json",
                root / "train_log.jsonl",
                root / "training_report.json",
                root / "best_checkpoint.pt",
                root / "best_selection.json",
            )
        )
        for validation in candidate["validation_curve"]:
            metrics_path = PROJECT / str(validation["metrics_path"])
            step = int(validation["optimizer_step"])
            paths.extend(
                (
                    metrics_path,
                    root / f"validation_predictions_epoch_{step:02d}.parquet",
                    root / f"validation_by_dialect_epoch_{step:02d}.csv",
                )
            )
    for candidate in continuation_payload["candidates"]:
        candidate_paths = [
            PROJECT / str(candidate[field])
            for field in (
                "resolved_config",
                "validation_log",
                "training_report",
                "final_checkpoint",
            )
        ]
        require_complete(candidate_paths[2], "COMPLETE_FIXED_BUDGET")
        paths.extend(candidate_paths)

    upcycling_selection = PROJECT / "reports/model/upcycling_strategy_selection.json"
    require_complete(upcycling_selection, "COMPLETE_VALIDATION_ONLY")
    upcycling_payload = json.loads(upcycling_selection.read_text(encoding="utf-8"))
    if upcycling_payload.get("test_data_access") != "forbidden_and_not_accessed":
        raise RuntimeError("Upcycling selection was not validation-only")
    paths.extend(
        (
            upcycling_selection,
            PROJECT / "configs/upcycling_strategy_selection.yaml",
            PROJECT / "reports/model/rejected_abrupt_bank_release.json",
            PROJECT / "reports/model/rejected_abrupt_bank_release_config.yaml",
            PROJECT
            / "runs/aborted/U_M2_STANDARD_MOE_200M_abrupt_bank_release_20260720/1701/train_log.jsonl",
            PROJECT
            / "runs/aborted/U_M2_STANDARD_MOE_200M_abrupt_bank_release_20260720/1701/validation_log.jsonl",
        )
    )

    switch_selection = PROJECT / "reports/model/switch_router_selection.json"
    require_complete(switch_selection, "COMPLETE_VALIDATION_ONLY")
    switch_payload = json.loads(switch_selection.read_text(encoding="utf-8"))
    if switch_payload.get("test_data_access") != "forbidden_and_not_accessed":
        raise RuntimeError("Switch-router selection was not validation-only")
    paths.extend(
        (
            switch_selection,
            PROJECT / "configs/switch_router_selection.yaml",
            PROJECT / "reports/model/rejected_switch_zero_router.json",
            PROJECT / "reports/model/rejected_switch_zero_router_config.yaml",
            PROJECT
            / "runs/aborted/U_M1_SWITCH_200M_zero_router_collapse_20260720/1701/train_log.jsonl",
            PROJECT
            / "runs/aborted/U_M1_SWITCH_200M_zero_router_collapse_20260720/1701/validation_log.jsonl",
        )
    )

    for variant in task["variants"]:
        for seed in task["seeds"]:
            paths.extend(
                task_paths(
                    protocol=str(task["protocol_id"]),
                    variant=str(variant),
                    suffix="base",
                    seed=int(seed),
                    identification=True,
                )
            )

    core = ablations["core_factorial"]
    factors = list(core["factors"].values())
    for enabled in itertools.product((False, True), repeat=len(factors)):
        removed = sorted(factor for factor, present in zip(factors, enabled) if not present)
        suffix = "base" if not removed else "__".join(removed)
        paths.extend(
            task_paths(
                protocol=str(core["protocol_id"]),
                variant=str(core["variant"]),
                suffix=suffix,
                seed=int(core["seed"]),
                identification=False,
            )
        )

    for section_name in ("confirmatory", "optimization"):
        section = ablations[section_name]
        for removed in section["ablations"]:
            suffix = "__".join(sorted(removed))
            for seed in section["seeds"]:
                paths.extend(
                    task_paths(
                        protocol=str(section["protocol_id"]),
                        variant=str(section["variant"]),
                        suffix=suffix,
                        seed=int(seed),
                        identification=True,
                    )
                )

    for learning_rate in optimizer_pilot["adamw_learning_rates"]:
        tag = f"{float(learning_rate):.0e}".replace("-", "m").replace("+", "p")
        paths.extend(
            task_paths(
                protocol=f"{optimizer_pilot['protocol_prefix']}_{tag}",
                variant=str(optimizer_pilot["variant"]),
                suffix="adamw_only",
                seed=int(optimizer_pilot["seed"]),
                identification=False,
            )
        )
    optimizer_selection = PROJECT / "reports/model/optimizer_pilot_selection.json"
    require_complete(optimizer_selection, "COMPLETE_VALIDATION_ONLY")
    paths.append(optimizer_selection)

    for seed in bidirectional["seeds"]:
        root = (
            PROJECT
            / "runs/task"
            / str(bidirectional["protocol_id"])
            / str(bidirectional["variant_id"])
            / str(seed)
        )
        require_complete(root / "stage_mntp/training_report.json", "COMPLETE")
        require_complete(root / "stage_id/training_report.json", "COMPLETE")
        require_complete(root / "run_manifest.json", "COMPLETE_VALIDATION_ONLY")
        paths.extend(
            (
                root / "stage_mntp/last_checkpoint.pt",
                root / "stage_mntp/mixture_report.json",
                root / "stage_mntp/training_report.json",
                root / "stage_id/best_checkpoint.pt",
                root / "stage_id/temperature_calibration.json",
                root / "stage_id/training_report.json",
                root / "contrastive_coverage_report.json",
                root / "initialization_report.json",
                root / "run_manifest.json",
            )
        )
        paths.extend(selected_validation_paths(root / "stage_id"))

    for task_name, task_values in (
        ("normalization", external["normalization"]),
        ("identification", external["identification"]),
    ):
        for model_name in task_values["models"]:
            for seed in external["seeds"]:
                root = PROJECT / "runs/external" / task_name / str(model_name) / str(seed)
                require_complete(root / "training_report.json", "COMPLETE_VALIDATION_ONLY")
                model_files = sorted(
                    path for path in (root / "best_model").rglob("*") if path.is_file()
                )
                if not model_files:
                    raise FileNotFoundError(f"Frozen external model missing: {root / 'best_model'}")
                paths.extend(model_files)
                paths.append(root / "training_report.json")
                selection_path = root / "best_selection.json"
                paths.append(selection_path)
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                selected_predictions = (
                    root
                    / f"validation_predictions_epoch_{int(selection['epoch']):02d}.parquet"
                )
                paths.append(selected_predictions)
                calibration = root / "temperature_calibration.json"
                if calibration.exists():
                    paths.append(calibration)

    inference = PROJECT / "reports/model/task_inference_benchmark.json"
    if not inference.exists():
        raise FileNotFoundError(f"Systems benchmark missing: {inference}")
    paths.append(inference)

    paths.extend(sorted((PROJECT / "data/final/v1").glob("*.parquet")))
    paths.extend(
        (
            PROJECT / "artifacts/tokenizers/frozen/tokenizer.json",
            PROJECT / "cache/tasks/maps.json",
            PROJECT / "data/final/v1/DATASET_CARD.md",
            PROJECT / "reports/EVALUATION_TRACK_CONTRACT.md",
            PROJECT / "reports/PRIOR_TEST_ACCESS_DISCLOSURE.md",
            PROJECT / "reports/final_dataset_report.json",
            PROJECT / "reports/tokenizer/TOKENIZER_FREEZE_REPORT.json",
        )
    )
    unique = sorted(set(paths), key=lambda path: path.relative_to(PROJECT).as_posix())
    missing = [path for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Freeze inputs missing: {missing[:5]}")
    return unique


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    for locked_output in (
        PROJECT / "predictions" / args.protocol_id,
        PROJECT / "predictions/locked_external_test_v1",
    ):
        if locked_output.exists() and any(locked_output.rglob("evaluation_manifest.json")):
            raise RuntimeError(
                f"Main locked neural outputs already exist before freeze: {locked_output}"
            )
    fingerprint = protocol_fingerprints(PROJECT)
    artifacts = file_manifest(PROJECT, selected_artifacts())
    payload = {
        "status": "FROZEN",
        "protocol_id": args.protocol_id,
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **fingerprint,
        "selected_artifacts_sha256": manifest_sha256(artifacts),
        "selected_artifacts": artifacts,
        "prior_test_access": {
            "fixed_classical_baseline_metrics": True,
            "tiny_pipeline_smoke_samples": True,
            "main_neural_checkpoint_metrics": False,
            "used_for_main_neural_selection": False,
        },
        "main_neural_test_metrics_computed_before_freeze": False,
        "test_files_hash_only_at_freeze": True,
    }
    path = freeze_manifest_path(PROJECT, args.protocol_id)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        comparable = dict(payload)
        comparable.pop("frozen_at")
        previous_comparable = dict(previous)
        previous_comparable.pop("frozen_at", None)
        if previous_comparable != comparable:
            raise RuntimeError(
                f"Existing immutable freeze differs; use a new protocol id: {path}"
            )
        print(f"Protocol already frozen identically: {path}")
        return
    atomic_json(path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "protocol_id": args.protocol_id,
                "code_sha256": payload["code_sha256"],
                "config_sha256": payload["config_sha256"],
                "selected_artifacts_sha256": payload["selected_artifacts_sha256"],
                "artifact_count": len(artifacts),
                "manifest": str(path.relative_to(PROJECT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
