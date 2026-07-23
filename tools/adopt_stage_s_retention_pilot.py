#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.tokenization import sha256_file  # noqa: E402
from run_task_experiments import (  # noqa: E402
    stage_complete,
    validate_stage_s_schedule_contract,
)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def hardlink_verified(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Pilot adoption source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"Pilot adoption destination conflict: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".link.tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(destination)


def main() -> None:
    main_config_path = PROJECT / "configs/task_experiments.yaml"
    main_config = yaml.safe_load(main_config_path.read_text(encoding="utf-8"))
    contract = validate_stage_s_schedule_contract(main_config)
    if contract is None:
        raise RuntimeError("Main task config has no Stage-S selection contract")
    selection_path = PROJECT / str(contract["selection_report"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pilot_config = yaml.safe_load(
        (PROJECT / str(selection["config_path"])).read_text(encoding="utf-8")
    )
    variant = str(selection["development_variant"])
    seed = int(selection["development_seed"])
    if variant != "M0":
        raise RuntimeError("Registered Stage-S adoption is restricted to M0 development")
    default_config = yaml.safe_load(
        (
            PROJECT / "configs/task_experiments_rejected_stage_s_default.yaml"
        ).read_text(encoding="utf-8")
    )
    if main_config["stage_a"] != default_config["stage_a"]:
        raise RuntimeError("Main and pilot Stage-A contracts differ")
    if main_config["variants"][variant] != default_config["variants"][variant]:
        raise RuntimeError("Main and pilot initialization contracts differ")

    source_stage_a = PROJECT / str(selection["stage_a_checkpoint"])
    source_default_root = source_stage_a.parents[1]
    selected_summary = next(
        row
        for row in selection["candidates"]
        if row["candidate_id"] == selection["selected_candidate_id"]
    )
    source_stage_s = PROJECT / str(selected_summary["run_dir"])
    destination_root = (
        PROJECT
        / "runs/task"
        / str(main_config["protocol_id"])
        / f"{variant}__base"
        / str(seed)
    )
    manifest_path = destination_root / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "COMPLETE_VALIDATION_ONLY":
            if not stage_complete(
                destination_root / "stage_s", "best_checkpoint.pt", "best_selection.json"
            ):
                raise RuntimeError("Finalized adopted run has an incomplete Stage-S")
            print(f"Main pilot adoption already finalized: {destination_root}")
            return

    source_selection = json.loads(
        (source_stage_s / "best_selection.json").read_text(encoding="utf-8")
    )
    selected_step = int(selection["selected_validation"]["optimizer_step"])
    if int(source_selection["validation_id"]) != selected_step:
        raise RuntimeError("Selected pilot checkpoint step does not match the report")
    if not bool(source_selection["validation"]["replay_guard_pass"]):
        raise RuntimeError("Selected pilot checkpoint fails the replay guard")
    checkpoint = torch.load(
        source_stage_s / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    if int(checkpoint["global_step"]) != selected_step:
        raise RuntimeError("Selected pilot tensor is not the registered checkpoint")
    if sha256_file(source_stage_a) != str(selection["stage_a_sha256"]):
        raise RuntimeError("Pilot Stage-A checkpoint hash mismatch")

    stage_a_files = (
        "last_checkpoint.pt",
        "mixture_report.json",
        "train_log.jsonl",
        "training_report.json",
    )
    for name in stage_a_files:
        hardlink_verified(
            source_default_root / "stage_a" / name,
            destination_root / "stage_a" / name,
        )
    hardlink_verified(
        source_default_root / "initialization_report.json",
        destination_root / "initialization_report.json",
    )
    stage_s_files = [
        path
        for path in source_stage_s.iterdir()
        if path.is_file()
        and (
            path.name
            in {
                "best_checkpoint.pt",
                "best_selection.json",
                "mixture_report.json",
                "train_log.jsonl",
                "training_report.json",
                "trainer_config.json",
                "candidate_contract.json",
            }
            or path.name.startswith("validation_")
        )
    ]
    for source in stage_s_files:
        hardlink_verified(source, destination_root / "stage_s" / source.name)

    report = {
        "status": "ADOPTED_VALIDATION_SELECTED_DEVELOPMENT_BRANCH",
        "destination": str(destination_root.relative_to(PROJECT)),
        "variant": variant,
        "seed": seed,
        "stage_a_source": str(source_default_root.relative_to(PROJECT)),
        "stage_a_sha256": sha256_file(source_stage_a),
        "stage_s_source": str(source_stage_s.relative_to(PROJECT)),
        "stage_s_checkpoint_sha256": sha256_file(
            source_stage_s / "best_checkpoint.pt"
        ),
        "selection_report": str(selection_path.relative_to(PROJECT)),
        "selection_report_sha256": sha256_file(selection_path),
        "selected_candidate_id": str(selection["selected_candidate_id"]),
        "selected_validation": selection["selected_validation"],
        "pilot_validation_checkpoints": int(pilot_config["validation_checkpoints"]),
        "hardlink_reuse": True,
        "test_data_access": False,
    }
    atomic_json(report, destination_root / "pilot_adoption_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
