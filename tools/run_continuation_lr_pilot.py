#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a stable mature-checkpoint continuation LR on validation only."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/continuation_lr_pilot.yaml"
    )
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_validation(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    args = parse_args()
    pilot = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_path = PROJECT / str(pilot["base_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    resolved_root = PROJECT / "reports/model/continuation_lr_pilot_configs"
    candidates = []

    for candidate in pilot["candidates"]:
        candidate_id = str(candidate["id"])
        resolved = copy.deepcopy(base)
        resolved.update(
            run_id=f"P_CONT_LR_{candidate_id.upper()}_10M",
            token_budget=int(pilot["token_budget"]),
            scheduler_token_budget=int(pilot["scheduler_token_budget"]),
            eval_every_tokens=int(pilot["eval_every_tokens"]),
            checkpoint_every_tokens=int(pilot["checkpoint_every_tokens"]),
            warmup_fraction=float(pilot["warmup_fraction"]),
            development_only=True,
            test_data_access="forbidden",
        )
        resolved.pop("optimizer_selection_report", None)
        resolved["optimizer"] = dict(resolved["optimizer"])
        resolved["optimizer"]["muon_lr"] = float(candidate["muon_lr"])
        resolved["optimizer"]["adamw_lr"] = float(candidate["adamw_lr"])
        resolved_path = resolved_root / f"{candidate_id}.yaml"
        serialized = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
        if resolved_path.exists() and resolved_path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"Resolved pilot config changed: {resolved_path}")
        atomic_text(resolved_path, serialized)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "tools/train_foundation.py"),
                "--config",
                str(resolved_path),
                "--resume",
            ],
            cwd=PROJECT,
            check=True,
            env={**os.environ, "PYTHONPATH": str(PROJECT / "src")},
        )
        run_root = PROJECT / "runs" / resolved["run_id"] / str(pilot["seed"])
        report_path = run_root / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE_FIXED_BUDGET":
            raise RuntimeError(f"Incomplete continuation LR candidate: {report_path}")
        validation_path = run_root / "validation_log.jsonl"
        validations = load_validation(validation_path)
        initial = next(row for row in validations if int(row["tokens_seen"]) == 0)
        post = [row for row in validations if int(row["tokens_seen"]) > 0]
        final = max(post, key=lambda row: int(row["tokens_seen"]))
        initial_bpc = float(initial["validation_bpc"])
        final_bpc = float(final["validation_bpc"])
        maximum_bpc = max(float(row["validation_bpc"]) for row in post)
        final_regression = final_bpc / initial_bpc - 1.0
        maximum_regression = maximum_bpc / initial_bpc - 1.0
        eligible = (
            final_regression <= float(pilot["final_relative_bpc_regression_max"])
            and maximum_regression
            <= float(pilot["intermediate_relative_bpc_regression_max"])
        )
        candidates.append(
            {
                "id": candidate_id,
                "muon_lr": float(candidate["muon_lr"]),
                "adamw_lr": float(candidate["adamw_lr"]),
                "initial_validation_bpc": initial_bpc,
                "final_validation_bpc": final_bpc,
                "maximum_post_start_validation_bpc": maximum_bpc,
                "final_relative_bpc_regression": final_regression,
                "maximum_relative_bpc_regression": maximum_regression,
                "eligible": eligible,
                "resolved_config": str(resolved_path.relative_to(PROJECT)),
                "validation_log": str(validation_path.relative_to(PROJECT)),
                "training_report": str(report_path.relative_to(PROJECT)),
                "final_checkpoint": str(
                    (run_root / "final_checkpoint.pt").relative_to(PROJECT)
                ),
            }
        )

    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError("No continuation LR candidate passed the validation stability guard")
    selected = sorted(
        eligible,
        key=lambda row: (row["final_validation_bpc"], row["muon_lr"]),
    )[0]

    trigger = pilot["rejected_trigger_run"]
    trigger_validation = PROJECT / str(trigger["path"]) / "validation_log.jsonl"
    trigger_rows = load_validation(trigger_validation)
    trigger_initial = next(row for row in trigger_rows if int(row["tokens_seen"]) == 0)
    trigger_final = max(trigger_rows, key=lambda row: int(row["tokens_seen"]))
    report = {
        "status": "COMPLETE_VALIDATION_ONLY",
        "protocol_id": pilot["protocol_id"],
        "selection_metric": pilot["selection_metric"],
        "selection_mode": pilot["selection_mode"],
        "selected_candidate": selected["id"],
        "selected_muon_lr": selected["muon_lr"],
        "selected_adamw_lr": selected["adamw_lr"],
        "scheduler_token_budget": int(pilot["scheduler_token_budget"]),
        "warmup_fraction": float(pilot["warmup_fraction"]),
        "candidates": candidates,
        "rejected_high_lr_trigger": {
            "path": str(trigger["path"]),
            "config_snapshot": str(trigger["config_snapshot"]),
            "muon_lr": float(trigger["muon_lr"]),
            "adamw_lr": float(trigger["adamw_lr"]),
            "warmup_fraction": float(trigger["warmup_fraction"]),
            "initial_validation_bpc": float(trigger_initial["validation_bpc"]),
            "post_restart_validation_bpc": float(trigger_final["validation_bpc"]),
            "relative_bpc_regression": float(trigger_final["validation_bpc"])
            / float(trigger_initial["validation_bpc"])
            - 1.0,
        },
        "test_data_access": False,
    }
    output = PROJECT / "reports/model/continuation_lr_pilot_selection.json"
    atomic_text(output, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
