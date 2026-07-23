#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select an AdamW-only learning rate using validation data only."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/optimizer_pilot.yaml"
    )
    return parser.parse_args()


def learning_rate_tag(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    candidates = []
    for learning_rate in config["adamw_learning_rates"]:
        value = float(learning_rate)
        protocol = f"{config['protocol_prefix']}_{learning_rate_tag(value)}"
        command = [
            sys.executable,
            str(PROJECT / "tools/run_task_experiments.py"),
            "--protocol-id",
            protocol,
            "--variants",
            str(config["variant"]),
            "--seeds",
            str(config["seed"]),
            "--stages",
            "a",
            "s",
            "--ablation",
            "adamw_only",
            "--adamw-learning-rate",
            str(value),
            "--stage-a-token-budget",
            str(config["stage_a_token_budget"]),
            "--stage-s-token-budget",
            str(config["stage_s_token_budget"]),
            "--normalization-validation-limit",
            str(config["normalization_validation_limit"]),
            "--replay-validation-examples",
            str(config["replay_validation_examples"]),
            "--stage-s-validation-checkpoints",
            str(config["stage_s_validation_checkpoints"]),
        ]
        subprocess.run(
            command,
            cwd=PROJECT,
            check=True,
            env={**os.environ, "PYTHONPATH": str(PROJECT / "src")},
        )
        run_root = (
            PROJECT
            / "runs/task"
            / protocol
            / f"{config['variant']}__adamw_only"
            / str(config["seed"])
        )
        selection_path = run_root / "stage_s/best_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        candidates.append(
            {
                "adamw_learning_rate": value,
                "protocol_id": protocol,
                "selection_value": float(selection["selection_value"]),
                "macro_chrfpp": float(selection["validation"]["macro_chrfpp"]),
                "replay_guard_pass": bool(selection["validation"]["replay_guard_pass"]),
                "selection_manifest": str(selection_path.relative_to(PROJECT)),
            }
        )
    eligible = [row for row in candidates if row["replay_guard_pass"]]
    if not eligible:
        raise RuntimeError("No AdamW learning-rate candidate passed the replay guard")
    selected = sorted(
        eligible,
        key=lambda row: (-row["selection_value"], row["adamw_learning_rate"]),
    )[0]
    report = {
        "status": "COMPLETE_VALIDATION_ONLY",
        "selection_metric": config["selection_metric"],
        "selection_mode": config["selection_mode"],
        "selected_adamw_learning_rate": selected["adamw_learning_rate"],
        "selected_protocol_id": selected["protocol_id"],
        "candidates": candidates,
        "test_data_access": False,
    }
    path = PROJECT / "reports/model/optimizer_pilot_selection.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
