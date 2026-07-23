#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit full-continuation router stability without test data."
    )
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        default=[],
        help="Registered continuation run; repeat for multiple runs.",
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--cv-threshold", type=float, default=0.5)
    parser.add_argument("--stability-start-fraction", type=float, default=0.10)
    parser.add_argument("--persistent-consecutive-logs", type=int, default=3)
    parser.add_argument("--persistent-fraction", type=float, default=0.05)
    return parser.parse_args()


def longest_true_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def main() -> None:
    args = parse_args()
    run_ids = args.run_ids or ["U_M1_SWITCH_200M", "U_M2_STANDARD_MOE_200M"]
    rows = []
    for run_id in run_ids:
        root = PROJECT / "runs" / run_id / str(args.seed)
        report_path = root / "training_report.json"
        train_path = root / "train_log.jsonl"
        validation_path = root / "validation_log.jsonl"
        if not all(path.exists() for path in (report_path, train_path, validation_path)):
            raise FileNotFoundError(f"Incomplete routing audit inputs under {root}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE_FIXED_BUDGET":
            raise RuntimeError(f"Continuation is not complete: {report_path}")
        train = pd.read_json(train_path, lines=True)
        validation = pd.read_json(validation_path, lines=True)
        required = {"router_load_cv", "router_entropy", "router_z_loss"}
        if not required.issubset(train.columns):
            raise ValueError(f"Missing router diagnostics for {run_id}: {required - set(train)}")
        cv = train["router_load_cv"].astype(float)
        stability_start_tokens = int(
            int(report["tokens_seen"]) * float(args.stability_start_fraction)
        )
        stable = train.loc[train["tokens_seen"].ge(stability_start_tokens)].copy()
        stable_cv = stable["router_load_cv"].astype(float)
        above = stable_cv.gt(float(args.cv_threshold)).tolist()
        finite_columns = [
            column
            for column in (
                "loss",
                "lm_loss",
                "mtp_loss",
                "gradient_norm",
                "router_load_cv",
                "router_entropy",
                "router_z_loss",
            )
            if column in train
        ]
        nonfinite = int(
            sum(
                not math.isfinite(float(value))
                for column in finite_columns
                for value in train[column]
            )
        )
        validation_bpc = validation["validation_bpc"].astype(float).tolist()
        regressions = sum(
            current > previous + 1e-12
            for previous, current in zip(validation_bpc, validation_bpc[1:])
        )
        longest = longest_true_run(above)
        fraction = float(sum(above) / max(1, len(above)))
        persistent = bool(
            longest >= int(args.persistent_consecutive_logs)
            or fraction >= float(args.persistent_fraction)
        )
        rows.append(
            {
                "run_id": run_id,
                "architecture": report["architecture"],
                "tokens_seen": int(report["tokens_seen"]),
                "logged_points": len(train),
                "router_load_cv_mean": float(cv.mean()),
                "router_load_cv_max": float(cv.max()),
                "router_load_cv_final": float(cv.iloc[-1]),
                "cv_threshold": float(args.cv_threshold),
                "stability_start_tokens": stability_start_tokens,
                "stability_logged_points": len(stable),
                "cv_threshold_exceedances": int(sum(above)),
                "cv_threshold_exceedance_fraction": fraction,
                "longest_consecutive_cv_exceedance": longest,
                "router_entropy_min": float(train["router_entropy"].min()),
                "router_entropy_final": float(train["router_entropy"].iloc[-1]),
                "router_z_loss_max": float(train["router_z_loss"].max()),
                "nonfinite_diagnostic_values": nonfinite,
                "validation_points": len(validation),
                "validation_bpc_regressions": int(regressions),
                "initial_validation_bpc": validation_bpc[0],
                "final_validation_bpc": validation_bpc[-1],
                "persistent_expert_collapse": persistent,
            }
        )
    frame = pd.DataFrame(rows)
    output_dir = PROJECT / "reports/model"
    frame.to_csv(output_dir / "foundation_router_diagnostics.csv", index=False)
    payload = {
        "status": "COMPLETE_VALIDATION_ONLY",
        "test_data_access": False,
        "guards": {
            "router_load_cv_threshold": float(args.cv_threshold),
            "stability_start_fraction": float(args.stability_start_fraction),
            "persistent_consecutive_logs": int(args.persistent_consecutive_logs),
            "persistent_exceedance_fraction": float(args.persistent_fraction),
            "nonfinite_values_allowed": 0,
        },
        "all_runs_pass_nonfinite_guard": bool(
            frame["nonfinite_diagnostic_values"].eq(0).all()
        ),
        "all_runs_pass_persistent_collapse_guard": bool(
            ~frame["persistent_expert_collapse"].any()
        ),
        "runs": rows,
    }
    (output_dir / "foundation_router_diagnostics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
