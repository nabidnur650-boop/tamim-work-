#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SCORES = (
    "dialect_authenticity_1_to_5",
    "target_adequacy_1_to_5",
    "target_fluency_1_to_5",
)
BINARY = ("label_correct_yes_no", "unsafe_or_pii_yes_no")
REQUIRED = ("row_id", "source_id", "dialect", "reviewer_id", *SCORES, *BINARY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the external native-speaker data review."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/native_review.yaml"
    )
    return parser.parse_args()


def yes_no(series: pd.Series, name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"yes": True, "y": True, "1": True, "no": False, "n": False, "0": False}
    invalid = sorted(set(normalized) - set(allowed))
    if invalid:
        raise ValueError(f"Invalid yes/no values in {name}: {invalid}")
    return normalized.map(allowed).astype(bool)


def analyze(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    missing_columns = sorted(set(REQUIRED) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Native review sheet lacks columns: {missing_columns}")
    completion_columns = ["reviewer_id", *SCORES, *BINARY]
    complete = frame[completion_columns].apply(
        lambda column: column.notna() & column.astype(str).str.strip().ne("")
    )
    completed_rows = int(complete.all(axis=1).sum())
    base = {
        "protocol_id": config["protocol_id"],
        "rows": len(frame),
        "completed_rows": completed_rows,
        "completion_fraction": completed_rows / max(1, len(frame)),
    }
    if completed_rows != len(frame):
        return {"status": "PENDING_INCOMPLETE", **base}, pd.DataFrame()

    reviewed = frame.copy()
    for column in SCORES:
        reviewed[column] = pd.to_numeric(reviewed[column], errors="raise")
        if not reviewed[column].between(1, 5).all():
            raise ValueError(f"Scores outside 1–5 in {column}")
    reviewed["label_correct"] = yes_no(reviewed["label_correct_yes_no"], "label_correct_yes_no")
    reviewed["unsafe_or_pii"] = yes_no(reviewed["unsafe_or_pii_yes_no"], "unsafe_or_pii_yes_no")

    rows = []
    for dialect, group in reviewed.groupby("dialect", sort=True, observed=True):
        rows.append(
            {
                "dialect": str(dialect),
                "rows": len(group),
                "mean_dialect_authenticity": float(group[SCORES[0]].mean()),
                "mean_target_adequacy": float(group[SCORES[1]].mean()),
                "mean_target_fluency": float(group[SCORES[2]].mean()),
                "label_accuracy": float(group["label_correct"].mean()),
                "unsafe_or_pii_flags": int(group["unsafe_or_pii"].sum()),
            }
        )
    by_dialect = pd.DataFrame(rows)
    thresholds = config["thresholds"]
    unique_reviewers = int(reviewed["reviewer_id"].astype(str).nunique())
    metrics = {
        "unique_reviewers": unique_reviewers,
        "overall_mean_dialect_authenticity": float(reviewed[SCORES[0]].mean()),
        "overall_mean_target_adequacy": float(reviewed[SCORES[1]].mean()),
        "overall_mean_target_fluency": float(reviewed[SCORES[2]].mean()),
        "overall_label_accuracy": float(reviewed["label_correct"].mean()),
        "unsafe_or_pii_flags": int(reviewed["unsafe_or_pii"].sum()),
        "worst_dialect_mean_dialect_authenticity": float(
            by_dialect["mean_dialect_authenticity"].min()
        ),
        "worst_dialect_mean_target_adequacy": float(
            by_dialect["mean_target_adequacy"].min()
        ),
        "worst_dialect_mean_target_fluency": float(
            by_dialect["mean_target_fluency"].min()
        ),
        "worst_dialect_label_accuracy": float(by_dialect["label_accuracy"].min()),
    }
    checks = {
        "minimum_unique_reviewers": unique_reviewers
        >= int(config["minimum_unique_reviewers"]),
        "overall_mean_dialect_authenticity": metrics[
            "overall_mean_dialect_authenticity"
        ]
        >= float(thresholds["overall_mean_dialect_authenticity"]),
        "overall_mean_target_adequacy": metrics["overall_mean_target_adequacy"]
        >= float(thresholds["overall_mean_target_adequacy"]),
        "overall_mean_target_fluency": metrics["overall_mean_target_fluency"]
        >= float(thresholds["overall_mean_target_fluency"]),
        "overall_label_accuracy": metrics["overall_label_accuracy"]
        >= float(thresholds["overall_label_accuracy"]),
        "per_dialect_mean_dialect_authenticity": metrics[
            "worst_dialect_mean_dialect_authenticity"
        ]
        >= float(thresholds["per_dialect_mean_dialect_authenticity"]),
        "per_dialect_mean_target_adequacy": metrics[
            "worst_dialect_mean_target_adequacy"
        ]
        >= float(thresholds["per_dialect_mean_target_adequacy"]),
        "per_dialect_mean_target_fluency": metrics[
            "worst_dialect_mean_target_fluency"
        ]
        >= float(thresholds["per_dialect_mean_target_fluency"]),
        "per_dialect_label_accuracy": metrics["worst_dialect_label_accuracy"]
        >= float(thresholds["per_dialect_label_accuracy"]),
        "maximum_unsafe_or_pii_flags": metrics["unsafe_or_pii_flags"]
        <= int(thresholds["maximum_unsafe_or_pii_flags"]),
    }
    passed = all(checks.values())
    return (
        {
            "status": "PASS_NATIVE_REVIEW" if passed else "FAIL_NATIVE_REVIEW",
            **base,
            "metrics": metrics,
            "thresholds": thresholds,
            "checks": checks,
            "failure_policy": config["policy_on_failure"],
        },
        by_dialect,
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_csv(PROJECT / config["input"], keep_default_na=False)
    report, by_dialect = analyze(frame, config)
    output = PROJECT / "reports/native_review_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not by_dialect.empty:
        by_dialect.to_csv(PROJECT / "reports/native_review_by_dialect.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
