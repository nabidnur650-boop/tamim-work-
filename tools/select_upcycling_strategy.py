#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs/upcycling_strategy_selection.yaml"
OUTPUT = PROJECT / "reports/model/upcycling_strategy_selection.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if protocol.get("test_data_access") != "forbidden":
        raise RuntimeError("Upcycling selection must remain validation-only")

    seed = int(protocol["seed"])
    conditions: dict[str, dict[str, object]] = {}
    for name, specification in protocol["conditions"].items():
        run_id = str(specification["run_id"])
        root = PROJECT / "runs" / run_id / str(seed)
        report_path = root / "training_report.json"
        validation_path = root / "validation_log.jsonl"
        if not report_path.exists() or not validation_path.exists():
            raise FileNotFoundError(f"Incomplete upcycling condition: {run_id}")
        training_report = json.loads(report_path.read_text(encoding="utf-8"))
        if training_report.get("status") != "COMPLETE_FIXED_BUDGET":
            raise RuntimeError(f"Upcycling condition is not complete: {run_id}")
        trajectory = jsonl(validation_path)
        if len(trajectory) < 2:
            raise RuntimeError(f"Upcycling trajectory is incomplete: {run_id}")
        initial = float(trajectory[0]["validation_bpc"])
        endpoint = dict(training_report["final_validation"])
        endpoint.update(
            global_step=int(training_report["optimizer_steps"]),
            tokens_seen=int(training_report["tokens_seen"]),
            evaluation="exact_final_training_progress",
        )
        final = float(endpoint["validation_bpc"])
        guard_points = [*trajectory, endpoint]
        changes = [
            100.0 * (float(point["validation_bpc"]) - initial) / initial
            for point in guard_points
        ]
        conditions[name] = {
            "run_id": run_id,
            "role": specification["role"],
            "tokens_seen": int(training_report["tokens_seen"]),
            "initial_validation_bpc": initial,
            "final_validation_bpc": final,
            "final_change_from_own_initial_percent": changes[-1],
            "maximum_transient_change_from_own_initial_percent": max(changes),
            "minimum_change_from_own_initial_percent": min(changes),
            "validation_points": trajectory,
            "exact_endpoint_validation": endpoint,
            "training_config_sha256": training_report["training_config_sha256"],
            "training_report_sha256": sha256(report_path),
            "validation_log_sha256": sha256(validation_path),
            "model_overrides": specification.get("model_overrides"),
        }

    baseline_name = str(protocol["baseline_condition"])
    baseline = float(conditions[baseline_name]["initial_validation_bpc"])
    guards = protocol["guards"]
    transient_limit = float(
        guards["maximum_transient_validation_bpc_regression_percent"]
    )
    final_limit = float(guards["maximum_final_validation_bpc_regression_percent"])
    initial_tolerance = float(guards["initial_bpc_match_absolute_tolerance"])
    eligible: list[str] = []
    for name, result in conditions.items():
        if result["role"] != "candidate":
            result["eligible"] = False
            result["eligibility_reasons"] = ["registered_negative_control"]
            continue
        initial_matches = (
            abs(float(result["initial_validation_bpc"]) - baseline)
            <= initial_tolerance
        )
        transient_regression = 100.0 * (
            max(
                *(
                    float(point["validation_bpc"])
                    for point in result["validation_points"]
                ),
                float(result["exact_endpoint_validation"]["validation_bpc"]),
            )
            - baseline
        ) / baseline
        final_regression = 100.0 * (
            float(result["final_validation_bpc"]) - baseline
        ) / baseline
        reasons = []
        if not initial_matches:
            reasons.append("initial_bpc_does_not_match_dense_equivalent_baseline")
        if transient_regression > transient_limit:
            reasons.append("transient_regression_guard_failed")
        if final_regression > final_limit:
            reasons.append("final_regression_guard_failed")
        result["transient_change_from_dense_equivalent_percent"] = transient_regression
        result["final_change_from_dense_equivalent_percent"] = final_regression
        result["eligible"] = not reasons
        result["eligibility_reasons"] = reasons or ["all_validation_guards_passed"]
        if not reasons:
            eligible.append(name)

    selected = (
        min(
            eligible,
            key=lambda name: float(conditions[name]["final_validation_bpc"]),
        )
        if eligible
        else None
    )
    payload = {
        "status": (
            "COMPLETE_VALIDATION_ONLY" if selected else "NO_STABLE_CANDIDATE"
        ),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(CONFIG),
        "test_data_access": "forbidden_and_not_accessed",
        "seed": seed,
        "dense_equivalent_baseline_validation_bpc": baseline,
        "guards": guards,
        "selection_rule": protocol["selection"],
        "eligible_candidates": eligible,
        "selected_strategy": selected,
        "selected_run_id": conditions[selected]["run_id"] if selected else None,
        "selected_model_overrides": (
            conditions[selected]["model_overrides"] if selected else None
        ),
        "conditions": conditions,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if selected is None:
        raise SystemExit("No upcycling strategy passed the validation-only guards")


if __name__ == "__main__":
    main()
