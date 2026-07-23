#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs/switch_router_selection.yaml"
OUTPUT = PROJECT / "reports/model/switch_router_selection.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if protocol.get("test_data_access") != "forbidden":
        raise RuntimeError("Switch router selection must remain validation-only")
    seed = int(protocol["seed"])
    candidates: dict[str, dict[str, object]] = {}
    for name, specification in protocol["candidates"].items():
        run_id = str(specification["run_id"])
        root = PROJECT / "runs" / run_id / str(seed)
        report_path = root / "training_report.json"
        validation_path = root / "validation_log.jsonl"
        train_path = root / "train_log.jsonl"
        if not all(path.exists() for path in (report_path, validation_path, train_path)):
            raise FileNotFoundError(f"Incomplete Switch router candidate: {run_id}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE_FIXED_BUDGET":
            raise RuntimeError(f"Switch router candidate is incomplete: {run_id}")
        validation = read_jsonl(validation_path)
        training = read_jsonl(train_path)
        initial = float(validation[0]["validation_bpc"])
        final = float(report["final_validation"]["validation_bpc"])
        load_cvs = [float(point["router_load_cv"]) for point in training]
        candidates[name] = {
            "run_id": run_id,
            "tokens_seen": int(report["tokens_seen"]),
            "initial_validation_bpc": initial,
            "final_validation_bpc": final,
            "final_change_percent": 100.0 * (final - initial) / initial,
            "maximum_logged_router_load_cv": max(load_cvs),
            "minimum_logged_router_load_cv": min(load_cvs),
            "final_logged_router_load_cv": load_cvs[-1],
            "validation_points": validation,
            "upcycle_router_init_std": float(
                specification["upcycle_router_init_std"]
            ),
            "model_overrides": specification["model_overrides"],
            "training_config_sha256": report["training_config_sha256"],
            "training_report_sha256": sha256(report_path),
            "validation_log_sha256": sha256(validation_path),
            "train_log_sha256": sha256(train_path),
        }

    baseline = min(
        float(candidate["initial_validation_bpc"])
        for candidate in candidates.values()
    )
    guards = protocol["guards"]
    tolerance = float(guards["initial_bpc_match_absolute_tolerance"])
    final_limit = float(guards["maximum_final_validation_bpc_regression_percent"])
    cv_limit = float(guards["maximum_logged_router_load_cv"])
    eligible: list[str] = []
    for name, result in candidates.items():
        reasons = []
        if abs(float(result["initial_validation_bpc"]) - baseline) > tolerance:
            reasons.append("initial_bpc_mismatch")
        if float(result["final_change_percent"]) > final_limit:
            reasons.append("final_validation_regression_guard_failed")
        if float(result["maximum_logged_router_load_cv"]) > cv_limit:
            reasons.append("router_load_cv_guard_failed")
        result["eligible"] = not reasons
        result["eligibility_reasons"] = reasons or ["all_validation_guards_passed"]
        if not reasons:
            eligible.append(name)

    selected = (
        min(
            eligible,
            key=lambda name: float(candidates[name]["final_validation_bpc"]),
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
        "selected_run_id": candidates[selected]["run_id"] if selected else None,
        "selected_upcycle_router_init_std": (
            candidates[selected]["upcycle_router_init_std"] if selected else None
        ),
        "selected_model_overrides": (
            candidates[selected]["model_overrides"] if selected else None
        ),
        "candidates": candidates,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if selected is None:
        raise SystemExit("No Switch router candidate passed the validation guards")


if __name__ == "__main__":
    main()
