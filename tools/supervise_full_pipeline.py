#!/usr/bin/env python3
"""Keep the long registered pipeline alive without duplicating a live run."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
STATE_PATH = PROJECT / "reports/pipeline/full_pipeline_state.json"
SUPERVISOR_STATE = PROJECT / "reports/pipeline/supervisor_state.json"
STAGES = [
    "main_task",
    "development_system_fusion",
    "optimizer_pilot",
    "task_inference_benchmark",
    "bidirectional_id",
    "external_normalization",
    "external_identification",
    "factorial_pilot",
    "confirmatory_ablations",
    "optimization_ablations",
    "protocol_freeze",
    "locked_main",
    "locked_confirmatory_ablations",
    "locked_optimization_ablations",
    "locked_bidirectional_id",
    "locked_external",
    "human_evaluation_packets",
    "robustness",
    "routing_analysis",
    "statistics",
    "publication_bundle",
]
FINALIZERS = [
    "tools/summarize_development_results.py",
    "tools/make_q1_figure_suite.py",
    "tools/validate_q1_bundle.py",
    "tools/make_paper_tables.py",
    "tools/build_q1_manuscript.py",
    "tools/capture_environment.py",
    "tools/audit_q1_readiness.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-restarts", type=int, default=12)
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"status": "MISSING", "stages": {}}


def write_supervisor(payload: dict) -> None:
    temporary = SUPERVISOR_STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(SUPERVISOR_STATE)


def resume_stage(state: dict) -> str | None:
    stage_states = state.get("stages", {})
    for stage in STAGES:
        if stage_states.get(stage, {}).get("status") != "COMPLETE":
            return stage
    return None


def run_finalizers(supervisor: dict) -> bool:
    supervisor["status"] = "FINALIZING_PUBLICATION_PACKAGE"
    write_supervisor(supervisor)
    environment = {**os.environ, "PYTHONPATH": str(PROJECT / "src")}
    for relative in FINALIZERS:
        command = [sys.executable, str(PROJECT / relative)]
        return_code = subprocess.call(command, cwd=PROJECT, env=environment)
        supervisor["events"].append(
            {
                "at": now(),
                "event": "finalizer_exit",
                "command": relative,
                "return_code": return_code,
            }
        )
        write_supervisor(supervisor)
        if return_code:
            supervisor["status"] = "FAILED_FINALIZATION"
            supervisor["failed_at"] = now()
            write_supervisor(supervisor)
            return False
    test_command = [sys.executable, "-m", "pytest", "-q"]
    test_environment = {**environment, "PYTHONPATH": f"{PROJECT}:{PROJECT / 'src'}"}
    test_code = subprocess.call(test_command, cwd=PROJECT, env=test_environment)
    supervisor["events"].append(
        {
            "at": now(),
            "event": "final_regression_tests_exit",
            "return_code": test_code,
        }
    )
    write_supervisor(supervisor)
    if test_code:
        supervisor["status"] = "FAILED_FINAL_TESTS"
        supervisor["failed_at"] = now()
        write_supervisor(supervisor)
        return False
    return True


def main() -> None:
    args = parse_args()
    supervisor = {
        "status": "WATCHING_EXISTING_PIPELINE",
        "started_at": now(),
        "supervisor_pid": os.getpid(),
        "initial_pipeline_pid": args.watch_pid,
        "restart_count": 0,
        "events": [],
    }
    write_supervisor(supervisor)
    while pid_alive(args.watch_pid):
        time.sleep(max(5, args.poll_seconds))
    supervisor["events"].append(
        {"at": now(), "event": "initial_pipeline_exited", "pid": args.watch_pid}
    )

    while True:
        state = load_state()
        if state.get("status") == "COMPLETE":
            if not run_finalizers(supervisor):
                raise RuntimeError("Post-pipeline Q1 finalization failed")
            supervisor["status"] = "COMPLETE"
            supervisor["completed_at"] = now()
            write_supervisor(supervisor)
            return
        stage = resume_stage(state)
        if stage is None:
            supervisor["status"] = "FAILED_NO_RESUME_STAGE"
            supervisor["failed_at"] = now()
            write_supervisor(supervisor)
            raise RuntimeError("Pipeline is not COMPLETE but no incomplete registered stage exists")
        if supervisor["restart_count"] >= args.max_restarts:
            supervisor["status"] = "FAILED_RESTART_LIMIT"
            supervisor["failed_at"] = now()
            supervisor["last_resume_stage"] = stage
            write_supervisor(supervisor)
            raise RuntimeError(f"Reached restart limit at stage {stage}")
        supervisor["restart_count"] += 1
        supervisor["status"] = "RUNNING_RECOVERY"
        supervisor["last_resume_stage"] = stage
        supervisor["events"].append(
            {
                "at": now(),
                "event": "pipeline_restart",
                "restart": supervisor["restart_count"],
                "from_stage": stage,
            }
        )
        write_supervisor(supervisor)
        environment = {**os.environ, "PYTHONPATH": str(PROJECT / "src")}
        command = [
            sys.executable,
            str(PROJECT / "tools/run_full_pipeline.py"),
            "--from-stage",
            stage,
        ]
        return_code = subprocess.call(command, cwd=PROJECT, env=environment)
        supervisor["events"].append(
            {
                "at": now(),
                "event": "pipeline_process_exit",
                "return_code": return_code,
                "from_stage": stage,
            }
        )
        write_supervisor(supervisor)
        if return_code:
            time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
