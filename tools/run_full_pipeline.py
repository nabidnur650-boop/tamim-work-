#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the registered end-to-end experiment pipeline with safe resume semantics."
    )
    parser.add_argument("--from-stage")
    parser.add_argument("--through-stage")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def python(*arguments: str) -> list[str]:
    return [sys.executable, *[str(PROJECT / value) if value.startswith("tools/") else value for value in arguments]]


def stages() -> list[tuple[str, list[list[str]]]]:
    confirmatory = [
        "no_lexical_prior",
        "no_dialect_head",
        "no_source_adversary",
        "no_task_conditioning",
        "randomized_lexical_prior",
        "no_groupdro",
        "no_synthetic",
        "no_general_replay",
    ]
    optimization = ["adamw_only", "no_mtp"]
    return [
        (
            "foundations",
            [
                python("tools/train_foundation.py", "--config", "configs/foundation_300m.yaml", "--resume"),
                python("tools/run_continuation_lr_pilot.py"),
                python("tools/train_foundation.py", "--config", "configs/continuation_m0_dense_200m.yaml", "--resume"),
                python("tools/train_foundation.py", "--config", "configs/pilot_m2_banked_20m.yaml", "--resume"),
                python("tools/train_foundation.py", "--config", "configs/pilot_m2_unbanked_20m.yaml", "--resume"),
                python("tools/train_foundation.py", "--config", "configs/pilot_m2_scratch_20m.yaml", "--resume"),
                python("tools/train_foundation.py", "--config", "configs/pilot_m2_annealed_20m.yaml", "--resume"),
                python("tools/train_foundation.py", "--config", "configs/pilot_m2_paired_20m.yaml", "--resume"),
                python("tools/select_upcycling_strategy.py"),
                python("tools/train_foundation.py", "--config", "configs/continuation_m2_200m.yaml", "--resume"),
                python("tools/train_foundation.py", "--config", "configs/pilot_m1_loss_free_router_10m.yaml", "--resume"),
                python("tools/train_foundation.py", "--config", "configs/pilot_m1_aux_router_10m.yaml", "--resume"),
                python("tools/select_switch_router.py"),
                python("tools/train_foundation.py", "--config", "configs/continuation_m1_switch_200m.yaml", "--resume"),
                python("tools/analyze_foundation_routing.py"),
                python("tools/preflight_task_models.py"),
            ],
        ),
        (
            "stage_s_default_pilot",
            [
                python(
                    "tools/run_task_experiments.py",
                    "--config",
                    "configs/task_experiments_rejected_stage_s_default.yaml",
                    "--stages",
                    "a",
                    "s",
                )
            ],
        ),
        (
            "stage_s_retention_pilot",
            [python("tools/run_stage_s_retention_pilot.py")],
        ),
        (
            "adopt_stage_s_retention_pilot",
            [python("tools/adopt_stage_s_retention_pilot.py")],
        ),
        ("main_task", [python("tools/run_task_experiments.py")]),
        (
            "development_system_fusion",
            [
                python("tools/summarize_development_results.py"),
                python("tools/audit_cross_task_input_firewall.py"),
                python("tools/audit_source_blind_baselines.py"),
                python("tools/run_identification_fusion_pilot.py"),
                python("tools/run_normalization_fusion_pilot.py", "--version", "v1"),
                python("tools/run_normalization_fusion_pilot.py", "--version", "v2"),
                python("tools/audit_development_fusion.py"),
                python("tools/evaluate_development_fusion_transfer.py"),
            ],
        ),
        ("optimizer_pilot", [python("tools/run_optimizer_pilot.py")]),
        ("task_inference_benchmark", [python("tools/benchmark_task_inference.py")]),
        ("bidirectional_id", [python("tools/run_bidirectional_identification.py")]),
        ("external_normalization", [python("tools/train_external_normalization.py")]),
        ("external_identification", [python("tools/train_external_identification.py")]),
        (
            "factorial_pilot",
            [python("tools/run_ablation_suite.py", "--section", "core_factorial", "--execute")],
        ),
        (
            "confirmatory_ablations",
            [python("tools/run_ablation_suite.py", "--section", "confirmatory", "--execute")],
        ),
        (
            "optimization_ablations",
            [python("tools/run_ablation_suite.py", "--section", "optimization", "--execute")],
        ),
        ("protocol_freeze", [python("tools/freeze_protocol.py")]),
        ("locked_main", [python("tools/evaluate_locked.py")]),
        (
            "locked_confirmatory_ablations",
            [
                python(
                    "tools/evaluate_locked.py",
                    "--variants",
                    "M3",
                    "--ablation",
                    ablation,
                    "--task-protocol-id",
                    "boichitro_confirmatory_ablation_v1",
                )
                for ablation in confirmatory
            ],
        ),
        (
            "locked_optimization_ablations",
            [
                python(
                    "tools/evaluate_locked.py",
                    "--variants",
                    "M3",
                    "--ablation",
                    ablation,
                    "--task-protocol-id",
                    "boichitro_optimization_ablation_v1",
                )
                for ablation in optimization
            ],
        ),
        (
            "locked_bidirectional_id",
            [python("tools/evaluate_bidirectional_identification_locked.py")],
        ),
        ("locked_external", [python("tools/evaluate_external_locked.py")]),
        ("human_evaluation_packets", [python("tools/prepare_human_evaluation.py")]),
        ("robustness", [python("tools/evaluate_robustness.py")]),
        ("routing_analysis", [python("tools/analyze_routing.py")]),
        (
            "statistics",
            [
                python("tools/run_statistics.py"),
                python(
                    "tools/run_statistics.py",
                    "--config",
                    "configs/statistical_analysis_specialized.yaml",
                ),
                python(
                    "tools/run_statistics.py",
                    "--config",
                    "configs/statistical_analysis_ablations.yaml",
                ),
                python(
                    "tools/run_statistics.py",
                    "--config",
                    "configs/statistical_analysis_fused.yaml",
                ),
            ],
        ),
        (
            "publication_bundle",
            [
                python("tools/summarize_development_results.py"),
                python("tools/make_paper_figures.py"),
                python("tools/make_q1_figure_suite.py"),
                python("tools/validate_q1_bundle.py"),
                python("tools/make_paper_tables.py"),
                python("tools/build_q1_manuscript.py"),
                python("tools/audit_q1_readiness.py"),
                python("tools/capture_environment.py"),
            ],
        ),
    ]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_command(arguments: list[str], log_path: Path) -> None:
    environment = {**os.environ, "PYTHONPATH": str(PROJECT / "src")}
    with log_path.open("a", encoding="utf-8") as log:
        header = f"\n[{utc_now()}] $ {' '.join(arguments)}\n"
        log.write(header)
        log.flush()
        print(header, end="", flush=True)
        process = subprocess.Popen(
            arguments,
            cwd=PROJECT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, arguments)


def main() -> None:
    args = parse_args()
    registered = stages()
    names = [name for name, _ in registered]
    if args.list:
        print("\n".join(names))
        return
    if args.from_stage and args.from_stage not in names:
        raise ValueError(f"Unknown --from-stage {args.from_stage}; choose from {names}")
    if args.through_stage and args.through_stage not in names:
        raise ValueError(f"Unknown --through-stage {args.through_stage}; choose from {names}")
    first = names.index(args.from_stage) if args.from_stage else 0
    last = names.index(args.through_stage) if args.through_stage else len(names) - 1
    if first > last:
        raise ValueError("--from-stage occurs after --through-stage")

    report_dir = PROJECT / "reports/pipeline"
    log_dir = report_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = report_dir / "full_pipeline_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"status": "PENDING", "created_at": utc_now(), "stages": {}}
    )
    state["status"] = "RUNNING"
    state.pop("error", None)
    state.pop("failed_at", None)
    state.pop("completed_at", None)
    state["last_started_at"] = utc_now()
    write_state(state_path, state)

    try:
        for stage_name, commands in registered[first : last + 1]:
            stage_state = {
                "status": "RUNNING",
                "started_at": utc_now(),
                "commands": commands,
            }
            state["stages"][stage_name] = stage_state
            write_state(state_path, state)
            log_path = log_dir / f"{stage_name}.log"
            for command_index, arguments in enumerate(commands, start=1):
                stage_state["command_index"] = command_index
                write_state(state_path, state)
                run_command(arguments, log_path)
            stage_state["status"] = "COMPLETE"
            stage_state["completed_at"] = utc_now()
            write_state(state_path, state)
    except BaseException as error:
        state["status"] = "FAILED"
        state["failed_at"] = utc_now()
        state["error"] = f"{type(error).__name__}: {error}"
        write_state(state_path, state)
        raise
    state["status"] = "COMPLETE"
    state["completed_at"] = utc_now()
    write_state(state_path, state)
    print(f"Full pipeline complete: {state_path}")


if __name__ == "__main__":
    main()
