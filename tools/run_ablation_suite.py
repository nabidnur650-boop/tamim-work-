#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize or execute the registered ablation matrix.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/ablation_registry.yaml"
    )
    parser.add_argument(
        "--section", choices=("core_factorial", "confirmatory", "optimization"), required=True
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def command(
    *,
    protocol: str,
    variant: str,
    seeds: list[int],
    ablations: list[str],
    stages: list[str],
    a_tokens: int | None = None,
    s_tokens: int | None = None,
) -> list[str]:
    result = [
        sys.executable,
        str(PROJECT / "tools/run_task_experiments.py"),
        "--config",
        str(PROJECT / "configs/task_experiments.yaml"),
        "--protocol-id",
        protocol,
        "--variants",
        variant,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--stages",
        *stages,
    ]
    if a_tokens is not None:
        result.extend(("--stage-a-token-budget", str(a_tokens)))
    if s_tokens is not None:
        result.extend(("--stage-s-token-budget", str(s_tokens)))
    for ablation in ablations:
        result.extend(("--ablation", ablation))
    return result


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    section = config[args.section]
    commands = []
    if args.section == "core_factorial":
        factors = list(section["factors"].values())
        for enabled in itertools.product((False, True), repeat=len(factors)):
            # enabled=True means the corresponding component is present.
            ablations = [factor for factor, present in zip(factors, enabled) if not present]
            commands.append(
                command(
                    protocol=section["protocol_id"],
                    variant=section["variant"],
                    seeds=[int(section["seed"])],
                    ablations=ablations,
                    stages=["a", "s"],
                    a_tokens=int(section["stage_a_token_budget"]),
                    s_tokens=int(section["stage_s_token_budget"]),
                )
            )
    else:
        for ablations in section["ablations"]:
            commands.append(
                command(
                    protocol=section["protocol_id"],
                    variant=section["variant"],
                    seeds=[int(seed) for seed in section["seeds"]],
                    ablations=list(ablations),
                    stages=["a", "s", "id"],
                )
            )
    for arguments in commands:
        printable = shlex.join(arguments)
        print(printable, flush=True)
        if args.execute:
            subprocess.run(
                arguments,
                cwd=PROJECT,
                check=True,
                env={**dict(os.environ), "PYTHONPATH": str(PROJECT / "src")},
            )


if __name__ == "__main__":
    main()
