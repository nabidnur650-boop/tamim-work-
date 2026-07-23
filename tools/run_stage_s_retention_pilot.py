#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import FixedTokenMixtureDataset, load_encoded_cache  # noqa: E402
from boichitro.tokenization import load_tokenizer, sha256_file  # noqa: E402
from boichitro.training import StageTrainerConfig, train_stage  # noqa: E402
from run_task_experiments import (  # noqa: E402
    model_from_task_checkpoint,
    normalization_validator,
    replay_nll,
    stage_complete,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a Stage-S optimizer/replay schedule on validation data only, "
            "subject to the fixed general-replay degradation guard."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/stage_s_retention_pilot.yaml",
    )
    return parser.parse_args()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_pilot_config(config: dict[str, Any]) -> None:
    if float(config["replay_degradation_limit"]) <= 0:
        raise ValueError("replay_degradation_limit must be positive")
    if int(config["token_budget"]) <= 0:
        raise ValueError("token_budget must be positive")
    identifiers: set[str] = set()
    required = {
        "normalization",
        "identification",
        "dialect_clm",
        "general_replay",
        "romanized",
    }
    for candidate in config["candidates"]:
        identifier = str(candidate["id"])
        if identifier in identifiers:
            raise ValueError(f"Duplicate candidate id: {identifier}")
        identifiers.add(identifier)
        mixture = {str(key): float(value) for key, value in candidate["mixture"].items()}
        if set(mixture) != required:
            raise ValueError(f"{identifier} mixture must contain exactly {sorted(required)}")
        if not math.isclose(sum(mixture.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"{identifier} mixture does not sum to one")
        if any(value <= 0 for value in mixture.values()):
            raise ValueError(f"{identifier} mixture proportions must be positive")
        optimizer = candidate["optimizer"]
        for name in ("muon_lr", "adamw_lr", "router_lr"):
            if float(optimizer[name]) <= 0:
                raise ValueError(f"{identifier} {name} must be positive")
    serialized = json.dumps(config, sort_keys=True).lower()
    if "locked" in serialized or "test.parquet" in serialized:
        raise ValueError("Retention-pilot config may not reference locked test artifacts")


def validation_curve(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("validation_metrics_epoch_*.json")):
        step = int(path.stem.rsplit("_", 1)[-1])
        values = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "optimizer_step": step,
                "macro_chrfpp": float(values["macro_chrfpp"]),
                "worst_dialect_chrfpp": float(values["worst_dialect_chrfpp"]),
                "replay_nll": float(values["replay_nll"]),
                "baseline_replay_nll": float(values["baseline_replay_nll"]),
                "replay_relative_degradation": float(
                    values["replay_relative_degradation"]
                ),
                "replay_guard_pass": bool(values["replay_guard_pass"]),
                "metrics_path": str(path.relative_to(PROJECT)),
            }
        )
    return sorted(rows, key=lambda row: row["optimizer_step"])


def selected_row(curve: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in curve if row["replay_guard_pass"]]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            -row["macro_chrfpp"],
            -row["worst_dialect_chrfpp"],
            row["replay_relative_degradation"],
            row["optimizer_step"],
        ),
    )[0]


def candidate_summary(
    identifier: str,
    run_dir: Path,
    *,
    candidate: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    curve = validation_curve(run_dir)
    selected = selected_row(curve)
    return {
        "candidate_id": identifier,
        "source": source,
        "run_dir": str(run_dir.relative_to(PROJECT)),
        "candidate": candidate,
        "validation_curve": curve,
        "eligible_checkpoint_count": sum(row["replay_guard_pass"] for row in curve),
        "selected_validation": selected,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_pilot_config(config)
    config_sha256 = sha256_file(config_path)
    protocol_id = str(config["protocol_id"])
    seed = int(config["development_seed"])
    output_root = PROJECT / "runs/pilots" / protocol_id
    report_path = PROJECT / "reports/model/stage_s_retention_pilot_selection.json"
    stage_a_path = PROJECT / str(config["stage_a_checkpoint"])
    rejected_default = PROJECT / str(config["rejected_default_stage_s"])
    if not stage_a_path.exists():
        raise FileNotFoundError(
            "The development Stage-A checkpoint must be trained before the retention pilot: "
            f"{stage_a_path}"
        )
    if not stage_complete(rejected_default, "best_checkpoint.pt", "best_selection.json"):
        raise RuntimeError(
            "The declared default Stage-S failure curve must be complete before tuning: "
            f"{rejected_default}"
        )

    tokenizer = load_tokenizer(PROJECT / "artifacts/tokenizers/frozen")
    pad_token_id = int(tokenizer.token_to_id("<pad>"))
    cache_root = PROJECT / "cache/tasks"
    components = {
        "normalization": load_encoded_cache(cache_root / "normalization_train.pt")[0],
        "identification": load_encoded_cache(cache_root / "identification_train.pt")[0],
        "dialect_clm": load_encoded_cache(cache_root / "dialect_clm_train.pt")[0],
        "romanized": load_encoded_cache(cache_root / "normalization_romanized_train.pt")[0],
        "general_replay": load_encoded_cache(cache_root / "general_replay_train.pt")[0],
    }
    maps = json.loads((cache_root / "maps.json").read_text(encoding="utf-8"))
    group_count = len(maps["group_to_id"])
    norm_validation = pd.read_parquet(
        PROJECT / "data/final/v1/normalization_validation.parquet"
    )
    validation_limit = config.get("normalization_validation_limit")
    if validation_limit is not None:
        raise ValueError(
            "The registered full pilot requires the complete normalization validation set"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    baseline_path = output_root / "baseline_replay.json"
    baseline_contract = {
        "stage_a_checkpoint": str(stage_a_path.relative_to(PROJECT)),
        "stage_a_sha256": sha256_file(stage_a_path),
        "replay_validation_examples": int(config["replay_validation_examples"]),
        "tokenizer_sha256": sha256_file(
            PROJECT / "artifacts/tokenizers/frozen/tokenizer.json"
        ),
    }
    if baseline_path.exists():
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_payload["contract"] != baseline_contract:
            raise RuntimeError("Existing pilot baseline contract does not match the config")
        baseline_replay = float(baseline_payload["baseline_replay_nll"])
    else:
        baseline_model = model_from_task_checkpoint(stage_a_path).to(device)
        baseline_replay = replay_nll(
            baseline_model,
            components["general_replay"],
            device=device,
            pad_token_id=pad_token_id,
            examples=int(config["replay_validation_examples"]),
        )
        atomic_json(
            {
                "status": "COMPLETE_VALIDATION_ONLY",
                "contract": baseline_contract,
                "baseline_replay_nll": baseline_replay,
                "test_data_access": False,
            },
            baseline_path,
        )
        del baseline_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summaries = [
        candidate_summary(
            "rejected_default",
            rejected_default,
            candidate=None,
            source="pre_pilot_failure_curve",
        )
    ]
    base_task = yaml.safe_load(
        (PROJECT / "configs/task_experiments.yaml").read_text(encoding="utf-8")
    )
    for candidate in config["candidates"]:
        identifier = str(candidate["id"])
        run_dir = output_root / identifier
        candidate_contract = {
            "protocol_id": protocol_id,
            "config_sha256": config_sha256,
            "candidate": candidate,
            "development_variant": str(config["development_variant"]),
            "development_seed": seed,
            **baseline_contract,
            "token_budget": int(config["token_budget"]),
            "validation_checkpoints": int(config["validation_checkpoints"]),
            "replay_degradation_limit": float(config["replay_degradation_limit"]),
            "test_data_access": False,
        }
        contract_path = run_dir / "candidate_contract.json"
        if contract_path.exists():
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing != candidate_contract:
                raise RuntimeError(f"Existing candidate contract mismatch: {run_dir}")
        else:
            atomic_json(candidate_contract, contract_path)

        if not stage_complete(run_dir, "best_checkpoint.pt", "best_selection.json"):
            model = model_from_task_checkpoint(stage_a_path)
            mixture_values = {
                str(name): float(value)
                for name, value in candidate["mixture"].items()
            }
            mixture = FixedTokenMixtureDataset(
                components,
                mixture_values,
                token_budget=int(config["token_budget"]),
                seed=seed + 100_003,
            )
            atomic_json(mixture.report(), run_dir / "mixture_report.json")
            trainer_values = dict(base_task["stage_s"]["trainer"])
            trainer_values.update(
                muon_lr=float(candidate["optimizer"]["muon_lr"]),
                adamw_lr=float(candidate["optimizer"]["adamw_lr"]),
                router_lr=float(candidate["optimizer"]["router_lr"]),
                validation_checkpoints=int(config["validation_checkpoints"]),
            )
            trainer = StageTrainerConfig(
                seed=seed,
                use_groupdro=True,
                use_muon=True,
                **trainer_values,
            )
            atomic_json(dataclasses.asdict(trainer), run_dir / "trainer_config.json")
            validate = normalization_validator(
                tokenizer,
                norm_validation,
                run_dir=run_dir,
                device=device,
                batch_size=int(base_task["evaluation"]["normalization_batch_size"]),
                max_new_tokens=int(
                    base_task["evaluation"]["normalization_max_new_tokens"]
                ),
                baseline_replay_nll=baseline_replay,
                replay_dataset=components["general_replay"],
                replay_examples=int(config["replay_validation_examples"]),
                replay_limit=float(config["replay_degradation_limit"]),
                pad_token_id=pad_token_id,
            )
            train_stage(
                model,
                mixture,
                pad_token_id=pad_token_id,
                group_count=group_count,
                config=trainer,
                run_dir=run_dir,
                validation_fn=validate,
                device=device,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        summaries.append(
            candidate_summary(
                identifier,
                run_dir,
                candidate=candidate,
                source="registered_candidate",
            )
        )

    eligible = [
        row
        for row in summaries
        if row["source"] == "registered_candidate"
        and row["selected_validation"] is not None
    ]
    status = "COMPLETE_VALIDATION_ONLY" if eligible else "FAILED_NO_ELIGIBLE_CANDIDATE"
    selected = (
        sorted(
            eligible,
            key=lambda row: (
                -row["selected_validation"]["macro_chrfpp"],
                -row["selected_validation"]["worst_dialect_chrfpp"],
                row["selected_validation"]["replay_relative_degradation"],
                row["candidate_id"],
            ),
        )[0]
        if eligible
        else None
    )
    report = {
        "status": status,
        "protocol_id": protocol_id,
        "config_path": str(config_path.relative_to(PROJECT)),
        "config_sha256": config_sha256,
        "development_variant": str(config["development_variant"]),
        "development_seed": seed,
        "stage_a_checkpoint": baseline_contract["stage_a_checkpoint"],
        "stage_a_sha256": baseline_contract["stage_a_sha256"],
        "baseline_replay_nll": baseline_replay,
        "selection_rule": config["selection"],
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_candidate": selected["candidate"] if selected else None,
        "selected_validation": selected["selected_validation"] if selected else None,
        "candidates": summaries,
        "test_data_access": False,
    }
    atomic_json(report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if selected is None:
        raise RuntimeError("No registered Stage-S schedule passed the replay guard")


if __name__ == "__main__":
    main()
