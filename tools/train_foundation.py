#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.modeling import (  # noqa: E402
    BoichitroConfig,
    BoichitroForMultiTask,
    upcycle_dense_to_moe,
)
from boichitro.optim import build_optimizers, set_scheduled_learning_rates  # noqa: E402
from boichitro.pretraining import OrderedBlocks, load_packed_dataset  # noqa: E402
from boichitro.tokenization import load_tokenizer, sha256_file  # noqa: E402
from boichitro.training import _atomic_hardlink, _atomic_torch_save, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fixed-budget dense foundation or general MoE continuation."
    )
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/foundation_300m.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--token-budget", type=int)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, loader, device, character_count: int, token_cap: int) -> dict[str, float]:
    model.eval()
    nll = 0.0
    predicted_tokens = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, labels=batch)
        tokens = batch.numel() - batch.size(0)
        nll += float(output["losses"]["language_model"]) * tokens
        predicted_tokens += tokens
        if predicted_tokens >= token_cap:
            break
    # Scale held-out characters to the evaluated fraction of held-out tokens.
    full_tokens = len(loader.dataset) * model.config.max_seq_len
    evaluated_characters = character_count * min(1.0, predicted_tokens / max(1, full_tokens))
    token_nll = nll / max(1, predicted_tokens)
    return {
        "validation_nll": token_nll,
        "validation_perplexity": math.exp(min(20.0, token_nll)),
        "validation_bpc": nll / math.log(2.0) / max(1.0, evaluated_characters),
        "validation_predicted_tokens": predicted_tokens,
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    token_budget = int(args.token_budget or config["token_budget"])
    run_dir = PROJECT / "runs" / config["run_id"] / str(config["seed"])
    # A completed fixed-budget artifact is self-contained. Validate its
    # identity before reconstructing the model so stricter future validators
    # cannot make an immutable historical run non-resumable.
    if args.resume and (run_dir / "training_report.json").exists():
        existing_report = json.loads(
            (run_dir / "training_report.json").read_text(encoding="utf-8")
        )
        if (
            existing_report.get("status") == "COMPLETE_FIXED_BUDGET"
            and int(existing_report.get("tokens_seen", 0)) >= token_budget
        ):
            current_config_sha256 = sha256_file(args.config)
            recorded_config_sha256 = existing_report.get("training_config_sha256")
            if recorded_config_sha256 is None and config["run_id"] == "F_DENSE_300M":
                supplement_path = (
                    PROJECT / "reports/model/f_dense_300m_provenance_supplement.json"
                )
                if not supplement_path.exists():
                    raise FileNotFoundError(
                        f"Dense provenance supplement is missing: {supplement_path}"
                    )
                supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
                if supplement.get("status") != "COMPLETE_PROVENANCE_SUPPLEMENT":
                    raise RuntimeError(
                        f"Dense provenance supplement is incomplete: {supplement_path}"
                    )
                recorded_config_sha256 = supplement[
                    "reconstructed_from_frozen_inputs"
                ]["training_config_sha256"]
            if recorded_config_sha256 != current_config_sha256:
                raise RuntimeError(
                    f"Completed run/config hash mismatch at {run_dir}: "
                    f"{recorded_config_sha256} != "
                    f"{current_config_sha256}"
                )
            for required in ("final_checkpoint.pt", "train_log.jsonl", "validation_log.jsonl"):
                if not (run_dir / required).exists():
                    raise FileNotFoundError(
                        f"Completed run is missing {required}: {run_dir}"
                    )
            print(f"Run already complete at fixed budget: {run_dir}")
            return
    optimizer_selection_source: dict[str, object] | None = None
    upcycling_selection_source: dict[str, object] | None = None
    switch_router_selection_source: dict[str, object] | None = None
    if config.get("optimizer_selection_report"):
        selection_path = PROJECT / str(config["optimizer_selection_report"])
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Continuation optimizer selection is missing: {selection_path}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "COMPLETE_VALIDATION_ONLY":
            raise RuntimeError(
                f"Continuation optimizer selection is not complete: {selection_path}"
            )
        config["optimizer"] = dict(config["optimizer"])
        config["optimizer"]["muon_lr"] = float(selection["selected_muon_lr"])
        config["optimizer"]["adamw_lr"] = float(selection["selected_adamw_lr"])
        optimizer_selection_source = {
            "path": str(selection_path.relative_to(PROJECT)),
            "sha256": sha256_file(selection_path),
            "selected_muon_lr": float(selection["selected_muon_lr"]),
            "selected_adamw_lr": float(selection["selected_adamw_lr"]),
        }
    if config.get("upcycling_selection_report"):
        selection_path = PROJECT / str(config["upcycling_selection_report"])
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Upcycling strategy selection is missing: {selection_path}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "COMPLETE_VALIDATION_ONLY":
            raise RuntimeError(
                f"Upcycling strategy selection is not complete: {selection_path}"
            )
        if selection.get("test_data_access") != "forbidden_and_not_accessed":
            raise RuntimeError("Upcycling strategy selection was not validation-only")
        selected_overrides = selection.get("selected_model_overrides")
        if not isinstance(selected_overrides, dict):
            raise RuntimeError("Selected upcycling strategy has no model overrides")
        config["model_overrides"] = {
            **dict(config.get("model_overrides", {})),
            **selected_overrides,
        }
        upcycling_selection_source = {
            "path": str(selection_path.relative_to(PROJECT)),
            "sha256": sha256_file(selection_path),
            "protocol_id": selection["protocol_id"],
            "selected_strategy": selection["selected_strategy"],
            "selected_run_id": selection["selected_run_id"],
            "selected_model_overrides": selected_overrides,
        }
    if config.get("switch_router_selection_report"):
        selection_path = PROJECT / str(config["switch_router_selection_report"])
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Switch router selection is missing: {selection_path}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "COMPLETE_VALIDATION_ONLY":
            raise RuntimeError(f"Switch router selection is incomplete: {selection_path}")
        if selection.get("test_data_access") != "forbidden_and_not_accessed":
            raise RuntimeError("Switch router selection was not validation-only")
        selected_overrides = selection.get("selected_model_overrides")
        if not isinstance(selected_overrides, dict):
            raise RuntimeError("Selected Switch router has no model overrides")
        config["model_overrides"] = {
            **dict(config.get("model_overrides", {})),
            **selected_overrides,
        }
        config["upcycle_router_init_std"] = float(
            selection["selected_upcycle_router_init_std"]
        )
        switch_router_selection_source = {
            "path": str(selection_path.relative_to(PROJECT)),
            "sha256": sha256_file(selection_path),
            "protocol_id": selection["protocol_id"],
            "selected_strategy": selection["selected_strategy"],
            "selected_run_id": selection["selected_run_id"],
            "selected_upcycle_router_init_std": selection[
                "selected_upcycle_router_init_std"
            ],
            "selected_model_overrides": selected_overrides,
        }
    set_seed(int(config["seed"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    packed_root = PROJECT / config["packed_data"]
    train_data, metadata = load_packed_dataset(packed_root, "train")
    validation_data, _ = load_packed_dataset(packed_root, "validation")
    block_size = int(metadata["block_size"])
    blocks_needed = math.ceil(token_budget / block_size)
    if blocks_needed > len(train_data):
        raise ValueError(f"Need {blocks_needed} blocks but corpus has {len(train_data)}")
    data_seed = int(config.get("data_seed", config["seed"]))
    order = np.random.default_rng(data_seed).permutation(len(train_data))[:blocks_needed]
    order_sha256 = hashlib.sha256(
        np.asarray(order, dtype="<i8").tobytes()
    ).hexdigest()
    ordered = OrderedBlocks(train_data, order)
    train_loader = DataLoader(
        ordered,
        batch_size=int(config["micro_batch_size"]),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=int(config["micro_batch_size"]),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    tokenizer = load_tokenizer(PROJECT / "artifacts/tokenizers/frozen")
    model_yaml = yaml.safe_load((PROJECT / config["model_config"]).read_text(encoding="utf-8"))
    values = dict(model_yaml["model"])
    architecture = str(config.get("architecture", "dense"))
    values.update(config.get("model_overrides", {}))
    values.update(
        vocab_size=tokenizer.get_vocab_size(),
        architecture=architecture,
        n_sources=1,
        use_classification_head=True,
        use_dialect_aux_head=True,
        use_source_adversary=False,
        use_task_conditioning=False,
        use_lexical_routing_prior=False,
    )
    if architecture == "dense":
        values["dense_prefix_layers"] = int(values["n_layers"])
    model = BoichitroForMultiTask(BoichitroConfig.from_dict(values)).to(device)
    initialization_report: dict[str, object] = {"scheme": "random_initialization"}
    initial_checkpoint = config.get("initial_checkpoint")
    if initial_checkpoint:
        checkpoint_path = PROJECT / str(initial_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source_config = BoichitroConfig.from_dict(checkpoint["model_config"])
        source_model = BoichitroForMultiTask(source_config)
        source_model.load_state_dict(checkpoint["model_state_dict"])
        if architecture == "dense":
            model.load_state_dict(source_model.state_dict())
            initialization_report = {"scheme": "dense_checkpoint_copy"}
        else:
            initialization_report = upcycle_dense_to_moe(
                source_model,
                model,
                noise_std=float(config.get("upcycle_noise_std", 1e-5)),
                router_init_std=float(config.get("upcycle_router_init_std", 0.0)),
            )
        initialization_report["source_checkpoint"] = str(initial_checkpoint)
        initialization_report["source_checkpoint_sha256"] = sha256_file(checkpoint_path)
        initialization_report["source_tokens_seen"] = int(checkpoint.get("tokens_seen", 0))
        del source_model, checkpoint
    optimizer_config = config["optimizer"]
    optimizers = build_optimizers(
        model,
        muon_lr=float(optimizer_config["muon_lr"]),
        adamw_lr=float(optimizer_config["adamw_lr"]),
        router_lr=float(optimizer_config["router_lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        use_muon=True,
    )
    accumulation = int(config["gradient_accumulation_steps"])
    configured_global_batch = int(config["global_token_batch"])
    expected_global_batch = (
        int(config["micro_batch_size"]) * accumulation * block_size
    )
    if configured_global_batch != expected_global_batch:
        raise ValueError(
            "global_token_batch does not match "
            "micro_batch_size * gradient_accumulation_steps * block_size: "
            f"{configured_global_batch} != {expected_global_batch}"
        )
    training_steps = math.ceil(len(train_loader) / accumulation)
    scheduler_token_budget = int(config.get("scheduler_token_budget", token_budget))
    scheduler_blocks = math.ceil(scheduler_token_budget / block_size)
    scheduler_microbatches = math.ceil(
        scheduler_blocks / int(config["micro_batch_size"])
    )
    scheduler_steps = math.ceil(scheduler_microbatches / accumulation)
    warmup = max(1, round(scheduler_steps * float(config["warmup_fraction"])))
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and (
        (run_dir / "last_checkpoint.pt").exists()
        or (run_dir / "final_checkpoint.pt").exists()
    ):
        raise RuntimeError(
            f"Run artifacts already exist at {run_dir}; use --resume instead of overwriting"
        )
    start_microbatch = 0
    global_step = 0
    tokens_seen = 0
    starting_tokens = 0
    next_eval = int(config["eval_every_tokens"])
    next_checkpoint = int(config["checkpoint_every_tokens"])
    if args.resume and (run_dir / "last_checkpoint.pt").exists():
        checkpoint = torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizers.adamw.load_state_dict(checkpoint["adamw_optimizer_state_dict"])
        if optimizers.muon is not None and checkpoint.get("muon_optimizer_state_dict"):
            optimizers.muon.load_state_dict(checkpoint["muon_optimizer_state_dict"])
        start_microbatch = int(checkpoint["microbatches_completed"])
        global_step = int(checkpoint["global_step"])
        tokens_seen = int(checkpoint["tokens_seen"])
        starting_tokens = tokens_seen
        next_eval = ((tokens_seen // int(config["eval_every_tokens"])) + 1) * int(
            config["eval_every_tokens"]
        )
        next_checkpoint = (
            (tokens_seen // int(config["checkpoint_every_tokens"])) + 1
        ) * int(config["checkpoint_every_tokens"])
        print(f"Resumed at microbatch={start_microbatch} step={global_step} tokens={tokens_seen}")

    def checkpoint_payload(microbatches_completed: int) -> dict:
        return {
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "adamw_optimizer_state_dict": optimizers.adamw.state_dict(),
            "muon_optimizer_state_dict": optimizers.muon.state_dict()
            if optimizers.muon is not None
            else None,
            "global_step": global_step,
            "microbatches_completed": microbatches_completed,
            "tokens_seen": tokens_seen,
            "tokenizer_sha256": sha256_file(
                PROJECT / "artifacts/tokenizers/frozen/tokenizer.json"
            ),
            "packed_metadata_sha256": sha256_file(packed_root / "metadata.json"),
            "training_config_sha256": sha256_file(args.config),
            "data_seed": data_seed,
            "block_order_sha256": order_sha256,
            "initialization_report": initialization_report,
            "optimizer_selection_source": optimizer_selection_source,
            "upcycling_selection_source": upcycling_selection_source,
            "switch_router_selection_source": switch_router_selection_source,
            "resolved_optimizer": optimizer_config,
            "scheduler_token_budget": scheduler_token_budget,
            "scheduler_steps": scheduler_steps,
        }

    optimizers.zero_grad()
    started = time.perf_counter()
    interval_started = started
    interval_tokens = tokens_seen
    log_path = run_dir / "train_log.jsonl"
    mode = "a" if start_microbatch else "w"
    torch.cuda.reset_peak_memory_stats()
    validation_log_path = run_dir / "validation_log.jsonl"
    if start_microbatch == 0 and bool(config.get("evaluate_at_start", False)):
        model.set_training_progress(0.0)
        initial_metrics = evaluate(
            model,
            validation_loader,
            device,
            int(metadata["splits"]["validation"]["characters"]),
            int(config["validation_token_cap"]),
        )
        initial_metrics.update(global_step=0, tokens_seen=0)
        validation_log_path.write_text(
            json.dumps(initial_metrics) + "\n", encoding="utf-8"
        )
        print(f"initial_validation {initial_metrics}", flush=True)
        model.train()
    with log_path.open(mode, encoding="utf-8") as log_handle:
        for microbatch_index, batch in enumerate(train_loader):
            if microbatch_index < start_microbatch:
                continue
            batch = batch.to(device, non_blocking=True)
            model.set_training_progress(tokens_seen / max(1, token_budget))
            # Weight every microbatch by its actual token count. This is identical
            # to dividing by `accumulation` for full windows, while correctly
            # handling the final partial window and its possibly short last batch.
            window_start = (microbatch_index // accumulation) * accumulation
            window_end = min(window_start + accumulation, len(train_loader))
            blocks_before_window = window_start * int(config["micro_batch_size"])
            blocks_in_window = min(
                (window_end - window_start) * int(config["micro_batch_size"]),
                len(ordered) - blocks_before_window,
            )
            window_tokens = blocks_in_window * block_size
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch, labels=batch)
                loss = output["loss"] * (batch.numel() / window_tokens)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at microbatch {microbatch_index}")
            loss.backward()
            tokens_seen += batch.numel()
            last_microbatch = microbatch_index + 1 == len(train_loader)
            if (microbatch_index + 1) % accumulation != 0 and not last_microbatch:
                continue
            scale = set_scheduled_learning_rates(
                optimizers,
                step=global_step,
                total_steps=scheduler_steps,
                warmup_steps=warmup,
                min_ratio=float(optimizer_config["min_lr_ratio"]),
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optimizer_config["gradient_clip"])
            )
            optimizers.step()
            model.update_router_biases()
            optimizers.zero_grad()
            global_step += 1
            if global_step % int(config["log_every_optimizer_steps"]) == 0:
                now = time.perf_counter()
                record = {
                    "global_step": global_step,
                    "microbatch": microbatch_index + 1,
                    "tokens_seen": tokens_seen,
                    "loss": float(output["loss"].detach()),
                    "lm_loss": float(output["losses"]["language_model"].detach()),
                    "mtp_loss": float(output["losses"]["mtp"].detach()),
                    "gradient_norm": float(gradient_norm),
                    "lr_scale": scale,
                    "interval_tokens_per_second": (tokens_seen - interval_tokens)
                    / max(1e-9, now - interval_started),
                    "overall_tokens_per_second": tokens_seen / max(1e-9, now - started),
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                if output["routing"]:
                    counts = torch.stack(
                        [item.counts.float().cpu() for item in output["routing"]]
                    )
                    mean_counts = counts.mean(dim=0)
                    record.update(
                        bank_constraint_strength=model.upcycle_bank_constraint_strength(),
                        router_load_cv=float(
                            mean_counts.std() / mean_counts.mean().clamp_min(1e-9)
                        ),
                        router_entropy=float(
                            torch.stack(
                                [item.entropy.detach().float().cpu() for item in output["routing"]]
                            ).mean()
                        ),
                        router_z_loss=float(output["losses"]["router_z"].detach()),
                    )
                log_handle.write(json.dumps(record) + "\n")
                log_handle.flush()
                print(
                    f"[step {global_step:05d}/{training_steps:05d}] tok={tokens_seen:,} "
                    f"loss={record['loss']:.4f} tok/s={record['interval_tokens_per_second']:,.0f}",
                    flush=True,
                )
                interval_started, interval_tokens = now, tokens_seen
            if tokens_seen >= next_eval:
                metrics = evaluate(
                    model,
                    validation_loader,
                    device,
                    int(metadata["splits"]["validation"]["characters"]),
                    int(config["validation_token_cap"]),
                )
                metrics.update(global_step=global_step, tokens_seen=tokens_seen)
                with validation_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                print(f"validation {metrics}", flush=True)
                next_eval += int(config["eval_every_tokens"])
                model.train()
            if tokens_seen >= next_checkpoint:
                _atomic_torch_save(
                    checkpoint_payload(microbatch_index + 1),
                    run_dir / f"checkpoint_tokens_{tokens_seen:012d}.pt",
                )
                next_checkpoint += int(config["checkpoint_every_tokens"])
            if (
                global_step % int(config["resume_checkpoint_every_steps"]) == 0
                or last_microbatch
            ):
                _atomic_torch_save(
                    checkpoint_payload(microbatch_index + 1), run_dir / "last_checkpoint.pt"
                )

    # The final microbatch always writes the complete resumable checkpoint.
    # Keep publication-facing aliases without storing identical model and
    # optimizer tensors two or three times.
    final_checkpoint = run_dir / "final_checkpoint.pt"
    _atomic_hardlink(run_dir / "last_checkpoint.pt", final_checkpoint)
    final_milestone = run_dir / f"checkpoint_tokens_{tokens_seen:012d}.pt"
    if final_milestone.exists():
        _atomic_hardlink(run_dir / "last_checkpoint.pt", final_milestone)
    model.set_training_progress(tokens_seen / max(1, token_budget))
    final_validation = evaluate(
        model,
        validation_loader,
        device,
        int(metadata["splits"]["validation"]["characters"]),
        int(config["validation_token_cap"]),
    )
    endpoint_validation = {
        **final_validation,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "training_progress": model.training_progress,
        "bank_constraint_strength": model.upcycle_bank_constraint_strength(),
        "evaluation": "exact_final_training_progress",
    }
    (run_dir / "endpoint_validation.json").write_text(
        json.dumps(endpoint_validation, indent=2) + "\n", encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    report = {
        "status": "COMPLETE_FIXED_BUDGET",
        "tokens_seen": tokens_seen,
        "optimizer_steps": global_step,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (tokens_seen - starting_tokens) / max(1e-9, elapsed),
        "parameter_report": model.parameter_report(),
        "architecture": architecture,
        "training_config_sha256": sha256_file(args.config),
        "data_seed": data_seed,
        "block_order_sha256": order_sha256,
        "initialization_report": initialization_report,
        "optimizer_selection_source": optimizer_selection_source,
        "upcycling_selection_source": upcycling_selection_source,
        "switch_router_selection_source": switch_router_selection_source,
        "resolved_optimizer": optimizer_config,
        "scheduler_token_budget": scheduler_token_budget,
        "scheduler_steps": scheduler_steps,
        "final_validation": final_validation,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (run_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
