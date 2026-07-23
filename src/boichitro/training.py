from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import collate_examples
from .optim import (
    OptimizerBundle,
    build_optimizers,
    global_grad_norm,
    set_scheduled_learning_rates,
)


@dataclass
class StageTrainerConfig:
    seed: int
    epochs: int = 8
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    muon_lr: float = 0.02
    adamw_lr: float = 3e-4
    router_lr: float = 2e-4
    weight_decay: float = 0.1
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    use_muon: bool = True
    use_groupdro: bool = False
    groupdro_eta: float = 0.01
    groupdro_uniform_mix: float = 0.10
    groupdro_max_ratio: float = 10.0
    adversary_ramp_fraction: float = 0.10
    num_workers: int = 2
    log_every_steps: int = 20
    validate_every_epochs: int = 1
    validation_checkpoints: int | None = None
    early_stopping_patience: int = 5
    selection_metric: str = "objective"
    selection_mode: str = "max"
    selection_tolerance: float = 0.0
    selection_tie_breaker: str | None = None
    selection_tie_mode: str = "min"
    max_optimizer_steps: int | None = None
    save_optimizer_state: bool = True
    retain_last_checkpoint: bool = True


class GroupDROState:
    def __init__(
        self,
        group_count: int,
        *,
        eta: float,
        uniform_mix: float,
        max_ratio: float,
        device: torch.device,
    ) -> None:
        self.eta = eta
        self.uniform_mix = uniform_mix
        self.max_ratio = max_ratio
        self.weights = torch.ones(group_count, device=device, dtype=torch.float32)
        self.weights /= max(1, group_count)
        self.loss_ema = torch.zeros_like(self.weights)
        self.update_count = torch.zeros_like(self.weights)

    @torch.no_grad()
    def example_weights(self, group_ids: torch.Tensor) -> torch.Tensor:
        valid = group_ids.ge(0)
        result = torch.ones_like(group_ids, dtype=torch.float32)
        if bool(valid.any()):
            selected = self.weights[group_ids[valid]]
            result[valid] = selected / self.weights.mean().clamp_min(1e-9)
        return result

    @torch.no_grad()
    def update(self, group_ids: torch.Tensor, losses: torch.Tensor) -> None:
        for group_id in group_ids.unique():
            index = int(group_id)
            if index < 0:
                continue
            mask = group_ids.eq(index)
            group_loss = losses[mask].float().mean()
            self.loss_ema[index].mul_(0.9).add_(0.1 * group_loss)
            self.update_count[index] += 1
            self.weights[index] *= torch.exp(self.eta * group_loss.clamp(max=20.0))
        uniform = torch.full_like(self.weights, 1.0 / len(self.weights))
        self.weights /= self.weights.sum().clamp_min(1e-9)
        self.weights.mul_(1.0 - self.uniform_mix).add_(self.uniform_mix * uniform)
        floor = self.weights.mean() / self.max_ratio
        ceiling = self.weights.mean() * self.max_ratio
        self.weights.clamp_(min=float(floor), max=float(ceiling))
        self.weights /= self.weights.sum().clamp_min(1e-9)

    def state_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "loss_ema": self.loss_ema,
            "update_count": self.update_count,
            "eta": self.eta,
            "uniform_mix": self.uniform_mix,
            "max_ratio": self.max_ratio,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.weights.copy_(state["weights"].to(self.weights.device))
        self.loss_ema.copy_(state["loss_ema"].to(self.loss_ema.device))
        self.update_count.copy_(state["update_count"].to(self.update_count.device))


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_hardlink(source: Path, destination: Path) -> None:
    """Expose one immutable checkpoint at two paths without duplicating bytes."""

    temporary = destination.with_suffix(destination.suffix + ".link.tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(destination)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizers: OptimizerBundle,
    config: StageTrainerConfig,
    epoch: int,
    global_step: int,
    tokens_seen: int,
    examples_seen: int,
    best_metric: float,
    groupdro: GroupDROState | None,
    include_optimizer: bool,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "trainer_config": dataclasses.asdict(config),
        "epoch": epoch,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "examples_seen": examples_seen,
        "best_metric": best_metric,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "groupdro_state": groupdro.state_dict() if groupdro is not None else None,
        "extra": extra or {},
    }
    if include_optimizer:
        payload["adamw_optimizer_state_dict"] = optimizers.adamw.state_dict()
        payload["muon_optimizer_state_dict"] = (
            optimizers.muon.state_dict() if optimizers.muon is not None else None
        )
    _atomic_torch_save(payload, path)


def _better(value: float, best: float, mode: str) -> bool:
    if mode == "max":
        return value > best
    if mode == "min":
        return value < best
    raise ValueError(f"Unknown selection mode: {mode}")


def _within_tolerance(value: float, ceiling: float, tolerance: float, mode: str) -> bool:
    if mode == "max":
        return value >= ceiling - tolerance
    if mode == "min":
        return value <= ceiling + tolerance
    raise ValueError(f"Unknown selection mode: {mode}")


def selection_decision(
    value: float,
    tie_value: float | None,
    *,
    selected_value: float,
    selected_tie: float | None,
    primary_ceiling: float,
    mode: str,
    tolerance: float,
    tie_mode: str,
) -> tuple[bool, float, bool]:
    """Apply a non-drifting primary tolerance followed by a tie-break metric."""

    ceiling_improved = _better(value, primary_ceiling, mode)
    new_ceiling = value if ceiling_improved else primary_ceiling
    if not math.isfinite(selected_value):
        return True, new_ceiling, ceiling_improved
    candidate_eligible = _within_tolerance(value, new_ceiling, tolerance, mode)
    if not candidate_eligible:
        return False, new_ceiling, ceiling_improved
    selected_eligible = _within_tolerance(
        selected_value, new_ceiling, tolerance, mode
    )
    if not selected_eligible:
        return True, new_ceiling, ceiling_improved
    if tie_value is None or selected_tie is None:
        return _better(value, selected_value, mode), new_ceiling, ceiling_improved
    if _better(tie_value, selected_tie, tie_mode):
        return True, new_ceiling, ceiling_improved
    if tie_value == selected_tie and _better(value, selected_value, mode):
        return True, new_ceiling, ceiling_improved
    return False, new_ceiling, ceiling_improved


def train_stage(
    model,
    dataset: Dataset,
    *,
    pad_token_id: int,
    group_count: int,
    config: StageTrainerConfig,
    run_dir: Path,
    validation_fn: Callable[[Any, int], dict[str, float]] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    set_seed(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda examples: collate_examples(examples, pad_token_id),
        drop_last=False,
    )
    optimizers = build_optimizers(
        model,
        muon_lr=config.muon_lr,
        adamw_lr=config.adamw_lr,
        router_lr=config.router_lr,
        weight_decay=config.weight_decay,
        use_muon=config.use_muon,
    )
    steps_per_epoch = math.ceil(len(loader) / config.gradient_accumulation_steps)
    planned_steps = steps_per_epoch * config.epochs
    if config.max_optimizer_steps is not None:
        planned_steps = min(planned_steps, config.max_optimizer_steps)
    if config.validation_checkpoints is not None and config.validation_checkpoints <= 0:
        raise ValueError("validation_checkpoints must be positive when provided")
    if config.selection_tolerance < 0:
        raise ValueError("selection_tolerance must be non-negative")
    if config.selection_mode not in ("min", "max"):
        raise ValueError("selection_mode must be min or max")
    if config.selection_tie_mode not in ("min", "max"):
        raise ValueError("selection_tie_mode must be min or max")
    validation_interval = (
        max(1, math.ceil(planned_steps / config.validation_checkpoints))
        if config.validation_checkpoints is not None
        else None
    )
    warmup_steps = max(1, round(planned_steps * config.warmup_fraction))
    groupdro = (
        GroupDROState(
            group_count,
            eta=config.groupdro_eta,
            uniform_mix=config.groupdro_uniform_mix,
            max_ratio=config.groupdro_max_ratio,
            device=device,
        )
        if config.use_groupdro and group_count > 0
        else None
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    best = -math.inf if config.selection_mode == "max" else math.inf
    primary_ceiling = best
    best_tie = (
        (-math.inf if config.selection_tie_mode == "max" else math.inf)
        if config.selection_tie_breaker is not None
        else None
    )
    no_improvement = 0
    global_step = 0
    tokens_seen = 0
    examples_seen = 0
    last_validation_step = 0
    last_validation: dict[str, float] = {}
    best_checkpoint_step: int | None = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    optimizers.zero_grad()
    muon_name_set = set(optimizers.muon_names)
    adamw_name_set = set(optimizers.adamw_names)

    def record_unselected_primary_ceiling(
        *,
        validation_id: int,
        validation_unit: str,
        value: float,
        tie_value: float | None,
        validation: dict[str, float],
    ) -> None:
        selection_path = run_dir / "best_selection.json"
        if not selection_path.exists():
            return
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["primary_metric_ceiling"] = primary_ceiling
        selection["selected_within_primary_tolerance"] = True
        selection["latest_unselected_ceiling_candidate"] = {
            "validation_id": validation_id,
            "validation_unit": validation_unit,
            "selection_value": value,
            "tie_breaker_value": tie_value,
            "validation": validation,
        }
        _atomic_json_save(selection, selection_path)

    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            for batch_index, batch in enumerate(loader):
                if global_step >= planned_steps:
                    break
                tensor_keys = (
                    "input_ids",
                    "labels",
                    "attention_mask",
                    "task_ids",
                    "dialect_labels",
                    "classification_labels",
                    "source_labels",
                    "group_ids",
                    "example_weights",
                )
                moved = {key: batch[key].to(device, non_blocking=True) for key in tensor_keys}
                progress = global_step / max(1, planned_steps - 1)
                model.set_training_progress(progress)
                adversary_scale = min(
                    1.0, progress / max(1e-9, config.adversary_ramp_fraction)
                )
                base_weights = moved.pop("example_weights")
                group_ids = moved.pop("group_ids")
                if groupdro is not None:
                    effective_weights = base_weights * groupdro.example_weights(group_ids)
                else:
                    effective_weights = base_weights
                # Full accumulation windows reduce to 1 / accumulation_steps.
                # For the final partial window, weight by the actual number of
                # examples so a short last batch cannot be over-represented.
                window_start = (
                    batch_index // config.gradient_accumulation_steps
                ) * config.gradient_accumulation_steps
                window_end = min(
                    window_start + config.gradient_accumulation_steps, len(loader)
                )
                examples_before_window = window_start * config.micro_batch_size
                examples_in_window = min(
                    (window_end - window_start) * config.micro_batch_size,
                    len(dataset) - examples_before_window,
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    output = model(
                        **moved,
                        example_weights=effective_weights,
                        adversary_scale=adversary_scale,
                    )
                    unscaled_loss = output["loss"]
                    loss = unscaled_loss * (
                        moved["input_ids"].size(0) / max(1, examples_in_window)
                    )
                if not bool(torch.isfinite(unscaled_loss)):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch}, batch={batch_index}, step={global_step}"
                    )
                loss.backward()
                if groupdro is not None:
                    groupdro.update(group_ids, output["per_example_loss"].detach())
                epoch_loss += float(unscaled_loss.detach())
                epoch_batches += 1
                tokens_seen += int(moved["attention_mask"].sum())
                examples_seen += int(moved["input_ids"].size(0))

                is_boundary = (batch_index + 1) % config.gradient_accumulation_steps == 0
                is_last = batch_index + 1 == len(loader)
                if not (is_boundary or is_last):
                    continue
                scale = set_scheduled_learning_rates(
                    optimizers,
                    step=global_step,
                    total_steps=planned_steps,
                    warmup_steps=warmup_steps,
                    min_ratio=config.min_lr_ratio,
                )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                muon_grad_norm = global_grad_norm(
                    parameter
                    for name, parameter in model.named_parameters()
                    if name in muon_name_set
                )
                adamw_grad_norm = global_grad_norm(
                    parameter
                    for name, parameter in model.named_parameters()
                    if name in adamw_name_set
                )
                optimizers.step()
                model.update_router_biases()
                optimizers.zero_grad()
                global_step += 1
                if global_step % config.log_every_steps == 0 or global_step == planned_steps:
                    elapsed = time.perf_counter() - started
                    record = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": float(unscaled_loss.detach()),
                        "language_model_loss": float(
                            output["losses"]["language_model"].detach()
                        ),
                        "classification_loss": float(
                            output["losses"]["classification"].detach()
                        ),
                        "contrastive_loss": float(
                            output["losses"]["contrastive"].detach()
                        ),
                        "dialect_loss": float(output["losses"]["dialect"].detach()),
                        "source_loss": float(output["losses"]["source"].detach()),
                        "mtp_loss": float(output["losses"]["mtp"].detach()),
                        "router_z_loss": float(output["losses"]["router_z"].detach()),
                        "balance_loss": float(output["losses"]["balance"].detach()),
                        "gradient_norm": float(gradient_norm),
                        "muon_gradient_norm": float(muon_grad_norm),
                        "adamw_gradient_norm": float(adamw_grad_norm),
                        "lr_scale": scale,
                        "tokens_seen": tokens_seen,
                        "examples_seen": examples_seen,
                        "tokens_per_second": tokens_seen / elapsed,
                        "elapsed_seconds": elapsed,
                        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30
                        if device.type == "cuda"
                        else 0.0,
                    }
                    log_handle.write(json.dumps(record) + "\n")
                    log_handle.flush()
                    print(
                        f"[epoch {epoch:02d} step {global_step:05d}/{planned_steps:05d}] "
                        f"loss={record['loss']:.4f} tok/s={record['tokens_per_second']:,.0f} "
                        f"grad={record['gradient_norm']:.3f}",
                        flush=True,
                    )

                should_validate_step = (
                    validation_fn is not None
                    and validation_interval is not None
                    and (
                        global_step % validation_interval == 0
                        or global_step == planned_steps
                    )
                )
                if should_validate_step:
                    validation = validation_fn(model, global_step)
                    value = float(validation[config.selection_metric])
                    tie_value = (
                        float(validation[config.selection_tie_breaker])
                        if config.selection_tie_breaker is not None
                        else None
                    )
                    select_candidate, primary_ceiling, ceiling_improved = (
                        selection_decision(
                            value,
                            tie_value,
                            selected_value=best,
                            selected_tie=best_tie,
                            primary_ceiling=primary_ceiling,
                            mode=config.selection_mode,
                            tolerance=config.selection_tolerance,
                            tie_mode=config.selection_tie_mode,
                        )
                    )
                    if select_candidate:
                        best = value
                        best_tie = tie_value
                        no_improvement = 0
                        save_checkpoint(
                            run_dir / "best_checkpoint.pt",
                            model=model,
                            optimizers=optimizers,
                            config=config,
                            epoch=epoch,
                            global_step=global_step,
                            tokens_seen=tokens_seen,
                            examples_seen=examples_seen,
                            best_metric=best,
                            groupdro=groupdro,
                            include_optimizer=False,
                            extra={
                                "validation": validation,
                                "validation_id": global_step,
                                "validation_unit": "optimizer_step",
                            },
                        )
                        _atomic_json_save(
                            {
                                "status": "SELECTED_ON_VALIDATION",
                                "selection_metric": config.selection_metric,
                                "selection_mode": config.selection_mode,
                                "selection_value": best,
                                "primary_metric_ceiling": primary_ceiling,
                                "selection_tolerance": config.selection_tolerance,
                                "tie_breaker_metric": config.selection_tie_breaker,
                                "tie_breaker_mode": config.selection_tie_mode,
                                "tie_breaker_value": best_tie,
                                "validation_id": global_step,
                                "validation_unit": "optimizer_step",
                                "epoch": epoch,
                                "global_step": global_step,
                                "validation": validation,
                            },
                            run_dir / "best_selection.json",
                        )
                        best_checkpoint_step = global_step
                    elif ceiling_improved:
                        record_unselected_primary_ceiling(
                            validation_id=global_step,
                            validation_unit="optimizer_step",
                            value=value,
                            tie_value=tie_value,
                            validation=validation,
                        )
                        no_improvement = 0
                    else:
                        no_improvement += 1
                    last_validation = validation
                    last_validation_step = global_step
                    print(
                        f"validation_step={global_step} validation={validation} "
                        f"best={best:.6f}",
                        flush=True,
                    )
                    model.train()

            validation: dict[str, float] = (
                last_validation if last_validation_step == global_step else {}
            )
            if (
                validation_fn is not None
                and validation_interval is None
                and epoch % config.validate_every_epochs == 0
            ):
                validation = validation_fn(model, epoch)
                value = float(validation[config.selection_metric])
                tie_value = (
                    float(validation[config.selection_tie_breaker])
                    if config.selection_tie_breaker is not None
                    else None
                )
                select_candidate, primary_ceiling, ceiling_improved = selection_decision(
                    value,
                    tie_value,
                    selected_value=best,
                    selected_tie=best_tie,
                    primary_ceiling=primary_ceiling,
                    mode=config.selection_mode,
                    tolerance=config.selection_tolerance,
                    tie_mode=config.selection_tie_mode,
                )
                if select_candidate:
                    best = value
                    best_tie = tie_value
                    no_improvement = 0
                    save_checkpoint(
                        run_dir / "best_checkpoint.pt",
                        model=model,
                        optimizers=optimizers,
                        config=config,
                        epoch=epoch,
                        global_step=global_step,
                        tokens_seen=tokens_seen,
                        examples_seen=examples_seen,
                        best_metric=best,
                        groupdro=groupdro,
                        include_optimizer=False,
                        extra={
                            "validation": validation,
                            "validation_id": epoch,
                            "validation_unit": "epoch",
                        },
                    )
                    _atomic_json_save(
                        {
                            "status": "SELECTED_ON_VALIDATION",
                            "selection_metric": config.selection_metric,
                            "selection_mode": config.selection_mode,
                            "selection_value": best,
                            "primary_metric_ceiling": primary_ceiling,
                            "selection_tolerance": config.selection_tolerance,
                            "tie_breaker_metric": config.selection_tie_breaker,
                            "tie_breaker_mode": config.selection_tie_mode,
                            "tie_breaker_value": best_tie,
                            "validation_id": epoch,
                            "validation_unit": "epoch",
                            "epoch": epoch,
                            "global_step": global_step,
                            "validation": validation,
                        },
                        run_dir / "best_selection.json",
                    )
                    best_checkpoint_step = global_step
                elif ceiling_improved:
                    record_unselected_primary_ceiling(
                        validation_id=epoch,
                        validation_unit="epoch",
                        value=value,
                        tie_value=tie_value,
                        validation=validation,
                    )
                    no_improvement = 0
                else:
                    no_improvement += 1
                print(f"validation={validation} best={best:.6f}", flush=True)
                model.train()

            last_checkpoint = run_dir / "last_checkpoint.pt"
            if best_checkpoint_step == global_step and not config.save_optimizer_state:
                _atomic_hardlink(run_dir / "best_checkpoint.pt", last_checkpoint)
            else:
                save_checkpoint(
                    last_checkpoint,
                    model=model,
                    optimizers=optimizers,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    tokens_seen=tokens_seen,
                    examples_seen=examples_seen,
                    best_metric=best,
                    groupdro=groupdro,
                    include_optimizer=config.save_optimizer_state,
                    extra={
                        "validation": validation,
                        "validation_id": (
                            last_validation_step if validation_interval is not None else epoch
                        ),
                        "validation_unit": (
                            "optimizer_step" if validation_interval is not None else "epoch"
                        ),
                    },
                )
            if no_improvement >= config.early_stopping_patience or global_step >= planned_steps:
                break

    elapsed = time.perf_counter() - started
    report = {
        "status": "COMPLETE",
        "global_steps": global_step,
        "tokens_seen": tokens_seen,
        "examples_seen": examples_seen,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
        "best_metric": best if math.isfinite(best) else None,
        "primary_metric_ceiling": primary_ceiling
        if math.isfinite(primary_ceiling)
        else None,
        "selection_tie_breaker": config.selection_tie_breaker,
        "best_tie_breaker_value": best_tie
        if best_tie is not None and math.isfinite(best_tie)
        else None,
        "selection_metric": config.selection_metric,
        "last_checkpoint_retained": config.retain_last_checkpoint,
        "parameter_report": model.parameter_report(),
        "optimizer": {
            "muon_parameter_tensors": len(optimizers.muon_names),
            "adamw_parameter_tensors": len(optimizers.adamw_names),
            "router_parameter_tensors": len(optimizers.router_names),
        },
        "groupdro": (
            {
                "active_groups": int(groupdro.update_count.gt(0).sum().item()),
                "total_groups": int(groupdro.weights.numel()),
                "minimum_weight": float(groupdro.weights.min().item()),
                "maximum_weight": float(groupdro.weights.max().item()),
                "maximum_to_minimum_ratio": float(
                    groupdro.weights.max().item()
                    / max(1e-12, groupdro.weights.min().item())
                ),
                "weights": groupdro.weights.detach().cpu().tolist(),
                "loss_ema": groupdro.loss_ema.detach().cpu().tolist(),
                "update_count": groupdro.update_count.detach().cpu().tolist(),
            }
            if groupdro is not None
            else None
        ),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30
        if device.type == "cuda"
        else 0.0,
    }
    (run_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not config.retain_last_checkpoint:
        if validation_fn is None or not (run_dir / "best_checkpoint.pt").exists():
            raise RuntimeError(
                "Cannot discard the last checkpoint without a validation-selected checkpoint"
            )
        (run_dir / "last_checkpoint.pt").unlink(missing_ok=True)
    return report
