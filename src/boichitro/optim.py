from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class OptimizerBundle:
    muon: torch.optim.Optimizer | None
    adamw: torch.optim.Optimizer
    muon_names: list[str]
    adamw_names: list[str]
    router_names: list[str]

    def zero_grad(self) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=True)
        self.adamw.zero_grad(set_to_none=True)

    def step(self) -> None:
        if self.muon is not None:
            self.muon.step()
        self.adamw.step()


def _is_adamw_family(name: str, parameter: nn.Parameter) -> bool:
    fragments = (
        "token_embedding",
        "lm_head",
        "norm",
        "router",
        "dynamic_bias",
        "task_router_bias",
        "dialect_router_map",
        "dialect_head",
        "classification_head",
        "source_head",
        "mtp_projection",
    )
    return parameter.ndim < 2 or any(fragment in name for fragment in fragments)


def partition_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter], list[str], list[str]]:
    muon_parameters: list[nn.Parameter] = []
    adamw_parameters: list[nn.Parameter] = []
    muon_names: list[str] = []
    adamw_names: list[str] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identifier = id(parameter)
        # Tied embeddings are named once in current PyTorch, but guard explicitly.
        if identifier in seen:
            continue
        seen.add(identifier)
        if _is_adamw_family(name, parameter):
            adamw_parameters.append(parameter)
            adamw_names.append(name)
        else:
            if parameter.ndim != 2:
                raise AssertionError(f"Muon parameter is not 2D: {name} {tuple(parameter.shape)}")
            muon_parameters.append(parameter)
            muon_names.append(name)

    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    owned = {id(parameter) for parameter in muon_parameters + adamw_parameters}
    if trainable != owned:
        raise AssertionError(
            f"Optimizer ownership mismatch: missing={len(trainable - owned)}, extra={len(owned - trainable)}"
        )
    if {id(parameter) for parameter in muon_parameters} & {
        id(parameter) for parameter in adamw_parameters
    }:
        raise AssertionError("A parameter belongs to both Muon and AdamW")
    return muon_parameters, adamw_parameters, muon_names, adamw_names


def build_optimizers(
    model: nn.Module,
    *,
    muon_lr: float,
    adamw_lr: float,
    router_lr: float,
    weight_decay: float,
    use_muon: bool = True,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    muon_momentum: float = 0.95,
) -> OptimizerBundle:
    muon_parameters, adamw_parameters, muon_names, adamw_names = partition_parameters(model)
    router_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(fragment in name for fragment in ("router", "dialect_head", "source_head", "classification_head"))
    }
    router_parameters = [parameter for parameter in adamw_parameters if id(parameter) in router_ids]
    residual_parameters = [parameter for parameter in adamw_parameters if id(parameter) not in router_ids]
    router_names = [name for name in adamw_names if id(dict(model.named_parameters())[name]) in router_ids]

    groups: list[dict[str, object]] = []
    if residual_parameters:
        groups.append({"params": residual_parameters, "lr": adamw_lr, "base_lr": adamw_lr})
    if router_parameters:
        groups.append({"params": router_parameters, "lr": router_lr, "base_lr": router_lr})
    adamw = torch.optim.AdamW(
        groups,
        lr=adamw_lr,
        betas=adamw_betas,
        eps=1e-8,
        weight_decay=weight_decay,
    )
    muon: torch.optim.Optimizer | None = None
    if use_muon and muon_parameters:
        muon = torch.optim.Muon(
            muon_parameters,
            lr=muon_lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            nesterov=True,
            adjust_lr_fn="match_rms_adamw",
        )
        for group in muon.param_groups:
            group["base_lr"] = muon_lr
    elif muon_parameters:
        # Fair AdamW-only ablation owns every trainable parameter.
        adamw.add_param_group(
            {
                "params": muon_parameters,
                "lr": adamw_lr,
                "base_lr": adamw_lr,
                "weight_decay": weight_decay,
            }
        )
        adamw_names.extend(muon_names)
        muon_names = []
    return OptimizerBundle(muon, adamw, muon_names, adamw_names, router_names)


def cosine_warmup_scale(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_ratio + (1.0 - min_ratio) * cosine


def set_scheduled_learning_rates(
    bundle: OptimizerBundle,
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float = 0.1,
) -> float:
    scale = cosine_warmup_scale(step, total_steps, warmup_steps, min_ratio)
    if bundle.muon is not None:
        for group in bundle.muon.param_groups:
            group["lr"] = float(group["base_lr"]) * scale
    for group in bundle.adamw.param_groups:
        group["lr"] = float(group["base_lr"]) * scale
    return scale


@torch.no_grad()
def global_grad_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    gradients = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0)
    return torch.stack(gradients).norm(2)
