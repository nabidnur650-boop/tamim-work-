from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from .modeling import BoichitroConfig, BoichitroForMultiTask


Variant = Literal["M0", "M1", "M2", "M3"]


def compatible_state_load(
    model: BoichitroForMultiTask, state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    destination = model.state_dict()
    copied: list[str] = []
    skipped_shape: list[str] = []
    unexpected: list[str] = []
    for name, value in state.items():
        if name not in destination:
            unexpected.append(name)
        elif destination[name].shape != value.shape:
            skipped_shape.append(name)
        else:
            destination[name].copy_(value)
            copied.append(name)
    missing = sorted(set(destination) - set(copied))
    return {
        "copied_tensors": len(copied),
        "missing_tensors": missing,
        "shape_mismatch_tensors": skipped_shape,
        "unexpected_tensors": unexpected,
    }


def task_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    variant: Variant,
    n_sources: int,
    ablations: set[str] | None = None,
) -> tuple[BoichitroForMultiTask, dict[str, Any]]:
    """Create a task model while preserving every shape-compatible base tensor."""

    ablations = ablations or set()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    values = dict(payload["model_config"])
    expected_architecture = {
        "M0": "dense",
        "M1": "switch",
        "M2": "standard_moe",
        "M3": "boichitro_moe",
    }[variant]
    source_architecture = str(values["architecture"])
    if variant == "M3" and source_architecture != "standard_moe":
        raise ValueError("M3 task adaptation must start from the common M2 continuation")
    if variant != "M3" and source_architecture != expected_architecture:
        raise ValueError(
            f"{variant} expects {expected_architecture}, checkpoint is {source_architecture}"
        )
    permanent_paired_bank = (
        source_architecture == "standard_moe"
        and float(values.get("banked_upcycle_fraction", 0.0)) == 1.0
        and float(values.get("banked_upcycle_release_fraction", 0.0)) == 1.0
    )
    values.update(
        architecture=expected_architecture,
        n_sources=n_sources,
        max_seq_len=max(256, int(values["max_seq_len"])),
        use_classification_head=True,
        use_dialect_aux_head="no_dialect_head" not in ablations,
        use_source_adversary=variant == "M3" and "no_source_adversary" not in ablations,
        use_task_conditioning=variant == "M3" and "no_task_conditioning" not in ablations,
        use_lexical_routing_prior=(
            variant == "M3"
            and "no_lexical_prior" not in ablations
        ),
        randomize_lexical_prior="randomized_lexical_prior" in ablations,
        bidirectional_attention="bidirectional" in ablations,
        use_mtp="no_mtp" not in ablations,
        # Preserve the validation-selected permanent complementary-bank
        # topology. Legacy/abrupt Stage-U checkpoints are explicitly disabled
        # so task progress cannot accidentally re-enable a transient bank.
        banked_upcycle_fraction=1.0 if permanent_paired_bank else 0.0,
        banked_upcycle_release_fraction=1.0 if permanent_paired_bank else 0.0,
    )
    model = BoichitroForMultiTask(BoichitroConfig.from_dict(values))
    load_report = compatible_state_load(model, payload["model_state_dict"])
    report = {
        "variant": variant,
        "source_checkpoint": str(checkpoint_path),
        "source_architecture": source_architecture,
        "destination_architecture": expected_architecture,
        "source_tokens_seen": int(payload.get("tokens_seen", 0)),
        "permanent_paired_bank_routing": permanent_paired_bank,
        "ablations": sorted(ablations),
        **load_report,
    }
    return model, report
