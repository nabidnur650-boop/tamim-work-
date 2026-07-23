#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import collate_examples, load_encoded_cache  # noqa: E402
from boichitro.experiments import task_model_from_checkpoint  # noqa: E402
from boichitro.optim import build_optimizers  # noqa: E402
from boichitro.tokenization import load_tokenizer, sha256_file  # noqa: E402
from boichitro.training import set_seed  # noqa: E402


def main() -> None:
    set_seed(8671)
    tokenizer = load_tokenizer(PROJECT / "artifacts/tokenizers/frozen")
    pad_token_id = int(tokenizer.token_to_id("<pad>"))
    maps = json.loads((PROJECT / "cache/tasks/maps.json").read_text(encoding="utf-8"))
    examples = []
    for name in (
        "general_replay_train.pt",
        "normalization_train.pt",
        "identification_train.pt",
    ):
        dataset, _ = load_encoded_cache(PROJECT / "cache/tasks" / name)
        examples.extend(dataset.examples[:2])
    batch = collate_examples(examples, pad_token_id)
    checkpoints = {
        "M0": PROJECT / "runs/U_M0_DENSE_200M/1701/final_checkpoint.pt",
        "M1": PROJECT / "runs/U_M1_SWITCH_200M/1701/final_checkpoint.pt",
        "M2": PROJECT / "runs/U_M2_STANDARD_MOE_200M/1701/final_checkpoint.pt",
        "M3": PROJECT / "runs/U_M2_STANDARD_MOE_200M/1701/final_checkpoint.pt",
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_keys = (
        "input_ids",
        "labels",
        "attention_mask",
        "task_ids",
        "dialect_labels",
        "classification_labels",
        "source_labels",
        "example_weights",
    )
    moved = {key: batch[key].to(device) for key in tensor_keys}
    checkpoint_hashes: dict[str, str] = {}
    results = []
    for variant, checkpoint in checkpoints.items():
        relative = str(checkpoint.relative_to(PROJECT))
        checkpoint_hashes.setdefault(relative, sha256_file(checkpoint))
        model, initialization = task_model_from_checkpoint(
            checkpoint,
            variant=variant,
            n_sources=len(maps["source_to_id"]),
        )
        model = model.to(device).train()
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(**moved, capture_routing=True)
        if not bool(torch.isfinite(output["loss"])):
            raise FloatingPointError(f"Non-finite preflight loss for {variant}")
        output["loss"].backward()
        optimizers = build_optimizers(
            model,
            muon_lr=0.01,
            adamw_lr=1.5e-4,
            router_lr=2e-4,
            weight_decay=0.1,
        )
        routed = sum(int(item.counts.sum()) for item in output["routing"])
        expected = (
            int(batch["attention_mask"].sum())
            * model.config.top_k
            * len(output["routing"])
        )
        if routed != expected:
            raise AssertionError(
                f"Dropped or duplicated routes for {variant}: {routed} != {expected}"
            )
        router_gradient = sum(
            float(parameter.grad.detach().abs().sum())
            for name, parameter in model.named_parameters()
            if "router" in name and parameter.grad is not None
        )
        if output["routing"] and router_gradient <= 0:
            raise AssertionError(f"No router gradient for {variant}")
        parameters = model.parameter_report()
        results.append(
            {
                "variant": variant,
                "source_checkpoint": relative,
                "loss": float(output["loss"].detach()),
                "total_parameters": parameters["total_parameters"],
                "active_parameters_per_token": parameters[
                    "active_parameters_per_token"
                ],
                "routed_assignments": routed,
                "expected_assignments": expected,
                "router_gradient_l1": router_gradient,
                "muon_parameter_tensors": len(optimizers.muon_names),
                "adamw_parameter_tensors": len(optimizers.adamw_names),
                "permanent_paired_bank_routing": initialization[
                    "permanent_paired_bank_routing"
                ],
                "source_adversary": model.config.use_source_adversary,
                "task_conditioning": model.config.use_task_conditioning,
                "causal_dialect_evidence_prior": model.config.use_lexical_routing_prior,
            }
        )
        del optimizers, output, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "status": "PASS_VALIDATION_ONLY",
        "test_data_access": False,
        "seed": 8671,
        "train_examples": len(examples),
        "tokenizer_sha256": sha256_file(
            PROJECT / "artifacts/tokenizers/frozen/tokenizer.json"
        ),
        "checkpoint_sha256": checkpoint_hashes,
        "models": results,
    }
    output_path = PROJECT / "reports/model/task_model_preflight.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
