#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.modeling import BoichitroConfig, BoichitroForMultiTask  # noqa: E402
from boichitro.optim import build_optimizers  # noqa: E402
from boichitro.tokenization import load_tokenizer  # noqa: E402


def model_config(model_id: str, vocab_size: int, common: dict) -> BoichitroConfig:
    values = {
        "vocab_size": vocab_size,
        **common,
        "n_sources": 32,
        "use_classification_head": True,
        "use_dialect_aux_head": True,
        "use_mtp": True,
    }
    if model_id == "M0_DENSE":
        values.update(architecture="dense")
    elif model_id == "M1_SWITCH":
        values.update(
            architecture="switch",
            top_k=1,
            shared_expert=False,
            expert_dim=int(common["dense_ffn_dim"]),
            balance_loss_weight=0.01,
        )
    elif model_id == "M2_STANDARD_MOE":
        values.update(
            architecture="standard_moe",
            use_task_conditioning=False,
            use_lexical_routing_prior=False,
            use_source_adversary=False,
        )
    elif model_id == "M3_BOICHITRO":
        values.update(
            architecture="boichitro_moe",
            use_task_conditioning=True,
            use_lexical_routing_prior=True,
            use_source_adversary=True,
        )
    else:
        raise ValueError(model_id)
    return BoichitroConfig.from_dict(values)


def benchmark_one(
    model_id: str,
    config: BoichitroConfig,
    *,
    batch_size: int,
    sequence_length: int,
    timed_steps: int,
) -> dict:
    torch.manual_seed(1701)
    device = torch.device("cuda")
    model = BoichitroForMultiTask(config).to(device)
    bundle = build_optimizers(
        model,
        muon_lr=0.02,
        adamw_lr=3e-4,
        router_lr=2e-4,
        weight_decay=0.1,
        use_muon=True,
    )
    input_ids = torch.randint(
        24, config.vocab_size, (batch_size, sequence_length), device=device
    )
    task_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
    dialect = torch.randint(0, config.n_dialects, (batch_size,), device=device)
    sources = torch.randint(0, config.n_sources, (batch_size,), device=device)
    classification = torch.full((batch_size,), -100, dtype=torch.long, device=device)

    def step() -> float:
        bundle.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids,
                labels=input_ids,
                task_ids=task_ids,
                dialect_labels=dialect,
                source_labels=sources if config.use_source_adversary else None,
                classification_labels=classification,
            )
            loss = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        bundle.step()
        model.update_router_biases()
        return float(loss.detach())

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    losses = []
    for _ in range(timed_steps):
        started = time.perf_counter()
        losses.append(step())
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    tokens = batch_size * sequence_length
    report = {
        "model_id": model_id,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "tokens_per_step": tokens,
        "median_step_seconds": statistics.median(durations),
        "mean_step_seconds": statistics.mean(durations),
        "tokens_per_second": tokens / statistics.mean(durations),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "last_loss": losses[-1],
        **model.parameter_report(),
        "optimizer_muon_tensors": len(bundle.muon_names),
        "optimizer_adamw_tensors": len(bundle.adamw_names),
        "grouped_gemm": config.use_grouped_gemm,
        "mtp": config.use_mtp,
    }
    del model, bundle, input_ids
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_float32_matmul_precision("high")
    tokenizer = load_tokenizer(PROJECT / "artifacts/tokenizers/frozen")
    registry = yaml.safe_load((PROJECT / "configs/model/main_80m.yaml").read_text())
    common = dict(registry["model"])
    common.pop("use_mtp", None)
    rows = []
    for model_id in ("M0_DENSE", "M1_SWITCH", "M2_STANDARD_MOE", "M3_BOICHITRO"):
        config = model_config(model_id, tokenizer.get_vocab_size(), common)
        print(f"Benchmarking {model_id}...", flush=True)
        try:
            row = benchmark_one(
                model_id,
                config,
                batch_size=4,
                sequence_length=512,
                timed_steps=5,
            )
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            row = {"model_id": model_id, "status": "OOM", "error": str(error)}
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    frame = pd.DataFrame(rows)
    report_dir = PROJECT / "reports/model"
    report_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_dir / "gb10_model_benchmark.csv", index=False)
    (report_dir / "gb10_model_benchmark.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
