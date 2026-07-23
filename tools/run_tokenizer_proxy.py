#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.modeling import BoichitroConfig, BoichitroForMultiTask  # noqa: E402
from boichitro.optim import cosine_warmup_scale  # noqa: E402
from boichitro.tokenization import (  # noqa: E402
    DIALECTS,
    SPECIAL_TOKENS,
    assert_special_tokens,
    load_tokenizer,
    nfc,
    sha256_file,
    sha256_tree,
)
from train_tokenizer_candidates import load_selection_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compute-matched causal-LM tokenizer proxies.")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/tokenizer_proxy.yaml")
    parser.add_argument("--resume", action="store_true", help="Reuse completed seed metrics.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_documents(tokenizer, texts: list[str]) -> list[list[int]]:
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    documents: list[list[int]] = []
    chunk = 4096
    for start in range(0, len(texts), chunk):
        encodings = tokenizer.encode_batch(texts[start : start + chunk], add_special_tokens=False)
        documents.extend([[bos, *encoding.ids, eos] for encoding in encodings if encoding.ids])
    return documents


def packed_blocks(
    documents: list[list[int]], *, token_budget: int, block_size: int, seed: int
) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    stream: list[int] = []
    while len(stream) < token_budget + block_size:
        for index in generator.permutation(len(documents)):
            stream.extend(documents[int(index)])
            if len(stream) >= token_budget + block_size:
                break
    block_count = math.ceil(token_budget / block_size)
    required = block_count * block_size
    array = np.asarray(stream[:required], dtype=np.int64).reshape(block_count, block_size)
    return torch.from_numpy(array)


def sample_evaluation(frame: pd.DataFrame, rows_per_dialect: int, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for offset, (_, group) in enumerate(frame.groupby("dialect", sort=True, observed=True)):
        parts.append(
            group.sample(
                n=min(rows_per_dialect, len(group)),
                random_state=seed + offset * 7919,
                replace=False,
            )
        )
    return pd.concat(parts, ignore_index=True).sort_values("row_id").reset_index(drop=True)


@torch.inference_mode()
def evaluate(
    model: BoichitroForMultiTask,
    tokenizer,
    frame: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
    block_size: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    pad = tokenizer.token_to_id("<pad>")
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    rows: list[dict[str, object]] = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        encoded = []
        for text in batch["text_model"]:
            ids = tokenizer.encode(nfc(text), add_special_tokens=False).ids
            encoded.append([bos, *ids[: block_size - 2], eos])
        length = max(map(len, encoded))
        input_ids = torch.full((len(encoded), length), pad, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for index, ids in enumerate(encoded):
            input_ids[index, : len(ids)] = torch.tensor(ids, device=device)
            attention[index, : len(ids)] = 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(input_ids, attention_mask=attention)
        logits = output["logits"][:, :-1].float()
        targets = input_ids[:, 1:]
        valid = attention[:, 1:].bool()
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).view(targets.shape)
        for local_index, (_, row) in enumerate(batch.iterrows()):
            nll = float(token_loss[local_index][valid[local_index]].sum().cpu())
            predicted_tokens = int(valid[local_index].sum().cpu())
            characters = max(1, len(nfc(row["text_model"])))
            rows.append(
                {
                    "row_id": row["row_id"],
                    "dialect": row["dialect"],
                    "source_id": row["source_id"],
                    "nll": nll,
                    "predicted_tokens": predicted_tokens,
                    "characters": characters,
                }
            )
    detail = pd.DataFrame(rows)

    def summarize(group: pd.DataFrame) -> pd.Series:
        nll = group["nll"].sum()
        tokens = group["predicted_tokens"].sum()
        characters = group["characters"].sum()
        return pd.Series(
            {
                "rows": len(group),
                "nll": nll,
                "predicted_tokens": tokens,
                "characters": characters,
                "nll_per_token": nll / max(1, tokens),
                "perplexity": math.exp(min(20.0, nll / max(1, tokens))),
                "bits_per_character": nll / math.log(2.0) / max(1, characters),
            }
        )

    by_dialect = detail.groupby("dialect", observed=True).apply(summarize, include_groups=False).reset_index()
    overall = summarize(detail).to_dict()
    overall["worst_dialect_bpc"] = float(by_dialect["bits_per_character"].max())
    overall["best_dialect_bpc"] = float(by_dialect["bits_per_character"].min())
    overall["dialect_bpc_range"] = overall["worst_dialect_bpc"] - overall["best_dialect_bpc"]
    return {key: float(value) for key, value in overall.items()}, by_dialect


def freeze_selected(
    selected_id: str,
    candidate_root: Path,
    frozen_dir: Path,
    summary: pd.DataFrame,
    proxy_config: dict,
    evaluation_hash: str,
) -> dict[str, object]:
    if frozen_dir.exists():
        shutil.rmtree(frozen_dir)
    shutil.copytree(candidate_root / selected_id, frozen_dir)
    tokenizer = load_tokenizer(frozen_dir)
    assert_special_tokens(tokenizer)
    selected = summary.loc[summary["candidate_id"] == selected_id].iloc[0].to_dict()
    manifest = {
        "status": "FROZEN_AFTER_THREE_SEED_CAUSAL_LM_PROXY",
        "selected_id": selected_id,
        "content_tree_sha256": sha256_tree(frozen_dir),
        "tokenizer_json_sha256": sha256_file(frozen_dir / "tokenizer.json"),
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "special_token_ids": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "selection_metrics": selected,
        "proxy_config": proxy_config,
        "selection_frame_sha256": evaluation_hash,
        "test_data_used_for_selection": False,
    }
    (frozen_dir / "tokenizer_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    manifest["manifest_sha256"] = sha256_file(frozen_dir / "tokenizer_manifest.json")
    return manifest


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if not torch.cuda.is_available():
        raise RuntimeError("Tokenizer proxy is registered as a CUDA experiment")
    device = torch.device("cuda")

    intrinsic_path = PROJECT / "reports/tokenizer/tokenizer_intrinsic_screen.csv"
    intrinsic = pd.read_csv(intrinsic_path)
    shortlisted = intrinsic.loc[intrinsic["proxy_shortlisted"], "candidate_id"].tolist()
    if not shortlisted:
        raise RuntimeError("No tokenizer candidates were shortlisted")
    manifest = pd.read_parquet(PROJECT / "data/manifests/tokenizer_train_v1.parquet")
    train_texts = manifest["text_model"].dropna().map(nfc).drop_duplicates().tolist()
    selection = load_selection_frame(PROJECT / "data/final/v1")
    selection = sample_evaluation(
        selection,
        rows_per_dialect=int(config["evaluation_rows_per_dialect"]),
        seed=91173,
    )
    evaluation_hash = hashlib.sha256(
        "\n".join(selection["row_id"].astype(str)).encode("utf-8")
    ).hexdigest()

    output_root = PROJECT / "runs/tokenizer_proxy"
    output_root.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, object]] = []
    dialect_frames: list[pd.DataFrame] = []
    candidate_root = PROJECT / "artifacts/tokenizers/candidates"
    budget = int(config["training_token_budget_per_seed"])
    block_size = int(config["block_size"])
    micro_batch = int(config["micro_batch_size"])

    for candidate_id in shortlisted:
        tokenizer = load_tokenizer(candidate_root / candidate_id)
        assert_special_tokens(tokenizer)
        print(f"Encoding training documents for {candidate_id}...", flush=True)
        documents = encode_documents(tokenizer, train_texts)
        vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        for seed in config["seeds"]:
            seed = int(seed)
            run_dir = output_root / candidate_id / str(seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            if args.resume and metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                result_rows.append(metrics)
                dialect = pd.read_csv(run_dir / "metrics_by_dialect.csv")
                dialect_frames.append(dialect)
                print(f"[resume] {candidate_id}/{seed}", flush=True)
                continue

            set_seed(seed)
            blocks = packed_blocks(documents, token_budget=budget, block_size=block_size, seed=seed)
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                TensorDataset(blocks),
                batch_size=micro_batch,
                shuffle=True,
                generator=generator,
                num_workers=2,
                pin_memory=True,
                drop_last=False,
            )
            model_config = BoichitroConfig(
                vocab_size=vocab_size,
                architecture="dense",
                max_seq_len=block_size,
                n_layers=int(config["model"]["n_layers"]),
                d_model=int(config["model"]["d_model"]),
                n_heads=int(config["model"]["n_heads"]),
                n_kv_heads=int(config["model"]["n_kv_heads"]),
                dense_ffn_dim=int(config["model"]["dense_ffn_dim"]),
                dense_prefix_layers=int(config["model"]["n_layers"]),
                dropout=float(config["model"]["dropout"]),
                use_mtp=False,
                use_classification_head=False,
                use_dialect_aux_head=False,
            )
            model = BoichitroForMultiTask(model_config).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["learning_rate"]),
                betas=(0.9, 0.95),
                weight_decay=float(config["weight_decay"]),
            )
            total_steps = len(loader)
            warmup = max(1, round(total_steps * float(config["warmup_fraction"])))
            model.train()
            started = time.perf_counter()
            seen_tokens = 0
            loss_sum = 0.0
            torch.cuda.reset_peak_memory_stats()
            for step, (batch,) in enumerate(loader):
                scale = cosine_warmup_scale(
                    step, total_steps, warmup, float(config["min_lr_ratio"])
                )
                for group in optimizer.param_groups:
                    group["lr"] = float(config["learning_rate"]) * scale
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch, labels=batch)
                    loss = output["loss"]
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["gradient_clip"])
                )
                optimizer.step()
                seen_tokens += batch.numel()
                loss_sum += float(loss.detach())
                if (step + 1) % 25 == 0 or step + 1 == total_steps:
                    elapsed = time.perf_counter() - started
                    print(
                        f"[{candidate_id}/{seed}] {step + 1:04d}/{total_steps:04d} "
                        f"loss={loss_sum / (step + 1):.4f} tok/s={seen_tokens / elapsed:,.0f} "
                        f"grad={float(gradient_norm):.3f}",
                        flush=True,
                    )
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - started
            evaluation, by_dialect = evaluate(
                model,
                tokenizer,
                selection,
                device=device,
                batch_size=micro_batch,
                block_size=block_size,
            )
            metrics = {
                "candidate_id": candidate_id,
                "seed": seed,
                "vocab_size": vocab_size,
                **model.parameter_report(),
                "training_tokens": seen_tokens,
                "training_steps": total_steps,
                "mean_training_loss": loss_sum / total_steps,
                "training_seconds": train_seconds,
                "training_tokens_per_second": seen_tokens / train_seconds,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                **{f"validation_{key}": value for key, value in evaluation.items()},
            }
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            by_dialect.insert(0, "seed", seed)
            by_dialect.insert(0, "candidate_id", candidate_id)
            by_dialect.to_csv(run_dir / "metrics_by_dialect.csv", index=False)
            torch.save(
                {
                    "model_config": model_config.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "candidate_id": candidate_id,
                    "seed": seed,
                },
                run_dir / "final_checkpoint.pt",
            )
            result_rows.append(metrics)
            dialect_frames.append(by_dialect)
            del model, optimizer, blocks, loader
            gc.collect()
            torch.cuda.empty_cache()

    results = pd.DataFrame(result_rows)
    results.to_csv(PROJECT / "reports/tokenizer/tokenizer_proxy_seed_metrics.csv", index=False)
    pd.concat(dialect_frames, ignore_index=True).to_csv(
        PROJECT / "reports/tokenizer/tokenizer_proxy_dialect_metrics.csv", index=False
    )
    aggregated = (
        results.groupby("candidate_id", observed=True)
        .agg(
            seeds=("seed", "nunique"),
            mean_bpc=("validation_bits_per_character", "mean"),
            std_bpc=("validation_bits_per_character", "std"),
            mean_worst_dialect_bpc=("validation_worst_dialect_bpc", "mean"),
            std_worst_dialect_bpc=("validation_worst_dialect_bpc", "std"),
            mean_tokens_per_second=("training_tokens_per_second", "mean"),
            mean_peak_memory_gib=("peak_memory_gib", "mean"),
            total_parameters=("total_parameters", "first"),
            vocab_size=("vocab_size", "first"),
        )
        .reset_index()
    )
    aggregated = aggregated.merge(
        intrinsic[["candidate_id", "tokens_per_character", "intrinsic_score"]],
        on="candidate_id",
        how="left",
    )
    rule = config["selection"]
    best_bpc = aggregated["mean_bpc"].min()
    best_cost = aggregated["tokens_per_character"].min()
    aggregated["passes_global_bpc_gate"] = aggregated["mean_bpc"] <= (
        float(rule["global_bpc_relative_to_best_max"]) * best_bpc
    )
    aggregated["passes_token_cost_gate"] = aggregated["tokens_per_character"] <= (
        float(rule["token_cost_relative_to_best_max"]) * best_cost
    )
    eligible = aggregated.loc[
        aggregated["passes_global_bpc_gate"] & aggregated["passes_token_cost_gate"]
    ]
    if eligible.empty:
        raise RuntimeError("No tokenizer passed the frozen proxy selection gates")
    selected_id = str(eligible.sort_values("mean_worst_dialect_bpc").iloc[0]["candidate_id"])
    aggregated["selected"] = aggregated["candidate_id"].eq(selected_id)
    aggregated.to_csv(PROJECT / "reports/tokenizer/tokenizer_proxy_summary.csv", index=False)

    frozen_dir = PROJECT / "artifacts/tokenizers/frozen"
    frozen_manifest = freeze_selected(
        selected_id,
        candidate_root,
        frozen_dir,
        aggregated,
        config,
        evaluation_hash,
    )
    final_report = {
        "status": "TOKENIZER_FROZEN",
        "selected_id": selected_id,
        "frozen_dir": str(frozen_dir.relative_to(PROJECT)),
        "frozen_tree_sha256": sha256_tree(frozen_dir),
        "content_tree_sha256": frozen_manifest["content_tree_sha256"],
        "manifest_sha256": frozen_manifest["manifest_sha256"],
        "candidate_count": len(shortlisted),
        "seeds": list(map(int, config["seeds"])),
        "training_token_budget_per_seed": budget,
        "selection_rows": len(selection),
        "selection_frame_sha256": evaluation_hash,
        "test_data_used_for_selection": False,
        "hardware": torch.cuda.get_device_name(0),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    (PROJECT / "reports/tokenizer/TOKENIZER_FREEZE_REPORT.json").write_text(
        json.dumps(final_report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nFROZEN TOKENIZER: {selected_id}")
    print(f"Path: {frozen_dir}")
    print(aggregated.sort_values("mean_worst_dialect_bpc").to_string(index=False))


if __name__ == "__main__":
    main()
