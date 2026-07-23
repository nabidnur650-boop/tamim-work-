from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from .data import EncodedDataset, TASK_TO_ID, collate_examples
from .tokenization import canonical_detokenized, nfc


@torch.inference_mode()
def predict_identification(
    model,
    dataset: EncodedDataset,
    *,
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
    temperature: float = 1.0,
    capture_routing: bool = False,
) -> tuple[pd.DataFrame, list[Any]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        collate_fn=lambda examples: collate_examples(examples, pad_token_id),
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    routing = []
    for batch in loader:
        tensor_keys = (
            "input_ids",
            "attention_mask",
            "task_ids",
        )
        tensors = {key: batch[key].to(device, non_blocking=True) for key in tensor_keys}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(
                **tensors,
                capture_routing=capture_routing,
                return_lm_logits=False,
            )
        logits = output["classification_logits"].float() / temperature
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        for index, row_id in enumerate(batch["row_ids"]):
            rows.append(
                {
                    "row_id": row_id,
                    "dialect": batch["dialects"][index],
                    "source_id": batch["source_ids"][index],
                    "label_id": int(batch["classification_labels"][index]),
                    "prediction_id": int(predictions[index]),
                    "probabilities": probabilities[index].tolist(),
                    "semantic_group_id": batch["semantic_group_ids"][index],
                }
            )
        if capture_routing:
            routing.append(output["routing"])
    return pd.DataFrame(rows), routing


def _normalization_prefix(tokenizer: Tokenizer, text: str, max_source_tokens: int) -> list[int]:
    required = ["<bos>", "<task_norm>", "<dial_unknown>", "<sep>"]
    special = {token: tokenizer.token_to_id(token) for token in required}
    if any(value is None for value in special.values()):
        raise ValueError(f"Tokenizer lacks normalization tokens: {special}")
    source = tokenizer.encode(nfc(text), add_special_tokens=False).ids[:max_source_tokens]
    return [
        int(special["<bos>"]),
        int(special["<task_norm>"]),
        int(special["<dial_unknown>"]),
        *source,
        int(special["<sep>"]),
    ]


@torch.inference_mode()
def greedy_normalize(
    model,
    tokenizer: Tokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int = 32,
    max_new_tokens: int = 96,
    use_kv_cache: bool = True,
) -> list[str]:
    model.eval()
    eos = int(tokenizer.token_to_id("<eos>"))
    pad = int(tokenizer.token_to_id("<pad>"))
    results: list[str] = []
    max_source = model.config.max_seq_len - max_new_tokens - 5
    for start in range(0, len(texts), batch_size):
        subset = texts[start : start + batch_size]
        prefixes = [_normalization_prefix(tokenizer, text, max_source) for text in subset]
        task_ids = torch.full(
            (len(prefixes),), TASK_TO_ID["normalization"], dtype=torch.long, device=device
        )
        if use_kv_cache:
            prompt_length = max(len(prefix) for prefix in prefixes)
            prompt = torch.full(
                (len(prefixes), prompt_length), pad, dtype=torch.long, device=device
            )
            attention = torch.zeros_like(prompt)
            for index, prefix in enumerate(prefixes):
                offset = prompt_length - len(prefix)
                prompt[index, offset:] = torch.tensor(prefix, device=device)
                attention[index, offset:] = 1
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(
                    prompt,
                    attention_mask=attention,
                    task_ids=task_ids,
                    use_cache=True,
                )
            past_key_values = output["past_key_values"]
            next_logits = output["logits"][:, -1].float()
            finished = torch.zeros(len(prefixes), dtype=torch.bool, device=device)
            generated: list[list[int]] = [[] for _ in prefixes]
            for generation_step in range(max_new_tokens):
                next_tokens = next_logits.argmax(dim=-1)
                for index in range(len(prefixes)):
                    if finished[index]:
                        continue
                    token = int(next_tokens[index])
                    if token == eos:
                        finished[index] = True
                    else:
                        generated[index].append(token)
                if bool(finished.all()) or generation_step + 1 == max_new_tokens:
                    break
                active = ~finished
                cache_input = torch.where(
                    active,
                    next_tokens,
                    torch.full_like(next_tokens, pad),
                ).unsqueeze(1)
                attention = torch.cat((attention, active.long().unsqueeze(1)), dim=1)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    output = model(
                        cache_input,
                        attention_mask=attention,
                        task_ids=task_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = output["past_key_values"]
                next_logits = output["logits"][:, -1].float()
            results.extend(
                canonical_detokenized(tokenizer.decode(ids, skip_special_tokens=True))
                for ids in generated
            )
            continue

        lengths = torch.tensor([len(prefix) for prefix in prefixes], device=device, dtype=torch.long)
        capacity = min(
            model.config.max_seq_len,
            max(len(prefix) for prefix in prefixes) + max_new_tokens,
        )
        sequences = torch.full((len(prefixes), capacity), pad, dtype=torch.long, device=device)
        for index, prefix in enumerate(prefixes):
            sequences[index, : len(prefix)] = torch.tensor(prefix, device=device)
        finished = torch.zeros(len(prefixes), dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in prefixes]
        for _ in range(max_new_tokens):
            active_length = int(lengths.max().item())
            attention = (
                torch.arange(active_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            ).long()
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(
                    sequences[:, :active_length],
                    attention_mask=attention,
                    task_ids=task_ids,
                )
            last_logits = output["logits"][
                torch.arange(len(prefixes), device=device), lengths - 1
            ].float()
            next_tokens = last_logits.argmax(dim=-1)
            for index in range(len(prefixes)):
                if finished[index] or lengths[index] >= capacity:
                    continue
                token = int(next_tokens[index])
                if token == eos:
                    finished[index] = True
                    continue
                sequences[index, lengths[index]] = token
                lengths[index] += 1
                generated[index].append(token)
            if bool(finished.all()):
                break
        results.extend(
            canonical_detokenized(tokenizer.decode(ids, skip_special_tokens=True))
            for ids in generated
        )
    return results


@torch.inference_mode()
def trace_routing(
    model,
    dataset: EncodedDataset,
    *,
    device: torch.device,
    pad_token_id: int,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Return per-example, per-layer dropless expert counts on real tokens."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        collate_fn=lambda examples: collate_examples(examples, pad_token_id),
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        tensors = {
            key: batch[key].to(device, non_blocking=True)
            for key in ("input_ids", "attention_mask", "task_ids")
        }
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(**tensors, capture_routing=True, return_lm_logits=False)
        if not output["routing"]:
            continue
        lengths = batch["attention_mask"].sum(dim=1).tolist()
        for example_index, row_id in enumerate(batch["row_ids"]):
            for relative_layer, routing in enumerate(output["routing"]):
                selected = routing.selected_experts[
                    example_index, : int(lengths[example_index])
                ].reshape(-1)
                counts = torch.bincount(
                    selected, minlength=model.config.n_routed_experts
                ).cpu()
                probabilities = counts.float() / counts.sum().clamp_min(1)
                entropy = -(
                    probabilities * probabilities.clamp_min(1e-9).log()
                ).sum()
                rows.append(
                    {
                        "row_id": row_id,
                        "task": batch["tasks"][example_index],
                        "dialect": batch["dialects"][example_index],
                        "source_id": batch["source_ids"][example_index],
                        "semantic_group_id": batch["semantic_group_ids"][example_index],
                        "layer_index": model.config.dense_prefix_layers + relative_layer,
                        "expert_counts": counts.tolist(),
                        "assignments": int(counts.sum()),
                        "expert_entropy": float(entropy),
                    }
                )
    return pd.DataFrame(rows)
