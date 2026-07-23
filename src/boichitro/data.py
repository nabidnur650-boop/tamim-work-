from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from .tokenization import DIALECTS, nfc


DIALECT_TO_ID: dict[str, int] = {dialect: index for index, dialect in enumerate(DIALECTS)}
TASK_TO_ID: dict[str, int] = {"clm": 0, "normalization": 1, "identification": 2}


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class EncodedExample:
    row_id: str
    task: str
    input_ids: list[int]
    labels: list[int]
    task_id: int
    dialect_label: int
    classification_label: int
    source_label: int
    group_id: int
    example_weight: float
    dialect: str
    source_id: str
    semantic_group_id: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _special_ids(tokenizer: Tokenizer) -> dict[str, int]:
    names = (
        "<pad>",
        "<bos>",
        "<eos>",
        "<sep>",
        "<cls>",
        "<task_norm>",
        "<task_id>",
        "<task_clm>",
        "<dial_unknown>",
    )
    result = {name: tokenizer.token_to_id(name) for name in names}
    if any(value is None for value in result.values()):
        raise ValueError(f"Tokenizer is missing a task special token: {result}")
    return {name: int(value) for name, value in result.items()}


def _truncate_pair(source: list[int], target: list[int], available: int) -> tuple[list[int], list[int]]:
    if len(source) + len(target) <= available:
        return source, target
    target_budget = min(len(target), max(8, available // 2))
    source_budget = max(1, available - target_budget)
    target_budget = max(1, available - min(len(source), source_budget))
    return source[:source_budget], target[:target_budget]


def encode_normalization(
    row: Mapping[str, Any],
    tokenizer: Tokenizer,
    *,
    max_length: int,
    source_to_id: Mapping[str, int],
    group_to_id: Mapping[str, int],
) -> EncodedExample:
    ids = _special_ids(tokenizer)
    source = tokenizer.encode(nfc(row["source_text_model"]), add_special_tokens=False).ids
    target = tokenizer.encode(nfc(row["target_text_model"]), add_special_tokens=False).ids
    fixed = 5  # bos, task, unknown dialect, sep, eos
    source, target = _truncate_pair(source, target, max_length - fixed)
    prefix = [ids["<bos>"], ids["<task_norm>"], ids["<dial_unknown>"], *source, ids["<sep>"]]
    answer = [*target, ids["<eos>"]]
    input_ids = prefix + answer
    labels = [-100] * len(prefix) + answer
    dialect = str(row["dialect"])
    source_id = str(row["source_id"])
    synthetic = bool(row.get("is_synthetic", False))
    group_key = f"{dialect}|{source_id}|{'synthetic' if synthetic else 'authentic'}"
    semantic = str(row.get("semantic_group_id") or row["row_id"])
    return EncodedExample(
        row_id=str(row["row_id"]),
        task="normalization",
        input_ids=input_ids,
        labels=labels,
        task_id=TASK_TO_ID["normalization"],
        dialect_label=DIALECT_TO_ID[dialect],
        classification_label=-100,
        source_label=int(source_to_id.get(source_id, -100)),
        group_id=int(group_to_id.get(group_key, -100)),
        example_weight=float(row.get("example_loss_weight", 1.0)),
        dialect=dialect,
        source_id=source_id,
        semantic_group_id=semantic,
    )


def encode_identification(
    row: Mapping[str, Any],
    tokenizer: Tokenizer,
    *,
    max_length: int,
    source_to_id: Mapping[str, int],
    group_to_id: Mapping[str, int],
) -> EncodedExample:
    ids = _special_ids(tokenizer)
    text = tokenizer.encode(nfc(row["text_model"]), add_special_tokens=False).ids
    text = text[: max(1, max_length - 4)]
    input_ids = [ids["<bos>"], ids["<task_id>"], ids["<dial_unknown>"], *text, ids["<cls>"]]
    dialect = str(row["dialect"])
    source_id = str(row["source_id"])
    synthetic = bool(row.get("is_synthetic", False))
    group_key = f"{dialect}|{source_id}|{'synthetic' if synthetic else 'authentic'}"
    return EncodedExample(
        row_id=str(row["row_id"]),
        task="identification",
        input_ids=input_ids,
        labels=[-100] * len(input_ids),
        task_id=TASK_TO_ID["identification"],
        dialect_label=DIALECT_TO_ID[dialect],
        classification_label=DIALECT_TO_ID[dialect],
        source_label=int(source_to_id.get(source_id, -100)),
        group_id=int(group_to_id.get(group_key, -100)),
        example_weight=1.0,
        dialect=dialect,
        source_id=source_id,
        semantic_group_id=str(row.get("row_id")),
    )


def encode_clm(
    row: Mapping[str, Any],
    tokenizer: Tokenizer,
    *,
    max_length: int,
    source_to_id: Mapping[str, int],
    group_to_id: Mapping[str, int],
) -> EncodedExample:
    ids = _special_ids(tokenizer)
    text_column = "text_model" if "text_model" in row else "source_text_model"
    text = tokenizer.encode(nfc(row[text_column]), add_special_tokens=False).ids
    text = text[: max(1, max_length - 4)]
    prefix = [ids["<bos>"], ids["<task_clm>"], ids["<dial_unknown>"]]
    answer = [*text, ids["<eos>"]]
    dialect = str(row.get("dialect", "STD"))
    source_id = str(row.get("source_id", "general_replay"))
    synthetic = bool(row.get("is_synthetic", False))
    group_key = f"{dialect}|{source_id}|{'synthetic' if synthetic else 'authentic'}"
    return EncodedExample(
        row_id=str(row.get("row_id", stable_id(nfc(row[text_column])))),
        task="clm",
        input_ids=prefix + answer,
        labels=[-100] * len(prefix) + answer,
        task_id=TASK_TO_ID["clm"],
        dialect_label=DIALECT_TO_ID.get(dialect, -100),
        classification_label=-100,
        source_label=int(source_to_id.get(source_id, -100)),
        group_id=int(group_to_id.get(group_key, -100)),
        example_weight=float(row.get("example_loss_weight", 1.0)),
        dialect=dialect,
        source_id=source_id,
        semantic_group_id=str(row.get("semantic_group_id") or row.get("row_id", "")),
    )


class EncodedDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: Sequence[EncodedExample | dict[str, Any]]) -> None:
        self.examples = [example.to_dict() if isinstance(example, EncodedExample) else example for example in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def exclude_sources(dataset: EncodedDataset, source_ids: Sequence[str]) -> EncodedDataset:
    blocked = frozenset(str(value) for value in source_ids)
    return EncodedDataset(
        [
            example
            for example in dataset.examples
            if str(example.get("source_id")) not in blocked
        ]
    )


def renormalize_proportions(
    proportions: Mapping[str, float], excluded: Sequence[str]
) -> dict[str, float]:
    blocked = frozenset(str(value) for value in excluded)
    retained = {
        str(name): float(value)
        for name, value in proportions.items()
        if str(name) not in blocked
    }
    total = sum(retained.values())
    if total <= 0:
        raise ValueError("At least one positive mixture component must remain")
    return {name: value / total for name, value in retained.items()}


class FixedMixtureDataset(Dataset[dict[str, Any]]):
    """A deterministic finite epoch sampled from named component datasets."""

    def __init__(
        self,
        components: Mapping[str, EncodedDataset],
        proportions: Mapping[str, float],
        *,
        epoch_examples: int,
        seed: int,
    ) -> None:
        missing = set(proportions) - set(components)
        if missing:
            raise ValueError(f"Mixture components missing: {sorted(missing)}")
        total = sum(float(value) for value in proportions.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Mixture proportions sum to {total}, expected 1")
        rng = random.Random(seed)
        exact = {name: epoch_examples * float(value) for name, value in proportions.items()}
        counts = {name: int(math.floor(value)) for name, value in exact.items()}
        remainder = epoch_examples - sum(counts.values())
        for name in sorted(exact, key=lambda item: exact[item] - counts[item], reverse=True)[:remainder]:
            counts[name] += 1
        schedule: list[tuple[str, int]] = []
        for name, count in counts.items():
            component = components[name]
            if len(component) == 0 and count:
                raise ValueError(f"Mixture component {name} is empty")
            for _ in range(count):
                schedule.append((name, rng.randrange(len(component))))
        rng.shuffle(schedule)
        self.components = dict(components)
        self.schedule = schedule

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name, component_index = self.schedule[index]
        return self.components[name][component_index]


class FixedTokenMixtureDataset(Dataset[dict[str, Any]]):
    """Deterministic sampling with proportions defined over non-padding tokens.

    Example-level mixtures substantially over-weight short classification rows.
    This sampler fills a fixed token quota independently for every component,
    then shuffles the combined schedule using the registered seed.
    """

    def __init__(
        self,
        components: Mapping[str, EncodedDataset],
        proportions: Mapping[str, float],
        *,
        token_budget: int,
        seed: int,
    ) -> None:
        missing = set(proportions) - set(components)
        if missing:
            raise ValueError(f"Mixture components missing: {sorted(missing)}")
        total = sum(float(value) for value in proportions.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Mixture proportions sum to {total}, expected 1")
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        rng = random.Random(seed)
        exact = {name: token_budget * float(value) for name, value in proportions.items()}
        quotas = {name: int(math.floor(value)) for name, value in exact.items()}
        remainder = token_budget - sum(quotas.values())
        for name in sorted(exact, key=lambda item: exact[item] - quotas[item], reverse=True)[
            :remainder
        ]:
            quotas[name] += 1

        schedule: list[tuple[str, int]] = []
        realized = {name: 0 for name in proportions}
        draws = {name: 0 for name in proportions}
        maximum_repeats = {name: 0 for name in proportions}
        for name, quota in quotas.items():
            component = components[name]
            if len(component) == 0 and quota:
                raise ValueError(f"Mixture component {name} is empty")
            order = list(range(len(component)))
            rng.shuffle(order)
            cursor = 0
            repetitions = [0] * len(component)
            while realized[name] < quota:
                if cursor == len(order):
                    rng.shuffle(order)
                    cursor = 0
                index = order[cursor]
                cursor += 1
                example = component[index]
                length = max(1, len(example["input_ids"]))
                schedule.append((name, index))
                realized[name] += length
                draws[name] += 1
                repetitions[index] += 1
            maximum_repeats[name] = max(repetitions, default=0)
        rng.shuffle(schedule)
        self.components = dict(components)
        self.schedule = schedule
        self.requested_tokens = quotas
        self.realized_tokens = realized
        self.draws = draws
        self.maximum_repeats = maximum_repeats

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name, component_index = self.schedule[index]
        return self.components[name][component_index]

    def report(self) -> dict[str, Any]:
        total = sum(self.realized_tokens.values())
        return {
            "examples": len(self),
            "requested_tokens": dict(self.requested_tokens),
            "realized_tokens": dict(self.realized_tokens),
            "draws": dict(self.draws),
            "maximum_example_repeats": dict(self.maximum_repeats),
            "realized_proportions": {
                name: value / max(1, total) for name, value in self.realized_tokens.items()
            },
        }


class MaskedNextTokenDataset(Dataset[dict[str, Any]]):
    """Deterministic 80/10/10 masking for bidirectional next-token adaptation.

    Labels remain at their original positions. The decoder's shifted language-
    model objective therefore predicts a masked token from the preceding hidden
    state, which can attend bidirectionally during MNTP adaptation.
    """

    def __init__(
        self,
        source: Dataset,
        *,
        mask_token_id: int,
        vocab_size: int,
        special_token_ids: Sequence[int],
        mask_probability: float = 0.15,
        seed: int,
    ) -> None:
        if not 0 < mask_probability <= 1:
            raise ValueError("mask_probability must be in (0, 1]")
        self.source = source
        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(vocab_size)
        self.special_token_ids = frozenset(int(value) for value in special_token_ids)
        self.mask_probability = float(mask_probability)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        original = self.source[index]
        example = dict(original)
        input_ids = list(original["input_ids"])
        labels = [-100] * len(input_ids)
        candidates = [
            position
            for position, token_id in enumerate(input_ids)
            if position > 0 and int(token_id) not in self.special_token_ids
        ]
        if not candidates:
            raise ValueError(f"MNTP example has no maskable tokens: {original['row_id']}")
        material = f"{self.seed}|{index}|{original['row_id']}".encode("utf-8")
        rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
        masked_count = max(1, round(len(candidates) * self.mask_probability))
        selected = rng.sample(candidates, min(masked_count, len(candidates)))
        for position in selected:
            original_token = int(input_ids[position])
            labels[position] = original_token
            draw = rng.random()
            if draw < 0.8:
                input_ids[position] = self.mask_token_id
            elif draw < 0.9:
                replacement = rng.randrange(self.vocab_size)
                while replacement in self.special_token_ids:
                    replacement = rng.randrange(self.vocab_size)
                input_ids[position] = replacement
        example.update(
            input_ids=input_ids,
            labels=labels,
            classification_label=-100,
            task="mntp",
        )
        return example


def collate_examples(examples: Sequence[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    batch_size = len(examples)
    max_length = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    attention = torch.zeros((batch_size, max_length), dtype=torch.long)
    for index, example in enumerate(examples):
        length = len(example["input_ids"])
        input_ids[index, :length] = torch.tensor(example["input_ids"], dtype=torch.long)
        labels[index, :length] = torch.tensor(example["labels"], dtype=torch.long)
        attention[index, :length] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention,
        "task_ids": torch.tensor([example["task_id"] for example in examples], dtype=torch.long),
        "dialect_labels": torch.tensor(
            [example["dialect_label"] for example in examples], dtype=torch.long
        ),
        "classification_labels": torch.tensor(
            [example["classification_label"] for example in examples], dtype=torch.long
        ),
        "source_labels": torch.tensor(
            [example["source_label"] for example in examples], dtype=torch.long
        ),
        "group_ids": torch.tensor([example["group_id"] for example in examples], dtype=torch.long),
        "example_weights": torch.tensor(
            [example["example_weight"] for example in examples], dtype=torch.float32
        ),
        "row_ids": [example["row_id"] for example in examples],
        "tasks": [example["task"] for example in examples],
        "dialects": [example["dialect"] for example in examples],
        "source_ids": [example["source_id"] for example in examples],
        "semantic_group_ids": [example["semantic_group_id"] for example in examples],
    }


def build_training_maps(normalization: pd.DataFrame, identification: pd.DataFrame) -> dict[str, Any]:
    sources = sorted(set(normalization["source_id"].astype(str)) | set(identification["source_id"].astype(str)))
    source_to_id = {source: index for index, source in enumerate(sources)}
    groups: set[str] = set()
    for frame in (normalization, identification):
        for row in frame.to_dict("records"):
            groups.add(
                f"{row['dialect']}|{row['source_id']}|"
                f"{'synthetic' if bool(row.get('is_synthetic', False)) else 'authentic'}"
            )
    group_to_id = {group: index for index, group in enumerate(sorted(groups))}
    return {
        "dialect_to_id": DIALECT_TO_ID,
        "task_to_id": TASK_TO_ID,
        "source_to_id": source_to_id,
        "group_to_id": group_to_id,
    }


def save_encoded_cache(path: Path, examples: Sequence[EncodedExample], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "examples": [example.to_dict() for example in examples],
        },
        path,
    )


def load_encoded_cache(path: Path) -> tuple[EncodedDataset, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return EncodedDataset(payload["examples"]), payload["metadata"]
