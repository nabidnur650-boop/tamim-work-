from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedMemmapDataset(Dataset[torch.Tensor]):
    def __init__(self, path: Path, block_size: int, block_count: int) -> None:
        self.path = path
        self.block_size = block_size
        self.block_count = block_count
        expected_bytes = block_size * block_count * np.dtype(np.uint16).itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Packed file size mismatch: {path.stat().st_size} != {expected_bytes}"
            )
        self.array = np.memmap(path, mode="r", dtype=np.uint16).reshape(block_count, block_size)

    def __len__(self) -> int:
        return self.block_count

    def __getitem__(self, index: int) -> torch.Tensor:
        # Copy detaches from the read-only mmap and avoids undefined tensor writes.
        return torch.from_numpy(np.array(self.array[index], dtype=np.int64, copy=True))


class OrderedBlocks(Dataset[torch.Tensor]):
    def __init__(self, source: PackedMemmapDataset, order: Sequence[int]) -> None:
        self.source = source
        self.order = np.asarray(order, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.source[int(self.order[index])]


def load_packed_dataset(root: Path, split: str) -> tuple[PackedMemmapDataset, dict]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    details = metadata["splits"][split]
    dataset = PackedMemmapDataset(
        root / details["filename"],
        block_size=int(metadata["block_size"]),
        block_count=int(details["blocks"]),
    )
    return dataset, metadata
