from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.boichitro.pretraining import PackedMemmapDataset


class PretrainingDataTests(unittest.TestCase):
    def test_memmap_shape_and_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x.bin"
            np.arange(24, dtype=np.uint16).tofile(path)
            dataset = PackedMemmapDataset(path, block_size=6, block_count=4)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset[2].tolist(), list(range(12, 18)))


if __name__ == "__main__":
    unittest.main()
