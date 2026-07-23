from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from boichitro.tokenization import (  # noqa: E402
    CandidateSpec,
    SPECIAL_TOKENS,
    assert_special_tokens,
    build_tokenizer,
    evaluate_tokenizer,
    temperature_balanced_frame,
)


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "row_id": ["a", "b", "c", "d"],
                "dialect": ["BAR", "BAR", "CHI", "CHI"],
                "source_id": ["x", "x", "y", "y"],
                "text_model": ["মুই ভাত খামু", "তুমি কই যাও", "আঁই ঘরত যাই", "আমি এখন যাব"],
                "sampling_weight": [1.0, 1.0, 1.0, 1.0],
            }
        )

    def test_temperature_sampling_is_deterministic(self) -> None:
        first = temperature_balanced_frame(self.frame, seed=1701, target_rows=20)
        second = temperature_balanced_frame(self.frame, seed=1701, target_rows=20)
        self.assertEqual(first["row_id"].tolist(), second["row_id"].tolist())
        self.assertEqual(len(first), 20)

    def test_byte_bpe_has_stable_special_ids_and_no_unk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            from tokenizers import Tokenizer

            build_tokenizer(
                CandidateSpec("byte_bpe_balanced", 512, "balanced"),
                self.frame["text_model"].tolist(),
                path,
            )
            tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
            assert_special_tokens(tokenizer)
            _, _, metrics = evaluate_tokenizer(tokenizer, self.frame)
            self.assertEqual(metrics["unk_rate"], 0.0)
            self.assertEqual(tokenizer.token_to_id("<pad>"), 0)
            self.assertEqual(tokenizer.token_to_id(SPECIAL_TOKENS[-1]), len(SPECIAL_TOKENS) - 1)


if __name__ == "__main__":
    unittest.main()
