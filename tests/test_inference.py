from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from boichitro.inference import greedy_normalize  # noqa: E402
from boichitro.modeling import BoichitroConfig, BoichitroForMultiTask  # noqa: E402
from boichitro.tokenization import CandidateSpec, build_tokenizer  # noqa: E402


class InferenceTests(unittest.TestCase):
    def test_cached_and_uncached_greedy_decoding_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texts = ["মুই ভাত খামু", "আমি ভাত খাব", "আঁই ঘরত যাই"]
            build_tokenizer(
                CandidateSpec("byte_bpe_balanced", 512, "balanced"), texts, root
            )
            tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
            torch.manual_seed(41)
            model = BoichitroForMultiTask(
                BoichitroConfig(
                    vocab_size=tokenizer.get_vocab_size(),
                    max_seq_len=64,
                    n_layers=2,
                    d_model=64,
                    n_heads=4,
                    n_kv_heads=2,
                    dense_ffn_dim=128,
                    dense_prefix_layers=2,
                    use_mtp=False,
                )
            ).eval()
            values = ["মুই ভাত খামু", "আঁই যাই"]
            cached = greedy_normalize(
                model,
                tokenizer,
                values,
                device=torch.device("cpu"),
                batch_size=2,
                max_new_tokens=6,
                use_kv_cache=True,
            )
            uncached = greedy_normalize(
                model,
                tokenizer,
                values,
                device=torch.device("cpu"),
                batch_size=2,
                max_new_tokens=6,
                use_kv_cache=False,
            )
            self.assertEqual(cached, uncached)


if __name__ == "__main__":
    unittest.main()
