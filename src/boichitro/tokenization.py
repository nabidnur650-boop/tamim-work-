from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


DIALECTS: tuple[str, ...] = (
    "BAR",
    "CHI",
    "KHU",
    "KIS",
    "MYM",
    "NAR",
    "NOA",
    "NSD",
    "RAJ",
    "RAN",
    "STD",
    "SYL",
    "TAN",
)

BASE_SPECIAL_TOKENS: tuple[str, ...] = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<mask>",
    "<cls>",
    "<task_norm>",
    "<task_id>",
    "<task_clm>",
    "<dial_unknown>",
)

DIALECT_TOKENS: tuple[str, ...] = tuple(f"<dial:{dialect}>" for dialect in DIALECTS)
SPECIAL_TOKENS: tuple[str, ...] = BASE_SPECIAL_TOKENS + DIALECT_TOKENS


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    vocab_size: int
    corpus: str

    @property
    def candidate_id(self) -> str:
        return f"{self.family}_{self.vocab_size // 1000}k"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def nfc(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def canonical_detokenized(text: object) -> str:
    """Canonical comparison only; the original text is never rewritten.

    WordPiece's decoder inserts spaces around punctuation separated by its
    pre-tokenizer. This equivalence collapses those reversible spacing changes
    while retaining every letter, digit, mark, and punctuation symbol.
    """

    value = nfc(text)
    value = re.sub(r"\s*([,.;:!?।\-–—/\\])\s*", r"\1", value)
    value = re.sub(r"([\(\[\{])\s+", r"\1", value)
    value = re.sub(r"\s+([\)\]\}])", r"\1", value)
    return " ".join(value.split())


def corpus_texts(frame: pd.DataFrame) -> list[str]:
    texts = [nfc(value) for value in frame["text_model"].tolist()]
    return [text for text in texts if text]


def temperature_balanced_frame(
    frame: pd.DataFrame,
    *,
    seed: int,
    alpha: float = 0.5,
    target_rows: int | None = None,
) -> pd.DataFrame:
    """Sample a fixed corpus with p(dialect) proportional to count**alpha.

    Sampling is deterministic and may use replacement only for strata whose
    requested allocation exceeds their available unique rows. The original
    row identifier is retained so every repeat remains auditable.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if target_rows is None:
        target_rows = len(frame)
    counts = frame.groupby("dialect", observed=True).size().sort_index()
    if counts.empty:
        raise ValueError("No dialect rows available")
    weights = counts.astype(float).pow(alpha)
    exact = weights / weights.sum() * int(target_rows)
    allocation = np.floor(exact).astype(int)
    remainder = int(target_rows - allocation.sum())
    for dialect in (exact - allocation).sort_values(ascending=False).index[:remainder]:
        allocation.loc[dialect] += 1

    parts: list[pd.DataFrame] = []
    for offset, dialect in enumerate(counts.index):
        group = frame.loc[frame["dialect"] == dialect]
        requested = int(allocation.loc[dialect])
        parts.append(
            group.sample(
                n=requested,
                replace=requested > len(group),
                random_state=seed + 104729 * (offset + 1),
                weights="sampling_weight" if "sampling_weight" in group.columns else None,
            )
        )
    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _alphabet(texts: Sequence[str]) -> list[str]:
    return sorted({character for text in texts for character in text})


def build_tokenizer(spec: CandidateSpec, texts: Sequence[str], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    alphabet = _alphabet(texts)

    if spec.family in {"wordpiece_natural", "wordpiece_balanced"}:
        tokenizer = Tokenizer(
            models.WordPiece(
                unk_token="<unk>",
                continuing_subword_prefix="##",
                max_input_chars_per_word=1000,
            )
        )
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=False)
        trainer = trainers.WordPieceTrainer(
            vocab_size=spec.vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=alphabet,
            continuing_subword_prefix="##",
        )
    elif spec.family == "unigram_balanced":
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
        trainer = trainers.UnigramTrainer(
            vocab_size=spec.vocab_size,
            unk_token="<unk>",
            show_progress=True,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=alphabet,
            shrinking_factor=0.75,
            max_piece_length=24,
        )
    elif spec.family == "byte_bpe_balanced":
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>", fuse_unk=False))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=spec.vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
    else:
        raise ValueError(f"Unsupported tokenizer family: {spec.family}")

    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    # Mark the already-trained control tokens as special without changing IDs.
    tokenizer.add_special_tokens(
        [AddedToken(token, special=True, normalized=False) for token in SPECIAL_TOKENS]
    )
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path), pretty=True)

    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    expected_ids = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    if special_ids != expected_ids:
        raise AssertionError(f"Special-token IDs changed: {special_ids} != {expected_ids}")
    actual_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab > spec.vocab_size:
        raise AssertionError(f"Tokenizer grew beyond requested vocabulary: {actual_vocab}")

    metadata = {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "corpus": spec.corpus,
        "requested_vocab_size": spec.vocab_size,
        "actual_vocab_size": actual_vocab,
        "special_token_ids": special_ids,
        "normalization": "Unicode NFC only",
        "training_rows": len(texts),
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def load_tokenizer(path: Path) -> Tokenizer:
    tokenizer_path = path / "tokenizer.json" if path.is_dir() else path
    return Tokenizer.from_file(str(tokenizer_path))


def gini(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.allclose(array, 0.0):
        return 0.0
    array = np.sort(np.maximum(array, 0.0))
    ranks = np.arange(1, array.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * array) / np.sum(array) - (array.size + 1)) / array.size)


def evaluate_tokenizer(
    tokenizer: Tokenizer,
    frame: pd.DataFrame,
    *,
    text_column: str = "text_model",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    unk_id = tokenizer.token_to_id("<unk>")
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        text = nfc(getattr(row, text_column))
        encoding = tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids
        tokens = len(ids)
        characters = max(1, len(text))
        bytes_count = max(1, len(text.encode("utf-8")))
        units = max(1, len(text.split()))
        decoded = nfc(tokenizer.decode(ids, skip_special_tokens=False))
        rows.append(
            {
                "row_id": getattr(row, "row_id", ""),
                "dialect": getattr(row, "dialect", "UNK"),
                "source_id": getattr(row, "source_id", "UNK"),
                "text": text,
                "characters": characters,
                "bytes": bytes_count,
                "whitespace_units": units,
                "tokens": tokens,
                "unk_tokens": int(sum(token_id == unk_id for token_id in ids)) if unk_id is not None else 0,
                "tokens_per_character": tokens / characters,
                "tokens_per_byte": tokens / bytes_count,
                "fertility": tokens / units,
                "roundtrip_exact": decoded == text,
                "canonical_roundtrip_exact": canonical_detokenized(decoded)
                == canonical_detokenized(text),
            }
        )
    per_row = pd.DataFrame(rows)

    def aggregate(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "rows": len(group),
                "tokens": int(group["tokens"].sum()),
                "characters": int(group["characters"].sum()),
                "bytes": int(group["bytes"].sum()),
                "unk_tokens": int(group["unk_tokens"].sum()),
                "tokens_per_character": group["tokens"].sum() / max(1, group["characters"].sum()),
                "tokens_per_byte": group["tokens"].sum() / max(1, group["bytes"].sum()),
                "mean_fertility": group["fertility"].mean(),
                "p95_tokens": group["tokens"].quantile(0.95),
                "roundtrip_exact_rate": group["roundtrip_exact"].mean(),
                "canonical_roundtrip_exact_rate": group[
                    "canonical_roundtrip_exact"
                ].mean(),
            }
        )

    by_dialect = per_row.groupby("dialect", observed=True).apply(aggregate, include_groups=False).reset_index()
    overall = aggregate(per_row).to_dict()
    dialect_costs = by_dialect["tokens_per_character"].to_numpy(dtype=float)
    total_tokens = int(per_row["tokens"].sum())
    overall.update(
        {
            "unk_rate": float(per_row["unk_tokens"].sum() / max(1, total_tokens)),
            "worst_dialect_tokens_per_character": float(dialect_costs.max()),
            "best_dialect_tokens_per_character": float(dialect_costs.min()),
            "dialect_cost_ratio": float(dialect_costs.max() / max(1e-12, dialect_costs.min())),
            "dialect_cost_gini": gini(dialect_costs),
            "dialect_cost_std": float(dialect_costs.std(ddof=0)),
        }
    )
    return per_row, by_dialect, {key: float(value) for key, value in overall.items()}


def intrinsic_selection_score(metrics: dict[str, float], vocab_size: int) -> float:
    """Lower is better; efficiency and worst-dialect cost dominate."""

    return float(
        metrics["tokens_per_character"]
        + 0.50 * metrics["worst_dialect_tokens_per_character"]
        + 0.25 * metrics["dialect_cost_std"]
        + 0.25 * metrics["dialect_cost_gini"]
        + 25.0 * metrics["unk_rate"]
        + 0.50 * (1.0 - metrics["canonical_roundtrip_exact_rate"])
        + 0.015 * (vocab_size / 32_000)
    )


def assert_special_tokens(tokenizer: Tokenizer) -> None:
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise AssertionError(f"{token}: expected ID {expected_id}, found {actual_id}")
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded.ids != [expected_id]:
            raise AssertionError(f"{token} is not atomic: {encoded.ids}")
