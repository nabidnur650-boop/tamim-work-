from __future__ import annotations

import math
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

import numpy as np

from .tokenization import DIALECTS, nfc


def _tokens(text: str) -> list[str]:
    return str(text).split()


def _token_jaccard(left: str, right: str) -> float:
    left_set = set(_tokens(left))
    right_set = set(_tokens(right))
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def _common_prefix_fraction(left: str, right: str) -> float:
    common = 0
    for left_character, right_character in zip(left, right):
        if left_character != right_character:
            break
        common += 1
    return common / max(1, min(len(left), len(right)))


def _lexical_diversity(text: str) -> float:
    tokens = _tokens(text)
    return len(set(tokens)) / max(1, len(tokens))


PAIR_NAMES = (
    "source_neural",
    "source_rewrite",
    "neural_rewrite",
)

FUSION_FEATURE_NAMES = tuple(
    [
        "source_characters",
        "neural_characters",
        "rewrite_characters",
        "source_words",
        "neural_words",
        "rewrite_words",
    ]
    + [
        f"{pair}_{suffix}"
        for pair in PAIR_NAMES
        for suffix in (
            "sequence_similarity",
            "token_jaccard",
            "common_prefix_fraction",
            "right_to_left_character_ratio",
            "right_to_left_word_ratio",
        )
    ]
    + [
        f"{candidate}_{suffix}"
        for candidate in ("source", "neural", "rewrite")
        for suffix in ("digits", "punctuation", "lexical_diversity")
    ]
    + [
        "neural_equals_source",
        "rewrite_equals_source",
        "neural_equals_rewrite",
        "inferred_dialect_max_probability",
        "inferred_dialect_margin",
        "inferred_dialect_entropy",
    ]
    + [f"inferred_dialect_probability_{index:02d}" for index in range(13)]
)


def normalization_fusion_features(
    source: str,
    neural: str,
    rewrite: str,
    inferred_dialect_probabilities: Sequence[float],
) -> np.ndarray:
    """Return reference-free features for choosing a normalization candidate.

    The function has no source-id or gold-dialect argument by design.  Dialect
    probabilities must be inferred from the input text by a frozen classifier.
    """

    texts = (str(source), str(neural), str(rewrite))
    character_lengths = [len(text) for text in texts]
    word_lengths = [len(_tokens(text)) for text in texts]
    values: list[float] = [
        *[float(value) for value in character_lengths],
        *[float(value) for value in word_lengths],
    ]
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        left = texts[left_index]
        right = texts[right_index]
        values.extend(
            [
                SequenceMatcher(a=left, b=right, autojunk=False).ratio(),
                _token_jaccard(left, right),
                _common_prefix_fraction(left, right),
                len(right) / max(1, len(left)),
                len(_tokens(right)) / max(1, len(_tokens(left))),
            ]
        )
    for text in texts:
        values.extend(
            [
                float(sum(character.isdigit() for character in text)),
                float(
                    sum(
                        unicodedata.category(character).startswith("P")
                        for character in text
                    )
                ),
                _lexical_diversity(text),
            ]
        )
    values.extend(
        [
            float(texts[1] == texts[0]),
            float(texts[2] == texts[0]),
            float(texts[1] == texts[2]),
        ]
    )
    probabilities = np.asarray(inferred_dialect_probabilities, dtype=np.float64)
    if probabilities.shape != (13,):
        raise ValueError(f"Expected 13 inferred dialect probabilities, got {probabilities.shape}")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Inferred dialect probabilities must be finite and non-negative")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("Inferred dialect probabilities must have positive mass")
    probabilities = probabilities / total
    ordered = np.sort(probabilities)
    entropy = float(
        -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
        / math.log(len(probabilities))
    )
    values.extend(
        [
            float(ordered[-1]),
            float(ordered[-1] - ordered[-2]),
            entropy,
            *probabilities.tolist(),
        ]
    )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FUSION_FEATURE_NAMES),):
        raise AssertionError(
            f"Feature contract mismatch: {result.shape} != {(len(FUSION_FEATURE_NAMES),)}"
        )
    return result


def blend_identification_probabilities(
    neural_probabilities: np.ndarray,
    svm_probabilities: np.ndarray,
    *,
    neural_weight: float,
) -> np.ndarray:
    if not 0.0 <= neural_weight <= 1.0:
        raise ValueError("neural_weight must be in [0, 1]")
    neural = np.asarray(neural_probabilities, dtype=np.float64)
    svm = np.asarray(svm_probabilities, dtype=np.float64)
    if neural.shape != svm.shape:
        raise ValueError(f"Probability shape mismatch: {neural.shape} != {svm.shape}")
    if neural.ndim != 2 or neural.shape[1] != 13:
        raise ValueError(f"Expected [examples, 13] probabilities, got {neural.shape}")
    result = neural_weight * neural + (1.0 - neural_weight) * svm
    result /= result.sum(axis=1, keepdims=True)
    return result


def temperature_scale_probabilities(
    probabilities: np.ndarray, *, temperature: float
) -> np.ndarray:
    """Apply scalar temperature calibration to a probability matrix."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(DIALECTS):
        raise ValueError(
            f"Expected [examples, {len(DIALECTS)}] probabilities, got {values.shape}"
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.clip(values, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return calibrated


def classical_identification_probabilities(
    texts: Sequence[str], artifact: Mapping[str, Any]
) -> np.ndarray:
    """Run the frozen character-SVM probability model on source text only."""

    required = ("vectorizer", "model", "temperature")
    missing = [name for name in required if name not in artifact]
    if missing:
        raise ValueError(f"Classical identification artifact is missing {missing}")
    matrix = artifact["vectorizer"].transform([nfc(str(text)) for text in texts])
    model = artifact["model"]
    logits = np.asarray(model.decision_function(matrix), dtype=np.float64)
    if logits.ndim == 1:
        logits = np.column_stack((-logits, logits))
    if logits.shape[1] != len(DIALECTS):
        expanded = np.full((len(logits), len(DIALECTS)), -1e9, dtype=np.float64)
        classes = np.asarray(model.classes_, dtype=np.int64)
        expanded[:, classes] = logits
        logits = expanded
    temperature = float(artifact["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Classical artifact temperature must be finite and positive")
    logits = logits / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def inferred_dialect_rewrite(
    texts: Sequence[str],
    probabilities: np.ndarray,
    mapping: Mapping[tuple[str, str], str],
    *,
    allowed_dialects: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Apply a word rewrite using dialect labels inferred from input text."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(texts), len(DIALECTS)):
        raise ValueError(
            "Dialect probability shape mismatch: "
            f"{values.shape} != {(len(texts), len(DIALECTS))}"
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Dialect probabilities must be finite and non-negative")
    candidates = tuple(allowed_dialects) if allowed_dialects is not None else DIALECTS
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("allowed_dialects must contain unique known dialect labels")
    try:
        candidate_indices = np.asarray(
            [DIALECTS.index(str(dialect)) for dialect in candidates], dtype=np.int64
        )
    except ValueError as error:
        raise ValueError("allowed_dialects contains an unknown dialect label") from error
    labels = [
        str(candidates[int(index)])
        for index in values[:, candidate_indices].argmax(axis=1)
    ]
    rewritten = [
        " ".join(
            mapping.get((label, token), token)
            for token in nfc(str(text)).split()
        )
        for text, label in zip(texts, labels, strict=True)
    ]
    return rewritten, labels


def select_normalization_candidates(
    sources: Sequence[str],
    neural_candidates: Sequence[str],
    rewrite_candidates: Sequence[str],
    inferred_dialect_probabilities: np.ndarray,
    artifact: Mapping[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Choose between neural and rewrite candidates with a frozen selector."""

    count = len(sources)
    if len(neural_candidates) != count or len(rewrite_candidates) != count:
        raise ValueError("Normalization candidate lengths do not match source length")
    probabilities = np.asarray(inferred_dialect_probabilities, dtype=np.float64)
    if probabilities.shape != (count, len(DIALECTS)):
        raise ValueError(
            "Dialect probability shape mismatch: "
            f"{probabilities.shape} != {(count, len(DIALECTS))}"
        )
    if artifact.get("source_blind") is not True:
        raise ValueError("Normalization selector is not marked source-blind")
    if tuple(artifact.get("feature_names", ())) != FUSION_FEATURE_NAMES:
        raise ValueError("Normalization selector feature contract mismatch")
    if "model" not in artifact or "threshold" not in artifact:
        raise ValueError("Normalization selector artifact is incomplete")
    features = np.stack(
        [
            normalization_fusion_features(source, neural, rewrite, probability)
            for source, neural, rewrite, probability in zip(
                sources,
                neural_candidates,
                rewrite_candidates,
                probabilities,
                strict=True,
            )
        ]
    )
    margins = np.asarray(artifact["model"].predict(features), dtype=np.float64)
    if margins.shape != (count,) or not np.isfinite(margins).all():
        raise ValueError("Normalization selector returned invalid margins")
    selected_neural = margins > float(artifact["threshold"])
    predictions = [
        str(neural) if choose_neural else str(rewrite)
        for neural, rewrite, choose_neural in zip(
            neural_candidates, rewrite_candidates, selected_neural, strict=True
        )
    ]
    return predictions, margins, selected_neural
