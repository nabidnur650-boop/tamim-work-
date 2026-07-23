#!/usr/bin/env python3
"""Audit normalization baselines under the source-blind task contract.

This development-only tool never opens a test split.  It distinguishes a fair
pooled rewrite system and an inferred-dialect rewrite system from the legacy
gold-dialect rewrite oracle.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import softmax


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.metrics import normalization_metrics  # noqa: E402
from boichitro.tokenization import DIALECTS, nfc  # noqa: E402


def replacement_counts(frame: pd.DataFrame) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in frame.itertuples(index=False):
        source = nfc(row.source_text_model).split()
        target = nfc(row.target_text_model).split()
        matcher = SequenceMatcher(a=source, b=target, autojunk=False)
        for operation, i1, i2, j1, j2 in matcher.get_opcodes():
            if operation == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
                counts[source[i1]][target[j1]] += 1
    return counts


def resolve_mapping(
    counts: dict[str, Counter[str]], *, min_count: int, minimum_confidence: float
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for source, alternatives in counts.items():
        target, count = alternatives.most_common(1)[0]
        if count >= min_count and count / sum(alternatives.values()) >= minimum_confidence:
            mapping[source] = target
    return mapping


def pooled_rewrite(texts: Iterable[str], mapping: dict[str, str]) -> list[str]:
    return [
        " ".join(mapping.get(token, token) for token in nfc(text).split())
        for text in texts
    ]


def labelled_rewrite(
    texts: Iterable[str], labels: Iterable[str], mapping: dict[tuple[str, str], str]
) -> list[str]:
    return [
        " ".join(mapping.get((str(label), token), token) for token in nfc(text).split())
        for text, label in zip(texts, labels, strict=True)
    ]


def prediction_frame(frame: pd.DataFrame, predictions: list[str]) -> pd.DataFrame:
    output = frame[["row_id", "dialect", "source_id", "semantic_group_id"]].copy()
    output["source"] = frame["source_text_model"].map(nfc)
    output["reference"] = frame["target_text_model"].map(nfc)
    output["prediction"] = predictions
    return output


def evaluate(frame: pd.DataFrame, predictions: list[str]) -> tuple[dict, pd.DataFrame]:
    return normalization_metrics(prediction_frame(frame, predictions))


def predict_dialects(texts: pd.Series) -> tuple[list[str], np.ndarray]:
    artifact_path = PROJECT / "artifacts/baselines/id_char_tfidf_svm.pkl"
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    matrix = artifact["vectorizer"].transform(texts.map(nfc))
    logits = np.asarray(artifact["model"].decision_function(matrix), dtype=np.float64)
    if logits.ndim == 1:
        logits = np.column_stack((-logits, logits))
    if logits.shape[1] != len(DIALECTS):
        expanded = np.full((len(logits), len(DIALECTS)), -1e9, dtype=np.float64)
        expanded[:, np.asarray(artifact["model"].classes_, dtype=int)] = logits
        logits = expanded
    probabilities = softmax(logits / float(artifact["temperature"]), axis=1)
    labels = [DIALECTS[index] for index in probabilities.argmax(axis=1)]
    return labels, probabilities


def main() -> None:
    data = PROJECT / "data/final/v1"
    train = pd.read_parquet(data / "normalization_train.parquet")
    train = train.loc[~train["is_synthetic"].astype(bool)].copy()
    validation = pd.read_parquet(data / "normalization_validation.parquet")
    identification_train = pd.read_parquet(data / "identification_train.parquet")

    overlap = set(validation["source_text_compact"].astype(str)) & set(
        identification_train["text_compact"].astype(str)
    )
    if overlap:
        raise RuntimeError(
            "The inferred-dialect control is invalid: normalization validation "
            f"overlaps identification training on {len(overlap)} compact texts"
        )

    print("Building dialect-agnostic rewrite counts", flush=True)
    counts = replacement_counts(train)
    grid: list[dict] = []
    candidates: list[tuple[float, float, int, float, dict[str, str], dict]] = []
    for min_count in (2, 3, 5, 8, 10):
        for confidence in (0.5, 0.6, 0.7, 0.8, 0.9):
            mapping = resolve_mapping(
                counts,
                min_count=min_count,
                minimum_confidence=confidence,
            )
            metrics, _ = evaluate(
                validation,
                pooled_rewrite(validation["source_text_model"], mapping),
            )
            row = {
                "min_count": min_count,
                "minimum_confidence": confidence,
                "mapping_entries": len(mapping),
                "macro_chrfpp": float(metrics["macro_chrfpp"]),
                "worst_dialect_chrfpp": float(metrics["worst_dialect_chrfpp"]),
            }
            grid.append(row)
            candidates.append(
                (
                    row["macro_chrfpp"],
                    row["worst_dialect_chrfpp"],
                    -min_count,
                    confidence,
                    mapping,
                    row,
                )
            )
    selected = max(candidates, key=lambda item: item[:4])
    pooled_mapping = selected[4]
    pooled_selection = selected[5]
    print(f"Selected pooled rewrite: {pooled_selection}", flush=True)

    oracle_path = PROJECT / "artifacts/baselines/normalization_word_rewrite.pkl"
    with oracle_path.open("rb") as handle:
        oracle_mapping: dict[tuple[str, str], str] = pickle.load(handle)
    predicted_dialects, dialect_probabilities = predict_dialects(
        validation["source_text_model"]
    )
    supported_dialects = sorted({str(dialect) for dialect, _ in oracle_mapping})
    supported_indices = np.asarray(
        [DIALECTS.index(dialect) for dialect in supported_dialects], dtype=np.int64
    )
    supported_predicted_dialects = [
        supported_dialects[int(index)]
        for index in dialect_probabilities[:, supported_indices].argmax(axis=1)
    ]

    by_word: dict[str, list[str]] = defaultdict(list)
    for (_, word), target in oracle_mapping.items():
        by_word[word].append(target)
    consensus_mapping = {
        word: targets[0]
        for word, targets in by_word.items()
        if len(set(targets)) == 1
    }

    systems = {
        "N_COPY_SOURCE_BLIND": validation["source_text_model"].map(nfc).tolist(),
        "N_WORD_REWRITE_POOLED_SOURCE_BLIND": pooled_rewrite(
            validation["source_text_model"], pooled_mapping
        ),
        "N_WORD_REWRITE_CONSENSUS_SOURCE_BLIND": pooled_rewrite(
            validation["source_text_model"], consensus_mapping
        ),
        "N_WORD_REWRITE_INFERRED_DIALECT": labelled_rewrite(
            validation["source_text_model"], predicted_dialects, oracle_mapping
        ),
        "N_WORD_REWRITE_INFERRED_SUPPORTED_DIALECT": labelled_rewrite(
            validation["source_text_model"],
            supported_predicted_dialects,
            oracle_mapping,
        ),
        "N_WORD_REWRITE_ORACLE_GOLD_DIALECT": labelled_rewrite(
            validation["source_text_model"],
            validation["dialect"].astype(str),
            oracle_mapping,
        ),
    }

    prediction_root = PROJECT / "predictions/development_source_blind"
    metric_root = PROJECT / "metrics/development_source_blind"
    report_root = PROJECT / "reports/model"
    prediction_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for model_id, predictions in systems.items():
        output = prediction_frame(validation, predictions)
        metrics, by_dialect = normalization_metrics(output)
        output.to_parquet(prediction_root / f"{model_id}_validation.parquet", index=False)
        by_dialect.to_csv(metric_root / f"{model_id}_validation_by_dialect.csv", index=False)
        payload = {
            "model_id": model_id,
            "split": "validation",
            "evidence": "development_only",
            "test_data_access": False,
            **metrics,
        }
        (metric_root / f"{model_id}_validation.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(payload)
        print(
            f"{model_id}: macro_chrfpp={metrics['macro_chrfpp']:.4f}; "
            f"worst={metrics['worst_dialect_chrfpp']:.4f}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    result.to_csv(report_root / "source_blind_normalization_baselines.csv", index=False)
    audit = {
        "status": "COMPLETE_VALIDATION_ONLY",
        "protocol_id": "boichitro_source_blind_baseline_audit_v1",
        "test_data_access": False,
        "task_contract": "normalization input contains no gold dialect or source metadata",
        "legacy_finding": (
            "The legacy N_WORD_REWRITE baseline used the gold dialect label and must be "
            "reported only as an oracle upper-bound, not a source-blind comparator."
        ),
        "normalization_validation_rows": len(validation),
        "id_train_compact_overlap_with_normalization_validation": len(overlap),
        "inferred_dialect_accuracy_on_normalization_validation": float(
            np.mean(np.asarray(predicted_dialects) == validation["dialect"].astype(str).to_numpy())
        ),
        "inferred_dialect_mean_confidence": float(dialect_probabilities.max(axis=1).mean()),
        "normalization_supported_dialects": supported_dialects,
        "supported_dialect_inference_rule": (
            "argmax frozen ID-SVM probability over dialects represented in the "
            "training-only normalization lexicon"
        ),
        "pooled_selection": pooled_selection,
        "pooled_grid": sorted(
            grid,
            key=lambda row: (row["macro_chrfpp"], row["worst_dialect_chrfpp"]),
            reverse=True,
        ),
        "consensus_mapping_entries": len(consensus_mapping),
        "systems": rows,
    }
    (report_root / "source_blind_normalization_baseline_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
