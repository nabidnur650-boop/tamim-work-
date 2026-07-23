from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from jiwer import cer, wer
from sacrebleu.metrics import BLEU, CHRF, TER
from scipy.stats import entropy
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from .tokenization import DIALECTS, nfc


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(result)


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    targets = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    probabilities: np.ndarray,
    *,
    label_names: Sequence[str] = DIALECTS,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    all_ids = np.arange(len(label_names))
    regional_ids = np.asarray(
        [index for index, name in enumerate(label_names) if name != "STD"], dtype=np.int64
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        labels=all_ids,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "label_id": all_ids,
            "dialect": list(label_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    present_regional = regional_ids[np.isin(regional_ids, np.unique(labels_array))]
    regional_f1 = f1_score(
        labels_array,
        predictions_array,
        labels=present_regional,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(labels_array, predictions_array, labels=all_ids, normalize="true")
    metrics = {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_array, predictions_array)),
        "macro_f1_13": float(
            f1_score(labels_array, predictions_array, labels=all_ids, average="macro", zero_division=0)
        ),
        "regional_macro_f1": float(regional_f1),
        "weighted_f1": float(
            f1_score(labels_array, predictions_array, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(labels_array, predictions_array)),
        "ece_15": expected_calibration_error(probabilities, labels_array, bins=15),
        "brier": multiclass_brier(probabilities, labels_array),
        "worst_present_dialect_f1": float(per_class.loc[per_class["support"] > 0, "f1"].min()),
        "examples": float(len(labels_array)),
    }
    return metrics, per_class, matrix


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"[০-৯0-9]+", text))


def normalization_metrics(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    required = {"prediction", "reference", "dialect"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing normalization columns: {sorted(required - set(frame.columns))}")
    chrf = CHRF(char_order=6, word_order=2, beta=2)
    bleu = BLEU(tokenize="none", effective_order=True)
    ter_metric = TER(normalized=True, no_punct=False, asian_support=True)

    rows: list[dict[str, object]] = []
    for dialect, group in frame.groupby("dialect", sort=True, observed=True):
        hypotheses = group["prediction"].map(nfc).tolist()
        references = group["reference"].map(nfc).tolist()
        number_required = np.asarray([bool(_numbers(text)) for text in references])
        number_preserved = np.asarray(
            [_numbers(reference).issubset(_numbers(hypothesis)) for reference, hypothesis in zip(references, hypotheses)]
        )
        rows.append(
            {
                "dialect": dialect,
                "rows": len(group),
                "chrfpp": chrf.corpus_score(hypotheses, [references]).score,
                "sacrebleu": bleu.corpus_score(hypotheses, [references]).score,
                "ter": ter_metric.corpus_score(hypotheses, [references]).score,
                "cer": cer(references, hypotheses),
                "wer": wer(references, hypotheses),
                "exact_match": float(np.mean(np.asarray(hypotheses) == np.asarray(references))),
                "number_preservation": float(number_preserved[number_required].mean())
                if number_required.any()
                else float("nan"),
            }
        )
    by_dialect = pd.DataFrame(rows)
    hypotheses = frame["prediction"].map(nfc).tolist()
    references = frame["reference"].map(nfc).tolist()
    overall = {
        "rows": len(frame),
        "corpus_chrfpp": chrf.corpus_score(hypotheses, [references]).score,
        "macro_chrfpp": float(by_dialect["chrfpp"].mean()),
        "worst_dialect_chrfpp": float(by_dialect["chrfpp"].min()),
        "corpus_sacrebleu": bleu.corpus_score(hypotheses, [references]).score,
        "macro_sacrebleu": float(by_dialect["sacrebleu"].mean()),
        "corpus_ter": ter_metric.corpus_score(hypotheses, [references]).score,
        "cer": cer(references, hypotheses),
        "wer": wer(references, hypotheses),
        "exact_match": float(np.mean(np.asarray(hypotheses) == np.asarray(references))),
        "chrf_signature": str(chrf.get_signature()),
        "bleu_signature": str(bleu.get_signature()),
        "ter_signature": str(ter_metric.get_signature()),
    }
    return overall, by_dialect


def expert_load_metrics(counts: np.ndarray) -> dict[str, float]:
    values = np.asarray(counts, dtype=np.float64)
    total = values.sum()
    probabilities = values / total if total else np.full_like(values, 1.0 / len(values))
    sorted_values = np.sort(values)
    ranks = np.arange(1, len(values) + 1)
    gini = (
        (2 * np.sum(ranks * sorted_values) / sorted_values.sum() - (len(values) + 1))
        / len(values)
        if sorted_values.sum()
        else 0.0
    )
    return {
        "load_cv": float(values.std() / max(1e-12, values.mean())),
        "load_gini": float(gini),
        "load_entropy": float(entropy(probabilities)),
        "load_entropy_normalized": float(entropy(probabilities) / math.log(len(values))),
        "minimum_load": float(values.min()),
        "maximum_load": float(values.max()),
        "dropped_rate": 0.0,
    }
