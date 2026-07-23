#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.data import DIALECT_TO_ID  # noqa: E402
from boichitro.metrics import classification_metrics, normalization_metrics  # noqa: E402
from boichitro.tokenization import DIALECTS, nfc  # noqa: E402


def temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_t: float) -> float:
        scaled = logits / math.exp(log_t)
        log_probabilities = scaled - np.logaddexp.reduce(scaled, axis=1, keepdims=True)
        return float(-log_probabilities[np.arange(len(labels)), labels].mean())

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    return float(math.exp(result.x))


def decision_logits(model, matrix) -> np.ndarray:
    values = np.asarray(model.decision_function(matrix), dtype=np.float64)
    if values.ndim == 1:
        values = np.column_stack((-values, values))
    # Expand models trained without a label only if necessary.
    if values.shape[1] != len(DIALECTS):
        expanded = np.full((len(values), len(DIALECTS)), -1e9, dtype=np.float64)
        expanded[:, np.asarray(model.classes_, dtype=int)] = values
        values = expanded
    return values


def evaluate_id_model(
    model_id: str,
    model,
    vectorizer,
    frame: pd.DataFrame,
    split_name: str,
    calibration_temperature: float,
) -> dict[str, float | str]:
    matrix = vectorizer.transform(frame["text_model"].map(nfc))
    logits = decision_logits(model, matrix)
    probabilities = softmax(logits / calibration_temperature, axis=1)
    labels = frame["dialect"].map(DIALECT_TO_ID).to_numpy()
    predictions = probabilities.argmax(axis=1)
    metrics, per_class, confusion = classification_metrics(labels, predictions, probabilities)
    out = PROJECT / "predictions/identification"
    out.mkdir(parents=True, exist_ok=True)
    prediction_frame = frame[["row_id", "dialect", "source_id", "split", "evaluation_track"]].copy()
    prediction_frame["label_id"] = labels
    prediction_frame["prediction_id"] = predictions
    prediction_frame["prediction"] = [DIALECTS[index] for index in predictions]
    prediction_frame["probabilities"] = probabilities.tolist()
    prediction_frame.to_parquet(out / f"{model_id}_{split_name}.parquet", index=False)
    metric_dir = PROJECT / "metrics/identification"
    metric_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(metric_dir / f"{model_id}_{split_name}_per_class.csv", index=False)
    np.save(metric_dir / f"{model_id}_{split_name}_confusion.npy", confusion)
    payload = {"model_id": model_id, "split": split_name, **metrics}
    (metric_dir / f"{model_id}_{split_name}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_identification() -> pd.DataFrame:
    data_dir = PROJECT / "data/final/v1"
    train = pd.read_parquet(data_dir / "identification_train.parquet")
    evaluation = pd.read_parquet(data_dir / "identification_evaluation.parquet")
    validation = evaluation.loc[evaluation["split"] == "validation"].copy()
    protocols = {
        "iid_test": evaluation.loc[evaluation["split"] == "test"].copy(),
        "source_ood": evaluation.loc[evaluation["split"] == "test_ood"].copy(),
        "external_transcript": evaluation.loc[evaluation["split"] == "test_external"].copy(),
    }
    y_train = train["dialect"].map(DIALECT_TO_ID).to_numpy()
    y_validation = validation["dialect"].map(DIALECT_TO_ID).to_numpy()
    reports: list[dict[str, float | str]] = []
    artifact_dir = PROJECT / "artifacts/baselines"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=250_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    started = time.perf_counter()
    x_train = char_vectorizer.fit_transform(train["text_model"].map(nfc))
    x_validation = char_vectorizer.transform(validation["text_model"].map(nfc))
    candidates = []
    for c_value in (0.25, 0.5, 1.0, 2.0):
        model = LinearSVC(C=c_value, class_weight="balanced", dual="auto", max_iter=5000)
        model.fit(x_train, y_train)
        logits = decision_logits(model, x_validation)
        temp = temperature(logits, y_validation)
        probs = softmax(logits / temp, axis=1)
        metrics, _, _ = classification_metrics(y_validation, probs.argmax(axis=1), probs)
        candidates.append((metrics["regional_macro_f1"], c_value, temp, model, metrics))
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, selected_c, selected_temp, svm, validation_metrics = candidates[0]
    model_id = "ID_CHAR_TFIDF_SVM"
    reports.append({"model_id": model_id, "split": "validation", **validation_metrics})
    for split_name, frame in protocols.items():
        reports.append(
            evaluate_id_model(model_id, svm, char_vectorizer, frame, split_name, selected_temp)
        )
    with (artifact_dir / "id_char_tfidf_svm.pkl").open("wb") as handle:
        pickle.dump({"vectorizer": char_vectorizer, "model": svm, "temperature": selected_temp}, handle)
    selection = {
        "selected_C": selected_c,
        "temperature": selected_temp,
        "validation_candidates": [
            {"C": c, "temperature": t, "regional_macro_f1": score}
            for score, c, t, _, _ in candidates
        ],
        "fit_seconds": time.perf_counter() - started,
    }
    (PROJECT / "reports/model/id_svm_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    # A deliberately cheaper stochastic linear baseline with five paired seeds.
    for seed in (1701, 2903, 4307, 5501, 6703):
        model = SGDClassifier(
            loss="log_loss",
            alpha=1e-6,
            class_weight="balanced",
            max_iter=30,
            tol=1e-4,
            random_state=seed,
            average=True,
        )
        model.fit(x_train, y_train)
        validation_probabilities = model.predict_proba(x_validation)
        validation_metrics, _, _ = classification_metrics(
            y_validation, validation_probabilities.argmax(axis=1), validation_probabilities
        )
        reports.append(
            {
                "model_id": "ID_CHAR_TFIDF_SGD",
                "seed": seed,
                "split": "validation",
                **validation_metrics,
            }
        )
        for split_name, frame in protocols.items():
            matrix = char_vectorizer.transform(frame["text_model"].map(nfc))
            probabilities = model.predict_proba(matrix)
            labels = frame["dialect"].map(DIALECT_TO_ID).to_numpy()
            predictions = probabilities.argmax(axis=1)
            metrics, _, _ = classification_metrics(labels, predictions, probabilities)
            reports.append(
                {
                    "model_id": "ID_CHAR_TFIDF_SGD",
                    "seed": seed,
                    "split": split_name,
                    **metrics,
                }
            )

    # Majority is fitted only on training counts.
    majority = int(pd.Series(y_train).value_counts().idxmax())
    priors = np.bincount(y_train, minlength=len(DIALECTS)).astype(float)
    priors /= priors.sum()
    for split_name, frame in {"validation": validation, **protocols}.items():
        labels = frame["dialect"].map(DIALECT_TO_ID).to_numpy()
        predictions = np.full(len(frame), majority)
        probabilities = np.tile(priors, (len(frame), 1))
        metrics, _, _ = classification_metrics(labels, predictions, probabilities)
        reports.append({"model_id": "ID_MAJORITY", "split": split_name, **metrics})

    result = pd.DataFrame(reports)
    result.to_csv(PROJECT / "reports/model/classical_identification_results.csv", index=False)
    return result


def learn_word_map(
    frame: pd.DataFrame, *, min_count: int, minimum_confidence: float
) -> dict[tuple[str, str], str]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in frame.itertuples(index=False):
        source = nfc(row.source_text_model).split()
        target = nfc(row.target_text_model).split()
        matcher = SequenceMatcher(a=source, b=target, autojunk=False)
        for operation, i1, i2, j1, j2 in matcher.get_opcodes():
            if operation == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
                counts[(row.dialect, source[i1])][target[j1]] += 1
    mapping: dict[tuple[str, str], str] = {}
    for key, alternatives in counts.items():
        target, count = alternatives.most_common(1)[0]
        if count >= min_count and count / sum(alternatives.values()) >= minimum_confidence:
            mapping[key] = target
    return mapping


def rewrite(frame: pd.DataFrame, mapping: dict[tuple[str, str], str]) -> list[str]:
    results = []
    for row in frame.itertuples(index=False):
        tokens = nfc(row.source_text_model).split()
        results.append(" ".join(mapping.get((row.dialect, token), token) for token in tokens))
    return results


def evaluate_normalizer(model_id: str, frame: pd.DataFrame, predictions: list[str], split: str):
    output = frame[["row_id", "dialect", "source_id", "semantic_group_id"]].copy()
    output["source"] = frame["source_text_model"].map(nfc)
    output["reference"] = frame["target_text_model"].map(nfc)
    output["prediction"] = predictions
    metrics, by_dialect = normalization_metrics(output)
    prediction_dir = PROJECT / "predictions/normalization"
    metric_dir = PROJECT / "metrics/normalization"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(prediction_dir / f"{model_id}_{split}.parquet", index=False)
    by_dialect.to_csv(metric_dir / f"{model_id}_{split}_by_dialect.csv", index=False)
    payload = {"model_id": model_id, "split": split, **metrics}
    (metric_dir / f"{model_id}_{split}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_normalization() -> pd.DataFrame:
    data_dir = PROJECT / "data/final/v1"
    train = pd.read_parquet(data_dir / "normalization_train.parquet")
    # Synthetic perturbations must not define lexical replacements.
    authentic_train = train.loc[~train["is_synthetic"]].copy()
    validation = pd.read_parquet(data_dir / "normalization_validation.parquet")
    iid = pd.read_parquet(data_dir / "normalization_test_iid.parquet")
    ood = pd.read_parquet(data_dir / "normalization_test_ood.parquet")
    supported_ood = ood.loc[~ood["dialect"].eq("RAJ")].copy()
    zero_shot_raj = ood.loc[ood["dialect"].eq("RAJ")].copy()
    reports = []

    for split, frame in {
        "validation": validation,
        "iid_test": iid,
        "source_ood": supported_ood,
        "zero_shot_raj": zero_shot_raj,
    }.items():
        reports.append(
            evaluate_normalizer(
                "N_COPY", frame, frame["source_text_model"].map(nfc).tolist(), split
            )
        )

    candidates = []
    for min_count in (2, 3, 5):
        for confidence in (0.5, 0.7, 0.9):
            mapping = learn_word_map(
                authentic_train, min_count=min_count, minimum_confidence=confidence
            )
            predictions = rewrite(validation, mapping)
            evaluation_frame = pd.DataFrame(
                {
                    "prediction": predictions,
                    "reference": validation["target_text_model"],
                    "dialect": validation["dialect"],
                }
            )
            metrics, _ = normalization_metrics(evaluation_frame)
            candidates.append((metrics["macro_chrfpp"], min_count, confidence, mapping))
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, selected_count, selected_confidence, mapping = candidates[0]
    for split, frame in {
        "validation": validation,
        "iid_test": iid,
        "source_ood": supported_ood,
        "zero_shot_raj": zero_shot_raj,
    }.items():
        reports.append(evaluate_normalizer("N_WORD_REWRITE", frame, rewrite(frame, mapping), split))
    with (PROJECT / "artifacts/baselines/normalization_word_rewrite.pkl").open("wb") as handle:
        pickle.dump(mapping, handle)
    (PROJECT / "reports/model/normalization_rewrite_selection.json").write_text(
        json.dumps(
            {
                "selected_min_count": selected_count,
                "selected_confidence": selected_confidence,
                "mapping_entries": len(mapping),
                "validation_grid": [
                    {"macro_chrfpp": score, "min_count": count, "confidence": confidence}
                    for score, count, confidence, _ in candidates
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = pd.DataFrame(reports)
    result.to_csv(PROJECT / "reports/model/classical_normalization_results.csv", index=False)
    return result


def main() -> None:
    id_results = run_identification()
    norm_results = run_normalization()
    print("Identification baselines:")
    print(
        id_results.groupby(["model_id", "split"], dropna=False)["regional_macro_f1"]
        .mean()
        .to_string()
    )
    print("\nNormalization baselines:")
    print(norm_results[["model_id", "split", "macro_chrfpp", "worst_dialect_chrfpp"]].to_string(index=False))


if __name__ == "__main__":
    main()
