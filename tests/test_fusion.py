from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boichitro.fusion import (
    FUSION_FEATURE_NAMES,
    blend_identification_probabilities,
    inferred_dialect_rewrite,
    normalization_fusion_features,
    select_normalization_candidates,
    temperature_scale_probabilities,
)
from tools.run_normalization_fusion_pilot import semantic_group_folds


def test_normalization_features_are_reference_free_and_deterministic() -> None:
    probabilities = np.arange(1, 14, dtype=np.float64)
    first = normalization_fusion_features(
        "আইজ আমি যামু",
        "আজ আমি যাব",
        "আজ আমি যামু",
        probabilities,
    )
    second = normalization_fusion_features(
        "আইজ আমি যামু",
        "আজ আমি যাব",
        "আজ আমি যামু",
        probabilities,
    )
    assert first.shape == (len(FUSION_FEATURE_NAMES),)
    np.testing.assert_allclose(first, second)
    assert np.isfinite(first).all()


def test_normalization_features_require_thirteen_probabilities() -> None:
    with pytest.raises(ValueError, match="13 inferred dialect probabilities"):
        normalization_fusion_features("a", "b", "c", [0.5, 0.5])


def test_probability_blend_is_normalized() -> None:
    neural = np.eye(13, dtype=np.float64)[:2]
    svm = np.full((2, 13), 1.0 / 13)
    result = blend_identification_probabilities(neural, svm, neural_weight=0.425)
    np.testing.assert_allclose(result.sum(axis=1), 1.0)
    assert result[0, 0] > result[0, 1]


def test_probability_blend_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        blend_identification_probabilities(
            np.full((2, 13), 1.0 / 13),
            np.full((3, 13), 1.0 / 13),
            neural_weight=0.5,
        )


def test_temperature_scaling_is_normalized_and_validated() -> None:
    probabilities = np.full((2, 13), 0.02, dtype=np.float64)
    probabilities[:, 3] = 0.76
    result = temperature_scale_probabilities(probabilities, temperature=2.0)
    np.testing.assert_allclose(result.sum(axis=1), 1.0)
    assert result[0, 3] < probabilities[0, 3]
    with pytest.raises(ValueError, match="positive"):
        temperature_scale_probabilities(probabilities, temperature=0.0)


def test_inferred_rewrite_never_needs_a_gold_label() -> None:
    probabilities = np.zeros((1, 13), dtype=np.float64)
    probabilities[0, 0] = 1.0
    rewritten, labels = inferred_dialect_rewrite(
        ["আইজ যামু"], probabilities, {("BAR", "আইজ"): "আজ"}
    )
    assert labels == ["BAR"]
    assert rewritten == ["আজ যামু"]

    supported = np.zeros((1, 13), dtype=np.float64)
    supported[0, 8] = 0.9  # RAJ has no normalization rewrite inventory.
    supported[0, 0] = 0.1
    rewritten, labels = inferred_dialect_rewrite(
        ["আইজ"],
        supported,
        {("BAR", "আইজ"): "আজ"},
        allowed_dialects=["BAR", "CHI"],
    )
    assert labels == ["BAR"]
    assert rewritten == ["আজ"]


def test_selector_enforces_source_blind_feature_contract() -> None:
    class ConstantModel:
        def predict(self, features):
            return np.ones(len(features), dtype=np.float64)

    probabilities = np.full((1, 13), 1.0 / 13)
    artifact = {
        "model": ConstantModel(),
        "threshold": 0.0,
        "source_blind": True,
        "feature_names": list(FUSION_FEATURE_NAMES),
    }
    selected, margins, mask = select_normalization_candidates(
        ["ক"], ["খ"], ["গ"], probabilities, artifact
    )
    assert selected == ["খ"]
    np.testing.assert_allclose(margins, [1.0])
    assert mask.tolist() == [True]

    invalid = dict(artifact, source_blind=False)
    with pytest.raises(ValueError, match="not marked source-blind"):
        select_normalization_candidates(
            ["ক"], ["খ"], ["গ"], probabilities, invalid
        )


def test_selector_folds_hold_out_complete_semantic_groups() -> None:
    rows = []
    for repeat in range(2):
        for group in range(10):
            for realization in range(2):
                rows.append(
                    {
                        "row_id": f"row-{group}-{realization}",
                        "semantic_group_id": f"group-{group}",
                        "dialect": "BAR" if realization == 0 else "CHI",
                        "repeat": repeat,
                    }
                )
    frame = pd.DataFrame(rows)
    folds, _ = semantic_group_folds(frame, n_splits=2, random_state=17)
    assigned = frame.assign(fold=folds)
    assert assigned.groupby("semantic_group_id")["fold"].nunique().max() == 1
    assert assigned.groupby("row_id")["fold"].nunique().max() == 1
