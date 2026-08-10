"""Feature selection.

Two complementary selection strategies, both operating on the *transformed*
feature matrix (post impute/scale/encode), since selection needs to see the
one-hot-expanded categorical columns to make a meaningful per-column
decision:

1. **Statistical (ANOVA F-value)**: `SelectKBest(f_classif)` -- fast,
   model-agnostic, good as a first-pass filter before any model is trained.
2. **Model-based (tree importance)**: fits a quick RandomForest and keeps
   features above a mean-importance threshold -- captures interactions the
   univariate F-test misses, at the cost of needing a trained model first.

`build_feature_selection_step` returns a scikit-learn-compatible transformer
that slots directly into the existing `Pipeline([("preprocessor", ...),
("feature_selection", ...), ("clf", ...)])` in `src/training/train.py`, so
selection is fit once per training run and reused unmodified at inference
time -- the same train/serve parity principle as the preprocessor itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif


@dataclass
class FeatureSelectionReport:
    method: str
    n_input_features: int
    n_selected_features: int
    selected_features: list[str]
    dropped_features: list[str]
    scores: dict[str, float]


def select_k_best_features(
    X_transformed: np.ndarray, y: pd.Series, feature_names: list[str], k: int | str = "all"
) -> FeatureSelectionReport:
    """ANOVA F-value univariate selection: ranks each (post-encoding) feature
    by how well it alone separates the two churn classes, and keeps the top k.
    """
    k_eff = min(k, len(feature_names)) if isinstance(k, int) else k
    selector = SelectKBest(score_func=f_classif, k=k_eff)
    selector.fit(X_transformed, y)

    scores = dict(zip(feature_names, np.nan_to_num(selector.scores_, nan=0.0).tolist(), strict=True))
    mask = selector.get_support()
    selected = [f for f, keep in zip(feature_names, mask, strict=True) if keep]
    dropped = [f for f, keep in zip(feature_names, mask, strict=True) if not keep]

    return FeatureSelectionReport(
        method="anova_f_value",
        n_input_features=len(feature_names),
        n_selected_features=len(selected),
        selected_features=selected,
        dropped_features=dropped,
        scores={k_: round(v, 4) for k_, v in scores.items()},
    )


def select_by_model_importance(
    X_transformed: np.ndarray, y: pd.Series, feature_names: list[str], importance_threshold: float = 0.5
) -> FeatureSelectionReport:
    """Model-based selection: fits a small RandomForest and keeps features
    whose importance is at least `importance_threshold` x the mean
    importance across all features (default: at or above average).
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_transformed, y)
    importances = rf.feature_importances_
    threshold_value = importance_threshold * importances.mean()

    mask = importances >= threshold_value
    selected = [f for f, keep in zip(feature_names, mask, strict=True) if keep]
    dropped = [f for f, keep in zip(feature_names, mask, strict=True) if not keep]
    scores = dict(zip(feature_names, importances.tolist(), strict=True))

    return FeatureSelectionReport(
        method="random_forest_importance",
        n_input_features=len(feature_names),
        n_selected_features=len(selected),
        selected_features=selected,
        dropped_features=dropped,
        scores={k: round(v, 4) for k, v in scores.items()},
    )


def build_feature_selection_step(k: int | str = "all") -> SelectKBest:
    """Returns a scikit-learn transformer usable as a Pipeline step:
    `Pipeline([("preprocessor", ...), ("feature_selection", build_feature_selection_step(k=15)), ("clf", ...)])`.

    Using `SelectKBest` (rather than the RandomForest-importance selector)
    as the pipeline-embeddable step keeps the pipeline fast to fit
    repeatedly under `RandomizedSearchCV` and avoids double-fitting a
    RandomForest model on every CV fold just to select features for a
    *different* candidate model.
    """
    return SelectKBest(score_func=f_classif, k=k)


def drop_zero_variance_features(X_transformed: np.ndarray, feature_names: list[str]) -> FeatureSelectionReport:
    """Drops constant (zero-variance) columns -- the cheapest possible
    selection pass, useful as a pre-filter before the more expensive
    statistical/model-based methods, and a real risk after one-hot encoding
    rare categories in small datasets.
    """
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X_transformed)
    mask = selector.get_support()
    selected = [f for f, keep in zip(feature_names, mask, strict=True) if keep]
    dropped = [f for f, keep in zip(feature_names, mask, strict=True) if not keep]
    return FeatureSelectionReport(
        method="variance_threshold",
        n_input_features=len(feature_names),
        n_selected_features=len(selected),
        selected_features=selected,
        dropped_features=dropped,
        scores={},
    )