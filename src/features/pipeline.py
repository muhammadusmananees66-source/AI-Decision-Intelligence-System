"""Feature engineering pipeline for the customer risk/churn model.

Wraps missing-value imputation, categorical encoding, and numeric scaling in
a single scikit-learn ColumnTransformer so training and inference apply
identical transformations (preventing train/serve skew).
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "satisfaction_score",
    "contract_length_months",
]
CATEGORICAL_FEATURES = ["region", "plan_type"]
TARGET = "churned"
ID_COLUMN = "customer_id"


@dataclass
class FeatureSet:
    X: pd.DataFrame
    y: pd.Series | None
    feature_names_in: list[str]


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer: impute -> scale (numeric), impute -> one-hot (categorical)."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_features_target(df: pd.DataFrame) -> FeatureSet:
    """Separate raw dataframe into feature matrix X and target y (if present)."""
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[feature_cols].copy()
    y = df[TARGET].copy() if TARGET in df.columns else None
    return FeatureSet(X=X, y=y, feature_names_in=feature_cols)


def validate_features(df: pd.DataFrame) -> list[str]:
    """Lightweight data validation: schema presence, dtype sanity, range checks.

    Returns a list of human-readable validation issues (empty if all pass).
    """
    issues: list[str] = []
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            issues.append(f"missing numeric column: {col}")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"column {col} expected numeric dtype, got {df[col].dtype}")
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            issues.append(f"missing categorical column: {col}")
    if "tenure_months" in df.columns and (df["tenure_months"] < 0).any():
        issues.append("tenure_months contains negative values")
    if "satisfaction_score" in df.columns:
        out_of_range = df["satisfaction_score"].dropna()
        if ((out_of_range < 0) | (out_of_range > 10)).any():
            issues.append("satisfaction_score out of expected [0,10] range")
    return issues


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str) -> ColumnTransformer:
    return joblib.load(path)