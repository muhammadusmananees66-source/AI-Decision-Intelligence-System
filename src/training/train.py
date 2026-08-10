"""Training pipeline: trains and compares multiple candidate models,
performs hyperparameter search + cross-validation, evaluates on a held-out
test split, and registers the best model in the model registry.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline

from src.features.pipeline import build_preprocessor, split_features_target
from src.features.selection import build_feature_selection_step
from src.models.registry import ModelRegistry
from src.training.experiment_tracking import log_candidate_trial, log_winner, start_training_run
from src.utils.config import get_settings
from src.utils.logging import get_logger, log_event

logger = get_logger(__name__)

MODEL_NAME = "customer_churn_classifier"

# Post-encoding, this dataset has 12 features (5 numeric + 7 one-hot
# categorical columns). Tuned as part of each candidate's hyperparameter
# search alongside the model's own hyperparameters -- "all" is included so
# the search can conclude selection doesn't help for a given model.
FEATURE_SELECTION_K_CHOICES: list[int | str] = [6, 8, 10, "all"]

CANDIDATE_MODELS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "param_dist": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "param_dist": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [4, 8, 12, None],
            "clf__min_samples_leaf": [1, 2, 4],
        },
    },
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_dist": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.03, 0.1, 0.2],
            "clf__max_depth": [2, 3, 4],
        },
    },
    # Neural network candidate ("deep learning where appropriate"). For this
    # ~2K-row tabular dataset, a small MLP is the appropriate depth of neural
    # net -- a full PyTorch/TensorFlow deep model would be both unnecessary
    # and prone to overfitting here. Swap in a PyTorch model with the same
    # sklearn-compatible fit/predict_proba interface (e.g. via skorch) for
    # genuinely large tabular data, or reserve deep nets for the unstructured
    # data paths (RAG embeddings, document/image models) where they earn
    # their complexity -- see docs/architecture.md "Design Principles".
    "neural_network_mlp": {
        "estimator": MLPClassifier(random_state=42, max_iter=500, early_stopping=True),
        "param_dist": {
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "clf__alpha": [0.0001, 0.001, 0.01],
            "clf__learning_rate_init": [0.001, 0.01],
        },
    },
}


def evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def train_and_compare(
    df: pd.DataFrame, n_search_iter: int = 6, cv_folds: int = 3, track_experiments: bool = True
) -> dict[str, Any]:
    """Train all candidate models with randomized hyperparameter search + CV,
    evaluate on a held-out test set, and return comparison results plus the
    best fitted pipeline. When `track_experiments` is True, every candidate
    trial is logged to MLflow as a nested run (not just the eventual winner).
    """
    feature_set = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        feature_set.X, feature_set.y, test_size=0.2, random_state=42, stratify=feature_set.y
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results: dict[str, Any] = {}
    best_name, best_score, best_pipeline, best_params = None, -1.0, None, None

    for name, cfg in CANDIDATE_MODELS.items():
        pipeline = SkPipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("feature_selection", build_feature_selection_step(k="all")),
                ("clf", cfg["estimator"]),
            ]
        )
        param_dist = {**cfg["param_dist"], "feature_selection__k": FEATURE_SELECTION_K_CHOICES}
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=min(n_search_iter, _count_combinations(param_dist)),
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        y_proba = search.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["cv_best_score"] = round(float(search.best_score_), 4)

        results[name] = {"metrics": metrics, "best_params": search.best_params_}
        log_event(logger, "model_trained", model=name, metrics=metrics)

        if track_experiments:
            try:
                log_candidate_trial(name, search.best_params_, metrics)
            except Exception as exc:  # noqa: BLE001
                log_event(logger, "mlflow_logging_failed", model=name, error=str(exc))

        if metrics["roc_auc"] > best_score:
            best_name, best_score = name, metrics["roc_auc"]
            best_pipeline = search.best_estimator_
            best_params = search.best_params_

    return {
        "results": results,
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "best_params": best_params,
        "best_metrics": results[best_name]["metrics"],
    }


def _count_combinations(param_dist: dict[str, list]) -> int:
    total = 1
    for v in param_dist.values():
        total *= len(v)
    return total


def run_training_job(csv_path: str | Path | None = None, registry: ModelRegistry | None = None) -> dict[str, Any]:
    """End-to-end training job: load data, train+compare models, register the winner."""
    settings = get_settings()
    csv_path = Path(csv_path) if csv_path else settings.data_raw_dir / "customers.csv"
    if not csv_path.exists():
        from src.data.synthetic_dataset import generate_customer_dataset

        df = generate_customer_dataset()
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    run_name = f"training-{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%S')}"

    # MLflow tracking is observability, not a hard training dependency: if the
    # tracking backend is unavailable, run_ctx becomes a no-op contextmanager
    # so training still proceeds untracked rather than failing the whole job.
    try:
        run_ctx = start_training_run(run_name, tags={"dataset_rows": str(len(df))})
        run_ctx.__enter__()
        mlflow_active = True
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "mlflow_tracking_unavailable", error=str(exc))
        run_ctx = None
        mlflow_active = False

    comparison = train_and_compare(df, track_experiments=mlflow_active)
    best_pipeline = comparison["best_pipeline"]

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        preproc_path = Path(tmpdir) / "preprocessor.joblib"
        joblib.dump(best_pipeline.named_steps["clf"], model_path)
        # Persist preprocessing + feature_selection together as one bundle:
        # the classifier was fit on the *selected* feature subset, so
        # inference must apply both steps in the same order, or the feature
        # matrix shape won't match what the model expects. Bundling them
        # keeps the registry's two-artifact (model + preprocessor) contract
        # unchanged for every downstream consumer (ChurnPredictor, the API,
        # explainability) -- `.transform()` on this bundle runs preprocessor
        # then feature_selection in sequence.
        preprocessing_bundle = SkPipeline(
            steps=[
                ("preprocessor", best_pipeline.named_steps["preprocessor"]),
                ("feature_selection", best_pipeline.named_steps["feature_selection"]),
            ]
        )
        joblib.dump(preprocessing_bundle, preproc_path)

        registry = registry or ModelRegistry()
        version = registry.register(
            model_name=MODEL_NAME,
            source_model_path=model_path,
            source_preprocessor_path=preproc_path,
            metrics=comparison["best_metrics"],
            params={"model_type": comparison["best_model_name"], **comparison["best_params"]},
        )

    if mlflow_active:
        try:
            log_winner(comparison["best_model_name"], version.version, comparison["best_metrics"])
        finally:
            run_ctx.__exit__(None, None, None)

    summary = {
        "model_name": MODEL_NAME,
        "registered_version": version.version,
        "winner": comparison["best_model_name"],
        "all_results": {k: v["metrics"] for k, v in comparison["results"].items()},
    }
    log_event(logger, "training_job_complete", **{k: summary[k] for k in ("model_name", "registered_version", "winner")})
    return summary


if __name__ == "__main__":
    summary = run_training_job()
    print(json.dumps(summary, indent=2))