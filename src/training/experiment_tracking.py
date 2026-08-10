"""Experiment tracking via MLflow.

Wraps the MLflow tracking API so every training run -- and every candidate
model trial within it, not just the eventual winner -- is logged with its
parameters, metrics, and artifacts. Uses a local file-based tracking URI
(`mlruns/`) by default so this works with zero external server; point
`AIDIP_MLFLOW_TRACKING_URI` at a real MLflow server (or Databricks) in
staging/prod without touching any calling code.

This is the "senior MLOps" upgrade over the model registry alone: the
registry answers "what's in production right now", MLflow answers
"what did we try, when, with what hyperparameters, and why did we pick
this one" -- full experiment lineage and comparison across runs.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import mlflow

from src.utils.config import get_settings
from src.utils.logging import get_logger, log_event

logger = get_logger(__name__)

EXPERIMENT_NAME = "customer_churn_classifier"


def _configure_tracking_uri() -> None:
    settings = get_settings()
    default_uri = f"sqlite:///{settings.model_store_dir / 'mlflow.db'}"
    tracking_uri = os.environ.get("AIDIP_MLFLOW_TRACKING_URI") or default_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


@contextmanager
def start_training_run(run_name: str, tags: dict[str, str] | None = None):
    """Parent run for a full train_and_compare() job (one per retraining trigger)."""
    _configure_tracking_uri()
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        log_event(logger, "mlflow_run_started", run_id=run.info.run_id, run_name=run_name)
        yield run


def log_candidate_trial(
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    parent_run_id: str | None = None,
) -> str:
    """Log one candidate model's hyperparameters + metrics as a nested MLflow run.

    Returns the child run_id so callers can cross-reference it against the
    model registry entry for the eventual winner.
    """
    with mlflow.start_run(run_name=model_name, nested=True) as run:
        mlflow.log_params({str(k): v for k, v in params.items()})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.set_tag("model_type", model_name)
        log_event(logger, "mlflow_trial_logged", run_id=run.info.run_id, model=model_name, metrics=metrics)
        return run.info.run_id


def log_winner(model_name: str, version: int, metrics: dict[str, float]) -> None:
    mlflow.set_tags({"winner_model": model_name, "registry_version": str(version)})
    mlflow.log_metrics({f"winner_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})


def list_recent_runs(max_results: int = 20) -> list[dict[str, Any]]:
    """Return recent MLflow runs for this experiment, newest first -- powers
    the dashboard's Experiments tab.
    """
    _configure_tracking_uri()
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
    )
    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "params": r.data.params,
            "metrics": r.data.metrics,
            "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")},
        }
        for r in runs
    ]