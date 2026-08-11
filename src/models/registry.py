"""Lightweight model registry.

Tracks model versions, metrics, and artifact paths in a JSON file backed by
the local filesystem. This mirrors the interface of a "real" registry (e.g.
MLflow Model Registry) -- `register`, `get_latest`, `promote`, `list_versions`
-- so swapping in MLflow later only touches this module.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.config import get_settings
from src.utils.logging import get_logger, log_event

logger = get_logger(__name__)


@dataclass
class ModelVersion:
    model_name: str
    version: int
    artifact_path: str
    preprocessor_path: str
    metrics: dict[str, float]
    stage: str = "staging"  # staging | production | archived
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    params: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """JSON-file-backed model registry with versioning and stage promotion."""

    def __init__(self, registry_path: Path | None = None, artifact_dir: Path | None = None):
        settings = get_settings()
        self.registry_path = registry_path or settings.registry_file
        self.artifact_dir = artifact_dir or settings.model_store_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_all({})

    def _read_all(self) -> dict[str, list[dict]]:
        return json.loads(self.registry_path.read_text())

    def _write_all(self, data: dict[str, list[dict]]) -> None:
        self.registry_path.write_text(json.dumps(data, indent=2))

    def register(
        self,
        model_name: str,
        source_model_path: Path,
        source_preprocessor_path: Path,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        stage: str = "staging",
    ) -> ModelVersion:
        """Register a new model version, copying artifacts into the registry's storage."""
        data = self._read_all()
        versions = data.get(model_name, [])
        next_version = (max((v["version"] for v in versions), default=0)) + 1

        version_dir = self.artifact_dir / model_name / f"v{next_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        model_dest = version_dir / "model.joblib"
        preproc_dest = version_dir / "preprocessor.joblib"
        shutil.copy(source_model_path, model_dest)
        shutil.copy(source_preprocessor_path, preproc_dest)

        mv = ModelVersion(
            model_name=model_name,
            version=next_version,
            artifact_path=str(model_dest),
            preprocessor_path=str(preproc_dest),
            metrics=metrics,
            stage=stage,
            params=params or {},
        )
        versions.append(asdict(mv))
        data[model_name] = versions
        self._write_all(data)
        log_event(logger, "model_registered", model=model_name, version=next_version, metrics=metrics)
        return mv

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        data = self._read_all()
        return [ModelVersion(**v) for v in data.get(model_name, [])]

    def get_latest(self, model_name: str, stage: str | None = None) -> ModelVersion | None:
        versions = self.list_versions(model_name)
        if stage:
            versions = [v for v in versions if v.stage == stage]
        if not versions:
            return None
        return max(versions, key=lambda v: v.version)

    def promote(self, model_name: str, version: int, stage: str = "production") -> None:
        data = self._read_all()
        versions = data.get(model_name, [])
        found = False
        for v in versions:
            if v["version"] == version:
                v["stage"] = stage
                found = True
            elif stage == "production" and v["stage"] == "production":
                v["stage"] = "archived"  # demote previous production model
        if not found:
            raise ValueError(f"Version {version} not found for model {model_name}")
        data[model_name] = versions
        self._write_all(data)
        log_event(logger, "model_promoted", model=model_name, version=version, stage=stage)

    def delete_version(self, model_name: str, version: int) -> None:
        """Remove a model version's registry entry and its artifact files.
        Refuses to delete the current production version -- promote a
        different version first (this is the "no accidental prod deletion"
        guardrail a senior MLOps engineer would expect).
        """
        data = self._read_all()
        versions = data.get(model_name, [])
        target = next((v for v in versions if v["version"] == version), None)
        if target is None:
            raise ValueError(f"Version {version} not found for model {model_name}")
        if target["stage"] == "production":
            raise ValueError(
                f"Refusing to delete version {version}: it is the current production model. "
                "Promote a different version to production first."
            )
        data[model_name] = [v for v in versions if v["version"] != version]
        self._write_all(data)

        artifact_dir = Path(target["artifact_path"]).parent
        if artifact_dir.exists() and artifact_dir.is_relative_to(self.artifact_dir):
            shutil.rmtree(artifact_dir)
        log_event(logger, "model_version_deleted", model=model_name, version=version)