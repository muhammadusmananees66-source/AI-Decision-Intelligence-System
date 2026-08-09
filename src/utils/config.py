"""
Central configuration for the Enterprise AI Decision Intelligence Platform.

All runtime configuration is environment-driven so the same codebase runs
unmodified across dev/staging/prod. Values fall back to sane local defaults
so the project works out of the box with zero setup.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIDIP_", env_file=".env", extra="ignore")

    # Environment
    environment: str = Field(default="dev", description="dev | staging | prod")
    log_level: str = Field(default="INFO")

    # Storage
    data_raw_dir: Path = Field(default=REPO_ROOT / "data_store" / "raw")
    data_processed_dir: Path = Field(default=REPO_ROOT / "data_store" / "processed")
    model_store_dir: Path = Field(default=REPO_ROOT / "model_store")
    sqlite_path: Path = Field(default=REPO_ROOT / "data_store" / "enterprise.db")

    # Model registry
    registry_file: Path = Field(default=REPO_ROOT / "model_store" / "registry.json")

    # RAG
    rag_index_dir: Path = Field(default=REPO_ROOT / "data_store" / "rag_index")
    rag_chunk_size: int = Field(default=400)
    rag_chunk_overlap: int = Field(default=60)
    rag_top_k: int = Field(default=4)
    rag_backend: str = Field(default="tfidf", description="tfidf (in-memory) | chroma (persistent vector DB)")

    # API
    api_title: str = Field(default="Enterprise AI Decision Intelligence Platform")
    api_version: str = Field(default="0.1.0")

    # Monitoring
    drift_psi_threshold: float = Field(default=0.2)
    drift_ks_pvalue_threshold: float = Field(default=0.05)

    def ensure_dirs(self) -> None:
        for d in [
            self.data_raw_dir,
            self.data_processed_dir,
            self.model_store_dir,
            self.rag_index_dir,
            self.sqlite_path.parent,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings