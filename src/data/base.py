"""Base abstractions for data ingestion connectors.

Every concrete connector implements `DataConnector.load()` and returns a
pandas DataFrame (for tabular sources) or list[dict] (for document sources),
so downstream pipelines can treat all sources uniformly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.utils.logging import get_logger, log_event

logger = get_logger(__name__)


class DataConnector(ABC):
    """Abstract base class for all data ingestion connectors."""

    source_type: str = "unknown"

    @abstractmethod
    def load(self, *args: Any, **kwargs: Any) -> Any:
        """Load data from the source and return it in a normalized form."""
        raise NotImplementedError

    def _log_loaded(self, n_records: int, **extra: Any) -> None:
        log_event(logger, "data_loaded", source=self.source_type, n_records=n_records, **extra)


class DataIngestionError(RuntimeError):
    """Raised when a connector fails to load data from its source."""