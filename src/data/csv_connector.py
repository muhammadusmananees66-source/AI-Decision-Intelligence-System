"""CSV ingestion connector."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.base import DataConnector, DataIngestionError


class CSVConnector(DataConnector):
    """Loads tabular data from a local or mounted CSV file."""

    source_type = "csv"

    def load(self, path: str | Path, **read_csv_kwargs: object) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise DataIngestionError(f"CSV file not found: {path}")
        try:
            df = pd.read_csv(path, **read_csv_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise DataIngestionError(f"Failed to parse CSV {path}: {exc}") from exc
        self._log_loaded(len(df), path=str(path))
        return df