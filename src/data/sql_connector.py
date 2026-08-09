"""SQL database ingestion connector.

Uses Python's built-in sqlite3 for a zero-dependency, runnable-out-of-the-box
demo. In staging/prod, point `connection_string` at Postgres/MySQL/etc. via
SQLAlchemy -- the interface (`load(query)`) stays identical.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.data.base import DataConnector, DataIngestionError


class SQLConnector(DataConnector):
    """Loads tabular data by executing a SQL query against a database."""

    source_type = "sql"

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def load(self, query: str) -> pd.DataFrame:
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
        except Exception as exc:  # noqa: BLE001
            raise DataIngestionError(f"SQL query failed against {self.db_path}: {exc}") from exc
        self._log_loaded(len(df), db_path=str(self.db_path))
        return df

    def write(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)