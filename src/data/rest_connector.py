"""REST API ingestion connector with retry/backoff."""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data.base import DataConnector, DataIngestionError


class RESTConnector(DataConnector):
    """Loads JSON data from a REST API endpoint."""

    source_type = "rest_api"

    def __init__(self, base_url: str, timeout: float = 10.0, headers: dict[str, str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
    def _get(self, path: str, params: dict[str, Any] | None) -> httpx.Response:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}{path}", params=params, headers=self.headers)
            resp.raise_for_status()
            return resp

    def load(self, path: str = "", params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            resp = self._get(path, params)
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise DataIngestionError(f"REST call failed for {self.base_url}{path}: {exc}") from exc
        records = data if isinstance(data, list) else [data]
        self._log_loaded(len(records), url=f"{self.base_url}{path}")
        return records