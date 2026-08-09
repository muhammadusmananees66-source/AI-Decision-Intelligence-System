"""Object storage connector.

Implements an `ObjectStorageConnector` against the local filesystem so the
project is runnable without cloud credentials. The interface mirrors what a
boto3 (S3) or google-cloud-storage (GCS) backed implementation would expose,
so swapping the backend in prod is a drop-in replacement -- callers only
depend on `list_objects`, `get_object`, and `put_object`.
"""
from __future__ import annotations

from pathlib import Path

from src.data.base import DataConnector, DataIngestionError


class ObjectStorageConnector(DataConnector):
    """Filesystem-backed object storage connector (local stand-in for S3/GCS)."""

    source_type = "object_storage"

    def __init__(self, bucket_root: str | Path):
        self.bucket_root = Path(bucket_root)
        self.bucket_root.mkdir(parents=True, exist_ok=True)

    def list_objects(self, prefix: str = "") -> list[str]:
        matches = [
            str(p.relative_to(self.bucket_root))
            for p in self.bucket_root.rglob("*")
            if p.is_file() and str(p.relative_to(self.bucket_root)).startswith(prefix)
        ]
        self._log_loaded(len(matches), prefix=prefix)
        return sorted(matches)

    def get_object(self, key: str) -> bytes:
        path = self.bucket_root / key
        if not path.exists():
            raise DataIngestionError(f"Object not found: {key}")
        return path.read_bytes()

    def put_object(self, key: str, content: bytes) -> str:
        path = self.bucket_root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    def load(self, prefix: str = "") -> list[str]:
        return self.list_objects(prefix)