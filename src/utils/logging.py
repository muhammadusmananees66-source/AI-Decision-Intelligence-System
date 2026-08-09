"""Structured JSON-capable logging setup used across the platform."""
from __future__ import annotations

import logging
import sys
from typing import Any

from src.utils.config import get_settings


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger configured with a consistent format.

    Format includes timestamp, level, logger name, and message so logs are
    greppable locally and parseable by log aggregators (e.g. Fluentd, Loki)
    in staging/prod.
    """
    settings = get_settings()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured log line: `event key1=val1 key2=val2`."""
    kv = " ".join(f"{k}={v!r}" for k, v in fields.items())
    logger.info(f"{event} {kv}".strip())