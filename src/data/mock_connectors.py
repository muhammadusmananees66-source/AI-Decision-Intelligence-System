"""Mock connectors for sources that require external systems in production:
email inboxes (IMAP/Graph API), web pages (scraper), and streaming platforms
(Kafka/Kinesis). Each exposes the same `load()` contract as real connectors
so pipelines are agnostic to whether a source is mocked or live -- in prod,
swap the class import for a real implementation (e.g. `KafkaConnector`).
"""
from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

from src.data.base import DataConnector


class MockEmailConnector(DataConnector):
    """Simulates pulling recent emails from an inbox (stand-in for IMAP/MS Graph)."""

    source_type = "email"

    _SUBJECTS = [
        "Q3 forecast review", "Vendor risk flag", "Customer churn alert",
        "Weekly ops report", "Budget approval needed",
    ]

    def load(self, n: int = 5, seed: int | None = 42) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        now = datetime.now(UTC)
        emails = [
            {
                "id": f"email-{i}",
                "subject": rng.choice(self._SUBJECTS),
                "sender": f"user{i}@example-corp.com",
                "received_at": (now - timedelta(hours=i)).isoformat(),
                "body": f"This is a mock email body #{i} regarding operational updates.",
            }
            for i in range(n)
        ]
        self._log_loaded(len(emails))
        return emails


class MockWebPageConnector(DataConnector):
    """Simulates scraping structured content from web pages."""

    source_type = "web_page"

    def load(self, urls: list[str] | None = None) -> list[dict[str, str]]:
        urls = urls or ["https://example-corp.com/news/1", "https://example-corp.com/news/2"]
        pages = [
            {"url": u, "title": f"Mock article for {u}", "text": f"Mock scraped body text for {u}."}
            for u in urls
        ]
        self._log_loaded(len(pages))
        return pages


class MockStreamingConnector(DataConnector):
    """Simulates a streaming source (stand-in for Kafka/Kinesis consumer)."""

    source_type = "streaming"

    def load(self, n_events: int = 10, seed: int | None = 42) -> Iterator[dict[str, Any]]:
        rng = random.Random(seed)
        for i in range(n_events):
            event = {
                "event_id": i,
                "ts": datetime.now(UTC).isoformat(),
                "metric": "transaction_amount",
                "value": round(rng.uniform(10, 5000), 2),
            }
            yield event
        self._log_loaded(n_events)