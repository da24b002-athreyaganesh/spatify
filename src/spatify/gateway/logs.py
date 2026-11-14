"""Structured logging: one JSON object per line, so logs can be filtered by field."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

# Extra fields passed with `log.info(..., extra={...})` that are copied into the JSON.
FIELDS = ("request_id", "method", "path", "status", "ms", "model_version", "detail")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        data.update({key: getattr(record, key) for key in FIELDS if hasattr(record, key)})
        if record.exc_info:
            data["error"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
