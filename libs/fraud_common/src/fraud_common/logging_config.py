"""Structured logging: one JSON object per line, so logs can be grepped by tx_id."""

from __future__ import annotations

import logging
import sys
import traceback
from datetime import UTC, datetime
from typing import Any, Final

import orjson

from fraud_common.settings import LoggingSettings

# Attributes every LogRecord has; anything else was passed via `extra=` and is emitted.
_RESERVED: Final = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = "".join(traceback.format_exception(*record.exc_info))
        return orjson.dumps(payload, default=str).decode()


def configure_logging(settings: LoggingSettings, service: str) -> None:
    """Install a single stdout handler on the root logger. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stdout)
    if settings.format == "json":
        handler.setFormatter(JsonFormatter(service))
    else:
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s %(levelname)-8s {service} %(name)s: %(message)s")
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.level)
