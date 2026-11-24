import json
import logging
import sys

import pytest

from fraud_common.logging_config import JsonFormatter, configure_logging
from fraud_common.settings import LoggingSettings


def _record(msg: str = "hello %s", args: tuple[object, ...] = ("world",)) -> logging.LogRecord:
    return logging.LogRecord("test.logger", logging.WARNING, __file__, 1, msg, args, None)


def test_json_has_core_fields() -> None:
    out = json.loads(JsonFormatter("scorer").format(_record()))
    assert out["msg"] == "hello world"
    assert out["level"] == "WARNING"
    assert out["service"] == "scorer"
    assert out["logger"] == "test.logger"
    assert out["ts"].endswith("+00:00")


def test_json_includes_extras_and_handles_unserialisable() -> None:
    rec = _record("scored", ())
    rec.tx_id = "abc"
    rec.obj = object()
    out = json.loads(JsonFormatter("s").format(rec))
    assert out["tx_id"] == "abc"
    assert out["obj"].startswith("<object object")


def test_json_includes_exception() -> None:
    def boom() -> None:
        raise ValueError("boom")

    try:
        boom()
    except ValueError:
        rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    out = json.loads(JsonFormatter("s").format(rec))
    assert "ValueError: boom" in out["exc"]


def test_output_is_single_line_even_with_newlines() -> None:
    line = JsonFormatter("s").format(_record("a\nb", ()))
    assert "\n" not in line


@pytest.mark.parametrize("fmt", ["json", "console"])
def test_configure_is_idempotent(monkeypatch: pytest.MonkeyPatch, fmt: str) -> None:
    monkeypatch.setenv("LOG_FORMAT", fmt)
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    configure_logging(LoggingSettings(), "svc")
    configure_logging(LoggingSettings(), "svc")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.WARNING
    assert isinstance(root.handlers[0].formatter, JsonFormatter) is (fmt == "json")
