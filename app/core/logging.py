"""
Structured logging.

Production services are read by machines (log aggregators) far more often
than by humans watching a terminal, so the default format is one JSON
object per line. Every log line automatically carries the request's
correlation ID (set by RequestContextMiddleware) so a single request can be
traced end-to-end across ingestion, retrieval, and generation.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
api_key_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("api_key_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 6),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
            "api_key_id": api_key_id_ctx.get(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"[{self.formatTime(record, '%H:%M:%S')}] {record.levelname:<8} " \
               f"rid={request_id_ctx.get()} {record.name}: {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root.addHandler(handler)

    # Keep noisy libraries at a sane level; they still inherit the handler above.
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).setLevel("WARNING")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_extra(**fields: Any) -> dict[str, Any]:
    """Usage: logger.info('ingested document', extra=log_extra(doc_id=doc_id, chunks=12))"""
    return {"extra_fields": fields}
