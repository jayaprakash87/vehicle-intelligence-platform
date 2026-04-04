"""Structured logging with JSON output and correlation IDs.

All log records include:
  - timestamp (ISO 8601)
  - level
  - logger name
  - message
  - run_id  (correlation ID — ties all log lines in a pipeline run together)

Use ``configure_logging()`` once at process start (CLI entry point) to set
the format and run ID.  Module-level code just calls ``get_logger(__name__)``.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Correlation ID (run_id)
# ---------------------------------------------------------------------------

_run_id: ContextVar[str] = ContextVar("run_id", default="")


def set_run_id(run_id: str | None = None) -> str:
    """Set the current run ID.  Returns the (possibly generated) ID."""
    rid = run_id or _new_run_id()
    _run_id.set(rid)
    return rid


def get_run_id() -> str:
    return _run_id.get()


def _new_run_id() -> str:
    """Generate a short, human-readable run ID: YYYYMMDD-HHMMSS-xxxx."""
    now = datetime.now(tz=timezone.utc)
    short = uuid.uuid4().hex[:4]
    return f"{now:%Y%m%d-%H%M%S}-{short}"


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class _JSONFormatter(logging.Formatter):
    """Emits one JSON object per log line — ready for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": get_run_id(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class _PrettyFormatter(logging.Formatter):
    """Human-readable format for interactive terminal use."""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        run_id = get_run_id()
        if run_id:
            # Copy the record so we don't mutate msg for other handlers
            record = logging.makeLogRecord(record.__dict__)
            record.msg = f"[{run_id}] {record.msg}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_configured = False


def configure_logging(
    *,
    level: int = logging.INFO,
    json_format: bool = False,
    run_id: str | None = None,
) -> str:
    """Configure the root logger once.  Returns the active run_id.

    Parameters
    ----------
    level : int
        Logging level (default INFO).
    json_format : bool
        If True, emit JSON lines.  If False, use human-readable format.
    run_id : str | None
        Explicit run ID.  None → auto-generate.
    """
    global _configured
    rid = set_run_id(run_id)

    if _configured:
        return rid
    _configured = True

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter() if json_format else _PrettyFormatter())
    root.addHandler(handler)

    return rid


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger.

    If ``configure_logging()`` has not been called yet (e.g. in tests),
    a default handler is attached so logs are not silently dropped.
    """
    logger = logging.getLogger(name)
    if not _configured and not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_PrettyFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
