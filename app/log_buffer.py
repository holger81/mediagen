"""In-memory ring buffer of recent log records for the admin UI."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class LogLine:
    ts: str
    level: str
    logger: str
    message: str


class MemoryLogHandler(logging.Handler):
    """Keep the last N formatted log lines in memory."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.capacity = max(50, capacity)
        self._lines: deque[LogLine] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            line = LogLine(
                ts=datetime.fromtimestamp(record.created, tz=UTC)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                level=record.levelname,
                logger=record.name,
                message=msg,
            )
            with self._lock:
                self._lines.append(line)
        except Exception:
            self.handleError(record)

    def snapshot(self, *, limit: int = 200) -> list[LogLine]:
        limit = max(1, min(int(limit), self.capacity))
        with self._lock:
            items = list(self._lines)
        return items[-limit:]


_handler: MemoryLogHandler | None = None
_handler_lock = threading.Lock()


def get_log_handler(*, capacity: int = 500) -> MemoryLogHandler:
    global _handler
    with _handler_lock:
        if _handler is None:
            _handler = MemoryLogHandler(capacity=capacity)
        return _handler


def install_log_buffer(*, capacity: int = 500, level: int = logging.INFO) -> MemoryLogHandler:
    """Attach a shared memory handler to the app logger tree (idempotent)."""
    handler = get_log_handler(capacity=capacity)
    handler.setLevel(level)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level)
    app_logger = logging.getLogger("app")
    app_logger.setLevel(level)
    if handler not in app_logger.handlers:
        app_logger.addHandler(handler)
    # Also capture uvicorn access/error when present.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        if handler not in lg.handlers:
            lg.addHandler(handler)
    return handler
