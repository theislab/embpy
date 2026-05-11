"""Lightweight logging helpers.

Kept dependency-free so unit tests don't need a configured root logger.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
) -> None:
    """Configure the root logger with a stream handler and optional file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger that defers to root configuration."""
    return logging.getLogger(name)


__all__ = ["get_logger", "setup_logging"]
