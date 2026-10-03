"""Structured logging utilities.

Provides a project-wide logger with consistent format and level
configuration.  Log files live under ``results/`` by default.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ANSI colour helpers used in console output.
_COLORS = {
    "DEBUG": "\033[0;36m",  # cyan
    "INFO": "\033[0;32m",   # green
    "WARNING": "\033[0;33m",  # yellow
    "ERROR": "\033[0;31m",  # red
    "CRITICAL": "\033[0;35m",  # purple
    "RESET": "\033[0m",
}


class TimestampFormatter(logging.Formatter):
    """Formatter that appends wall-clock time and thread id."""

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            base = super().format(record)
            return f"[{base}]"
        return super().format(record)


def build_logger(
    name: str = "hgnn_drl",
    *,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """Create and configure a logger for the project.

    Parameters
    ----------
    name : str
        Logger name (e.g. ``__name__``).
    level : int
        Logging level (default ``INFO``).
    log_file : str or None
        If given, also write logs to this file.
    console : bool
        If True, stream logs to ``sys.stdout``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Unique fmt per call so that handlers don't duplicate formatters.
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"

    formatter = TimestampFormatter(fmt=fmt, datefmt=datefmt)

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(level)
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Avoid duplicate handlers on repeated calls.
    logger.propagate = False
    return logger


def get_logger(name: str = "hgnn_drl") -> logging.Logger:
    """Return a pre-configured logger (reuses the default build)."""
    return logging.getLogger(name)


# Backwards-compatible alias.
logger = build_logger
logger = get_logger  # type: ignore[assignment]
