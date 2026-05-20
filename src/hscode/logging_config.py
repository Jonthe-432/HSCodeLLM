"""Structured logging setup for HSCode."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from hscode.config import SETTINGS

_CONFIGURED = False


def configure_logging(level: Optional[str] = None) -> None:
    """Configure the root logger for the ``hscode`` package.

    Idempotent — safe to call multiple times.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger = logging.getLogger("hscode")
    logger.setLevel((level or SETTINGS.log_level).upper())

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``hscode`` package logger."""
    configure_logging()
    return logging.getLogger(f"hscode.{name}")
