"""Small, explicit logging configuration helpers."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


def configure_logging(
    *,
    level: int = logging.INFO,
    log_file: Path | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure console logging and, optionally, a rotating log file."""
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root
