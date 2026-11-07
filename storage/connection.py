"""SQLite connection factory with WAL mode and foreign keys enabled."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse, unquote

from core.exceptions import ConfigurationError


def parse_sqlite_path(url: str) -> str:
    """Return the filesystem path for a ``sqlite:///`` URL.

    Supports ``sqlite:///:memory:`` and absolute/relative file paths.  A
    ``file:`` query parameter is accepted for clarity.
    """
    parsed = urlparse(url)
    if parsed.scheme != "sqlite":
        raise ConfigurationError("Only sqlite:// URLs are supported")
    if parsed.netloc and parsed.netloc not in {"", ":memory:"}:
        raise ConfigurationError("SQLite URLs cannot specify a network host")
    path = unquote(parsed.path) if parsed.path else ""
    if path.startswith("/"):
        path = path[1:]
    if not path or path == ":memory:":
        return ":memory:"
    return path


def connect(url: str, *, factory: Callable[..., sqlite3.Connection] = sqlite3.Connection) -> sqlite3.Connection:
    """Open a SQLite connection with the bot's safety pragmas enabled."""
    path = parse_sqlite_path(url)
    if path != ":memory:":
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
    # Autocommit mode: the database layer issues explicit BEGIN/COMMIT so a
    # multi-statement trading operation can never be partially committed by an
    # implicit sqlite3 transaction boundary.
    conn = sqlite3.connect(
        path,
        factory=factory,
        check_same_thread=False,
        isolation_level=None,
    )
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.row_factory = sqlite3.Row
    return conn


def sqlite_path_from_url(url: str) -> str:
    """Public helper used by tests to derive the on-disk path."""
    return parse_sqlite_path(url)
