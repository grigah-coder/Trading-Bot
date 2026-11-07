"""Database lifecycle, explicit transactions, and repository access.

A single :class:`TradingDatabase` instance owns the SQLite connection for the
process.  It opens the database, applies migrations, exposes repositories, and
provides explicit transaction scopes.

Nested transaction scopes use SQLite savepoints, so repository methods can be
composed safely::

    with database.transaction():
        database.repositories.orders.upsert(order)
        database.repositories.executions.insert(fill)

If anything raises, every write in the outer scope is rolled back.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Sequence

from config.settings import DatabaseSettings, Settings
from storage.connection import connect
from storage.repositories import RepositoryRegistry
from storage.schema import SCHEMA_VERSION, current_version, migrate
from storage.validation import StorageError, utc_timestamp


class TradingDatabase:
    """Own and validate one SQLite connection."""

    def __init__(
        self,
        url: str = "sqlite:///data/trading_bot.sqlite",
        *,
        connection: sqlite3.Connection | None = None,
        migrate_schema: bool = True,
        clock: Any = None,
    ) -> None:
        self.url = url
        self._lock = threading.RLock()
        self._depth = 0
        self._savepoint_counter = 0
        self._clock = clock or (lambda: datetime.now().astimezone())
        self.connection = connection or connect(url)
        if not migrate_schema:
            # Still ensure the schema_migrations table exists so health checks
            # and version queries have a well-defined answer.
            from storage.schema import _ensure_migration_table  # local import

            _ensure_migration_table(self.connection)
        else:
            migrate(self.connection)
        self.repositories = RepositoryRegistry.create(self)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        migrate_schema: bool = True,
    ) -> "TradingDatabase":
        return cls(settings.database.url, migrate_schema=migrate_schema)

    @classmethod
    def from_database_settings(
        cls,
        settings: DatabaseSettings,
        *,
        migrate_schema: bool = True,
    ) -> "TradingDatabase":
        return cls(settings.url, migrate_schema=migrate_schema)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> "TradingDatabase":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            connection = getattr(self, "connection", None)
            if connection is None:
                return
            try:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
            finally:
                connection.close()
                self.connection = None  # type: ignore[assignment]

    @property
    def schema_version(self) -> int:
        return current_version(self._require_connection())

    def health_check(self) -> None:
        """Fail loudly if the database is unavailable or has the wrong schema."""
        with self._lock:
            connection = self._require_connection()
            try:
                connection.execute("SELECT 1").fetchone()
                foreign_keys = int(connection.execute("PRAGMA foreign_keys").fetchone()[0])
                version = current_version(connection)
            except sqlite3.Error as exc:
                raise StorageError("Database health check failed") from exc
            if foreign_keys != 1:
                raise StorageError("SQLite foreign key enforcement is disabled")
            if version != SCHEMA_VERSION:
                raise StorageError(
                    "Database schema version does not match this build",
                    context={"actual": version, "expected": SCHEMA_VERSION},
                )

    # ------------------------------------------------------------------
    # Transactions and statements
    # ------------------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator["TradingDatabase"]:
        """Run a block atomically.

        The outermost scope uses ``BEGIN IMMEDIATE`` so writers serialise and a
        crash cannot leave a partially applied trading operation.  Nested scopes
        use savepoints.
        """
        with self._lock:
            connection = self._require_connection()
            savepoint: str | None = None
            started_outer = False
            if self._depth == 0:
                if not connection.in_transaction:
                    connection.execute("BEGIN IMMEDIATE")
                    started_outer = True
            else:
                savepoint = f"sp_{self._savepoint_counter}"
                self._savepoint_counter += 1
                connection.execute(f"SAVEPOINT {savepoint}")
            self._depth += 1
            try:
                yield self
            except BaseException:
                self._depth -= 1
                if savepoint is not None:
                    connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                elif started_outer:
                    connection.execute("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if savepoint is not None:
                    connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                elif started_outer:
                    connection.execute("COMMIT")

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            try:
                return self._require_connection().execute(sql, tuple(parameters))
            except sqlite3.IntegrityError:
                raise
            except sqlite3.Error as exc:
                raise StorageError("Database statement failed") from exc

    def executemany(
        self, sql: str, parameters: Sequence[Sequence[Any]]
    ) -> sqlite3.Cursor:
        with self._lock:
            try:
                return self._require_connection().executemany(sql, parameters)
            except sqlite3.IntegrityError:
                raise
            except sqlite3.Error as exc:
                raise StorageError("Database batch statement failed") from exc

    def fetchone(
        self, sql: str, parameters: Sequence[Any] = ()
    ) -> sqlite3.Row | None:
        with self._lock:
            return self.execute(sql, parameters).fetchone()

    def fetchall(
        self, sql: str, parameters: Sequence[Any] = ()
    ) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.execute(sql, parameters).fetchall())

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def now(self) -> str:
        """Current UTC timestamp in the canonical stored format."""
        return utc_timestamp(self._clock(), "clock")

    def record_event(
        self,
        level: str,
        phase: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.repositories.application_events.record(
            level, phase, message, context=context
        )

    def _require_connection(self) -> sqlite3.Connection:
        if self.connection is None:
            raise StorageError("Database connection is closed")
        return self.connection
