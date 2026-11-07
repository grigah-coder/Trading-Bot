"""Database and persistence package.

Provides SQLite persistence (transactions, migrations, unique constraints),
repositories for orders/fills/positions/trades/signals/risk events/account
snapshots, startup recovery and broker-to-database reconciliation.

Implemented in Phase 8 (Database and Persistence). Intentionally empty of
trading logic in Phase 1.
"""

__all__ = ["models", "migrations", "repository"]
