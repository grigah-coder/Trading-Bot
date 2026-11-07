"""Broker integration package.

Provides the Alpaca broker adapter: account information, buying power, equity,
positions, orders (market/limit/cancel/replace), status monitoring, partial-fill
handling, broker error handling, idempotency and state reconciliation.

Implemented in Phase 4 (Broker Integration). Intentionally empty of trading
logic in Phase 1.
"""

__all__ = ["adapter"]
