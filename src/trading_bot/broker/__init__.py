"""Broker integration package.

Provides the Alpaca broker adapter: account information, buying power, equity,
positions, orders (market/limit/cancel/replace), status monitoring, partial-fill
handling, broker error handling, idempotency and state reconciliation.

Implemented in Phase 4 (Broker Integration). Paper trading only.
"""

from trading_bot.broker.adapter import AlpacaBrokerAdapter

__all__ = ["AlpacaBrokerAdapter", "adapter"]
