"""Custom exception hierarchy for the trading bot.

All exceptions in this module are pure standard library and carry no trading
logic. They are used to distinguish operational failure modes so callers can
fail closed (block new trades) instead of failing open.
"""

from __future__ import annotations

from typing import Optional


class TradingBotError(Exception):
    """Base exception for all trading-bot errors."""


class ConfigurationError(TradingBotError):
    """Raised when configuration is invalid, missing or unreadable."""


class CredentialsError(ConfigurationError):
    """Raised when API credentials are missing or invalid."""


class MarketDataError(TradingBotError):
    """Raised when market data is unavailable, stale or failed validation."""

    def __init__(self, message: str, symbol: Optional[str] = None) -> None:
        super().__init__(message)
        self.symbol = symbol


class DataStaleError(MarketDataError):
    """Raised when market data freshness cannot be guaranteed."""


class BrokerError(TradingBotError):
    """Base exception for broker-adapter failures."""


class BrokerUnavailableError(BrokerError):
    """Raised when the broker endpoint cannot be reached."""


class OrderRejectedError(BrokerError):
    """Raised when the broker rejects an order."""


class OrderTimeoutError(BrokerError):
    """Raised when an order is not acknowledged within its deadline."""


class DuplicateOrderError(BrokerError):
    """Raised when a duplicate order submission is detected."""


class RiskError(TradingBotError):
    """Base exception for risk-engine failures."""


class RiskDataUnavailableError(RiskError):
    """Raised when account/market/broker state is too uncertain to assess risk."""


class RiskRejectedError(RiskError):
    """Raised when the risk engine rejects a trade."""


class CircuitBreakerTrippedError(RiskError):
    """Raised when a circuit breaker has halted new entries."""


class OrderStateError(TradingBotError):
    """Raised when an order state transition is invalid."""


class DatabaseError(TradingBotError):
    """Raised when persistence fails."""


class StrategyError(TradingBotError):
    """Raised when a strategy produces an invalid signal."""


class BacktestError(TradingBotError):
    """Raised when backtesting assumptions are violated."""
