"""Market data package.

Provides the Alpaca market-data provider, OHLCV processing, data validation,
freshness checks, timezone normalization, market-session detection, caching,
retry policies and rate-limit handling.

Implemented in Phase 3 (Market Data Engine).
"""

from trading_bot.data.market_calendar import MarketCalendar
from trading_bot.data.provider import AlpacaDataProvider
from trading_bot.data.validation import (
    Bar,
    DataValidationError,
    validate_bar,
    validate_bars,
)

__all__ = [
    "AlpacaDataProvider",
    "Bar",
    "DataValidationError",
    "validate_bar",
    "validate_bars",
    "MarketCalendar",
]
