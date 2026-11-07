"""Configuration package.

Loads and validates application settings from environment variables and an
optional ``.env`` file. Enforces the paper-trading-only operating mode.

Implemented in Phase 2 (Core Architecture).
"""

from trading_bot.config.settings import (
    BrokerMode,
    Settings,
    TradingMode,
    load_settings,
)

__all__ = ["Settings", "load_settings", "TradingMode", "BrokerMode"]
