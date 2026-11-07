"""trading_bot — reliable, risk-controlled algorithmic trading system.

Package layout (clean architecture):

* ``trading_bot.engine``   — central trading engine & application lifecycle
* ``trading_bot.data``     — market data provider (Alpaca), validation, caching
* ``trading_bot.broker``   — broker adapter (Alpaca paper trading)
* ``trading_bot.risk``     — independent risk engine (approves/rejects trades)
* ``trading_bot.strategy`` — strategy framework & signal generation
* ``trading_bot.order``    — order state machine & position lifecycle
* ``trading_bot.db``       — persistence (SQLite), migrations, repositories
* ``trading_bot.backtest`` — backtesting engine & quantitative metrics
* ``trading_bot.monitoring`` — health checks, alerts, notifications

Default operating mode: paper trading only. Live trading is disabled.
"""

__version__ = "2.0.0"

__all__ = [
    "engine",
    "data",
    "broker",
    "risk",
    "strategy",
    "order",
    "db",
    "backtest",
    "monitoring",
]
