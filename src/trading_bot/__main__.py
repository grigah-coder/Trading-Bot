"""Application entry point.

Usage::

    python -m trading_bot        # runs the paper-trading engine
    TRADING_MODE=paper python -m trading_bot

Defaults to paper trading. Live trading is disabled by configuration.

Wires every phase-1 through phase-11 component together:

    settings -> state ->
      data provider, broker adapter, risk engine, strategy,
      order manager, database, monitoring ->
        TradingEngine.run_forever()
"""

from __future__ import annotations

import sys
import time
import uuid
from typing import Any

from trading_bot.backtest.engine import BacktestConfig, BacktestEngine
from trading_bot.broker.adapter import AlpacaBrokerAdapter
from trading_bot.config.settings import Settings, load_settings
from trading_bot.data.provider import AlpacaDataProvider
from trading_bot.db.repository import Repository
from trading_bot.engine import TradingEngine
from trading_bot.lifecycle import Lifecycle
from trading_bot.logging_config import configure_logging, get_logger
from trading_bot.monitoring.health import HealthMonitor
from trading_bot.monitoring.notifier import build_notifier
from trading_bot.order.manager import OrderManager
from trading_bot.risk.engine import RiskEngine
from trading_bot.scheduler import Scheduler
from trading_bot.state import AppState
from trading_bot.strategy.signals import MovingAverageCrossStrategy

logger = get_logger(__name__)

# Default scheduler cadence when SLEEPTIME is missing or unparseable.
_DEFAULT_SLEEP_SECONDS = 60.0
_SLEEP_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0}


def _parse_sleep_time(value: Any) -> float:
    """Parse a ``SLEEPTIME`` value such as ``"1M"``/``"30S"`` into seconds.

    A bare number is treated as seconds. Unknown or invalid values fall back
    to one minute so a typo cannot spin the loop at full speed.
    """
    if isinstance(value, (int, float)):
        return max(0.05, float(value))
    text = str(value or "").strip().lower()
    if not text:
        return _DEFAULT_SLEEP_SECONDS
    suffix = text[-1]
    try:
        if suffix in _SLEEP_UNITS:
            return max(0.05, float(text[:-1]) * _SLEEP_UNITS[suffix])
        return max(0.05, float(text))
    except ValueError:
        logger.warning(
            "Invalid SLEEPTIME %r; falling back to %.0fs",
            value, _DEFAULT_SLEEP_SECONDS,
        )
        return _DEFAULT_SLEEP_SECONDS


def _build_components(settings, state: AppState):
    """Instantiate and wire all trading components.

    Returns a dict of component instances ready for injection into
    :class:`TradingEngine`.
    """
    # --- Market data ----------------------------------------------------
    data_provider = AlpacaDataProvider(settings=settings)

    # --- Broker ---------------------------------------------------------
    broker_adapter = AlpacaBrokerAdapter(settings=settings)

    # --- Risk engine ----------------------------------------------------
    risk_engine = RiskEngine(settings=settings, state=state)

    # --- Strategy -------------------------------------------------------
    strategy = MovingAverageCrossStrategy(settings=settings)

    # --- Order manager --------------------------------------------------
    order_manager = OrderManager(broker_adapter=broker_adapter)

    # --- Database / persistence -----------------------------------------
    database = Repository(settings=settings)

    # --- Monitoring / alerting ------------------------------------------
    notifier = build_notifier(settings)
    monitoring = HealthMonitor(notifier=notifier)

    def _data_health() -> dict:
        """Report data health as freshness, not whether the market is open."""
        stale: list[str] = []
        for symbol in settings.watchlist:
            try:
                if not data_provider.is_data_fresh(symbol):
                    stale.append(symbol)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
        return {"ok": not stale, "stale_symbols": stale}

    monitoring.add_component("broker", broker_adapter.is_connected)
    monitoring.add_component("data", _data_health)
    monitoring.add_component("risk", lambda: not risk_engine.is_halted())
    # The risk engine loads its persisted breaker state on construction;
    # mirror that onto the monitor so health is correct before iteration 1.
    if risk_engine.is_halted():
        monitoring.set_circuit_breaker(True, "risk engine halted at startup")

    # Attach to state for cross-component visibility.
    state.data_provider = data_provider
    state.broker_adapter = broker_adapter
    state.risk_engine = risk_engine
    state.strategy = strategy
    state.order_manager = order_manager
    state.database = database
    state.monitoring = monitoring

    return {
        "data_provider": data_provider,
        "broker_adapter": broker_adapter,
        "risk_engine": risk_engine,
        "strategy": strategy,
        "order_manager": order_manager,
        "database": database,
        "monitoring": monitoring,
    }


def main(backtest: bool = False) -> int:
    """Run the paper-trading engine (or a backtest when ``backtest`` is true)."""
    configure_logging()
    settings = load_settings()
    logger.info(
        "Starting trading bot (mode=%s, paper=%s, backtest=%s)",
        settings.trading_mode,
        settings.is_paper(),
        backtest,
    )

    # Validate credentials before constructing any broker/data components so
    # we fail fast with a clear message instead of an Alpaca SDK error.
    placeholder_key = settings.alpaca_api_key in ("", "your_paper_api_key_here")
    placeholder_secret = settings.alpaca_api_secret in (
        "", "your_paper_api_secret_here",
    )
    if placeholder_key or placeholder_secret:
        logger.critical(
            "Alpaca API credentials are missing or still set to placeholder "
            "values in .env. Copy .env.example to .env and fill in your "
            "paper-trading API key and secret from alpaca.markets."
        )
        return 2

    state = AppState()

    if backtest:
        # Backtest mode: run a single historical simulation and exit.
        return _run_backtest(settings, state)

    components = _build_components(settings, state)
    engine = TradingEngine(
        settings=settings,
        state=state,
        data_provider=components["data_provider"],
        broker_adapter=components["broker_adapter"],
        risk_engine=components["risk_engine"],
        strategy=components["strategy"],
        order_manager=components["order_manager"],
        database=components["database"],
        monitoring=components["monitoring"],
    )

    try:
        engine.validate()
    except Exception as exc:  # noqa: BLE001
        logger.critical("Validation failed: %s", exc)
        return 2

    monitoring = components["monitoring"]
    scheduler = Scheduler(
        state,
        interval_seconds=_parse_sleep_time(settings.sleep_time),
    )
    lifecycle = Lifecycle(state)
    lifecycle.on_stop(monitoring.close)
    lifecycle.install_signal_handlers()

    try:
        # Recover fills/positions/orders before the first trading decision.
        engine.reconcile_startup()
        lifecycle.start(f"session-{uuid.uuid4().hex[:8]}")
        scheduler.start(engine.run_iteration)
        while state.is_running() and scheduler.is_running():
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        scheduler.stop()
        try:
            lifecycle.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lifecycle stop reported: %s", exc)
        lifecycle.restore_signal_handlers()
    return 0


def _run_backtest(settings: Settings, state: AppState) -> int:
    """Run a one-off backtest over the configured historical window."""
    data_provider = AlpacaDataProvider(settings=settings)
    strategy = MovingAverageCrossStrategy(settings=settings)
    risk_engine = RiskEngine(settings=settings, state=state)

    bars: dict[str, list[dict]] = {}
    for symbol in settings.watchlist:
        symbol_bars = data_provider.get_historical_bars(
            symbol, settings.indicator_lookback, settings.indicator_timeframe
        )
        if symbol_bars:
            bars[symbol] = symbol_bars

    engine = BacktestEngine(
        config=BacktestConfig(
            initial_capital=100_000.0,
            commission_per_trade=1.0,
            slippage_pct=0.0005,
        ),
        settings=settings,
    )
    result = engine.run(bars, strategy, risk_engine)
    logger.info(
        "Backtest complete: final_equity=%.2f total_return=%.4f "
        "trades=%d sharpe=%.3f max_drawdown=%.4f",
        result.final_equity,
        result.total_return,
        result.metrics.get("trade_count", 0),
        result.metrics.get("sharpe", 0.0),
        result.metrics.get("max_drawdown", 0.0),
    )
    return 0


if __name__ == "__main__":
    backtest = "--backtest" in sys.argv
    sys.exit(main(backtest=backtest))
