"""Central trading engine.

Wires the data provider, broker adapter, risk engine, strategy, order
manager, database and monitoring together and runs the trading loop:

    market data → strategy → signal validation → risk approval →
    portfolio validation → order management → broker execution →
    fill handling → position management → database → monitoring

Components are injected through the constructor (dependency injection). The
engine owns no trading logic itself — it orchestrates. The risk engine is the
sole gatekeeper: no order reaches the broker without its approval.

Implemented in Phase 2 (Core Architecture). Concrete components are supplied
in Phases 3–8.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from trading_bot.config.settings import Settings
from trading_bot.exceptions import (
    ConfigurationError,
    RiskDataUnavailableError,
)
from trading_bot.logging_config import configure_logging
from trading_bot.scheduler import Scheduler
from trading_bot.state import AppState

logger = configure_logging(__name__).getChild("engine")


# ---------------------------------------------------------------------------
# Component interfaces (protocols)
# ---------------------------------------------------------------------------
@runtime_checkable
class DataProvider(Protocol):
    """Market data provider protocol."""

    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]: ...
    def get_historical_bars(
        self, symbol: str, limit: int, timeframe: str
    ) -> Optional[List[Dict[str, Any]]]: ...
    def is_market_open(self) -> bool: ...
    def is_data_fresh(self, symbol: str) -> bool: ...


@runtime_checkable
class BrokerAdapter(Protocol):
    """Broker adapter protocol (paper trading only)."""

    def get_account(self) -> Optional[Dict[str, Any]]: ...
    def get_positions(self) -> List[Dict[str, Any]]: ...
    def get_orders(self) -> List[Dict[str, Any]]: ...
    def submit_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def is_connected(self) -> bool: ...


@runtime_checkable
class RiskEngine(Protocol):
    """Independent risk engine protocol — approves or rejects every trade."""

    def assess(
        self, signal: Dict[str, Any], account: Dict[str, Any],
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]: ...
    def is_halted(self) -> bool: ...


@runtime_checkable
class Strategy(Protocol):
    """Strategy protocol — generates signals from market data."""

    def generate_signals(
        self, bars: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]: ...


@runtime_checkable
class OrderManager(Protocol):
    """Order manager protocol — submission, fills, stops, lifecycle."""

    def submit(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]: ...
    def on_fill(self, fill: Dict[str, Any]) -> None: ...
    def reconcile(self, positions: List[Dict[str, Any]], orders: List[Dict[str, Any]]) -> None: ...


@runtime_checkable
class Database(Protocol):
    """Persistence protocol."""

    def save_order(self, order: Dict[str, Any]) -> None: ...
    def save_fill(self, fill: Dict[str, Any]) -> None: ...
    def save_position(self, position: Dict[str, Any]) -> None: ...
    def save_signal(self, signal: Dict[str, Any]) -> None: ...
    def save_risk_event(self, event: Dict[str, Any]) -> None: ...
    def save_account_snapshot(self, snapshot: Dict[str, Any]) -> None: ...
    def save_system_event(self, event: Dict[str, Any]) -> None: ...


@runtime_checkable
class Monitoring(Protocol):
    """Monitoring / alerting protocol."""

    def health_check(self) -> Dict[str, Any]: ...
    def alert(self, message: str, level: str = "error") -> None: ...
    def record_metric(self, name: str, value: float) -> None: ...


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class TradingEngine:
    """Central trading engine.

    Parameters are injected so the engine can be tested with mock components
    and so no component is a global singleton.
    """

    def __init__(
        self,
        settings: Settings,
        state: AppState,
        data_provider: Optional[DataProvider] = None,
        broker_adapter: Optional[BrokerAdapter] = None,
        risk_engine: Optional[RiskEngine] = None,
        strategy: Optional[Strategy] = None,
        order_manager: Optional[OrderManager] = None,
        database: Optional[Database] = None,
        monitoring: Optional[Monitoring] = None,
    ) -> None:
        self.settings = settings
        self.state = state
        self.data_provider = data_provider
        self.broker_adapter = broker_adapter
        self.risk_engine = risk_engine
        self.strategy = strategy
        self.order_manager = order_manager
        self.database = database
        self.monitoring = monitoring

        self._iteration = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Initialization / validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Validate that required components and configuration are present.

        Fails closed: raises :class:`ConfigurationError` if the system cannot
        operate safely.
        """
        if not self.settings.is_paper():
            raise ConfigurationError(
                "system is not configured for paper trading; refusing to start"
            )
        if not self.settings.alpaca_api_key or not self.settings.alpaca_api_secret:
            raise ConfigurationError(
                "ALPACA_API_KEY / ALPACA_API_SECRET are not set; refusing to start"
            )
        # Broker and data provider are required for live/paper operation.
        # Risk engine is mandatory — trades cannot proceed without it.
        if self.risk_engine is None:
            raise ConfigurationError(
                "risk engine is required; trades cannot proceed without it"
            )
        logger.info(
            "Engine validated (paper=%s, watchlist=%d symbols)",
            self.settings.is_paper(),
            len(self.settings.watchlist),
        )

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------
    def run_iteration(self) -> Dict[str, Any]:
        """Execute one trading iteration.

        Returns a summary dict describing what happened. Never raises —
        failures are caught, logged, recorded as system events, and the
        iteration continues with a safe (no-trade) outcome.
        """
        self._iteration += 1
        summary: Dict[str, Any] = {
            "iteration": self._iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": 0,
            "approved": 0,
            "rejected": 0,
            "orders": 0,
            "errors": [],
        }

        try:
            if not self.state.is_running():
                summary["errors"].append("application not running")
                return summary

            # Fail closed if risk/account/market data is uncertain.
            if self.risk_engine is not None and self.risk_engine.is_halted():
                summary["errors"].append("circuit breaker halted; no new trades")
                self.state.set_data_stale(True)
                return summary

            # 1. Market data
            bars: Dict[str, List[Dict[str, Any]]] = {}
            if self.data_provider is not None:
                for symbol in self.settings.watchlist:
                    try:
                        symbol_bars = self.data_provider.get_historical_bars(
                            symbol, self.settings.indicator_lookback,
                            self.settings.indicator_timeframe,
                        )
                        if symbol_bars is None or not self.data_provider.is_data_fresh(symbol):
                            summary["errors"].append(f"stale/missing data for {symbol}")
                            self.state.set_data_stale(True)
                            continue
                        bars[symbol] = symbol_bars
                    except Exception as exc:  # noqa: BLE001
                        summary["errors"].append(f"data error for {symbol}: {exc}")
                        self.state.set_data_stale(True)
                self.state.set_data_stale(False)
            else:
                summary["errors"].append("no data provider")
                self.state.set_data_stale(True)
                return summary

            # 2. Strategy -> signals
            signals: List[Dict[str, Any]] = []
            if self.strategy is not None:
                try:
                    signals = self.strategy.generate_signals(bars) or []
                except Exception as exc:  # noqa: BLE001
                    summary["errors"].append(f"strategy error: {exc}")
                    logger.error("Strategy failed: %s", exc, exc_info=True)
            summary["signals"] = len(signals)
            self.state.increment("signals_generated", len(signals))

            # 3. Risk approval (gatekeeper) + 4. portfolio validation
            account = self._fetch_account(summary)
            positions = self._fetch_positions(summary)

            approved: List[Dict[str, Any]] = []
            for signal in signals:
                try:
                    decision = self.risk_engine.assess(signal, account, positions)
                except RiskDataUnavailableError as exc:
                    summary["errors"].append(f"risk data unavailable: {exc}")
                    self.state.set_data_stale(True)
                    break  # fail closed — stop processing signals
                except Exception as exc:  # noqa: BLE001
                    summary["errors"].append(f"risk assessment error: {exc}")
                    logger.error("Risk assessment failed: %s", exc, exc_info=True)
                    continue
                if decision.get("approved"):
                    approved.append(signal)
                    summary["approved"] += 1
                    self.state.increment("signals_approved")
                    if self.database is not None:
                        try:
                            self.database.save_signal(signal)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("failed to save signal: %s", exc)
                else:
                    summary["rejected"] += 1
                    self.state.increment("signals_rejected")

            # 5-7. Order management -> broker execution -> fills
            for signal in approved:
                order = self._build_order(signal)
                try:
                    submitted = self._submit_order(order, summary)
                except Exception as exc:  # noqa: BLE001
                    summary["errors"].append(f"order submission failed: {exc}")
                    logger.error("Order submission failed: %s", exc, exc_info=True)
                    continue
                if submitted is None:
                    summary["rejected"] += 1
                    continue
                summary["orders"] += 1
                self.state.increment("orders_submitted")
                if self.database is not None:
                    try:
                        self.database.save_order(submitted)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("failed to save order: %s", exc)

            # 8-10. Reconciliation + persistence + monitoring
            self._reconcile(summary)
            self._emit_metrics(summary)

        except Exception as exc:  # noqa: BLE001 - never crash the loop
            summary["errors"].append(f"unexpected: {exc}")
            logger.critical("Trading iteration crashed: %s", exc, exc_info=True)
            self._record_system_event("iteration_crash", str(exc))

        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fetch_account(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        if self.broker_adapter is None:
            return {}
        try:
            account = self.broker_adapter.get_account()
            return account or {}
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"account fetch failed: {exc}")
            self.state.set_data_stale(True)
            return {}

    def _fetch_positions(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.broker_adapter is None:
            return []
        try:
            return self.broker_adapter.get_positions() or []
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"position fetch failed: {exc}")
            self.state.set_data_stale(True)
            return []

    def _build_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "order_id": str(uuid.uuid4()),
            "symbol": signal.get("symbol"),
            "side": signal.get("side", "buy"),
            "quantity": signal.get("quantity", 0),
            "order_type": "market",
            "signal_id": signal.get("signal_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _submit_order(
        self, order: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if self.order_manager is not None:
            return self.order_manager.submit(order)
        if self.broker_adapter is not None:
            return self.broker_adapter.submit_order(order)
        return None

    def _reconcile(self, summary: Dict[str, Any]) -> None:
        if self.order_manager is None or self.broker_adapter is None:
            return
        try:
            positions = self.broker_adapter.get_positions() or []
            orders = self.broker_adapter.get_orders() or []
            self.order_manager.reconcile(positions, orders)
            if self.database is not None:
                for position in positions:
                    try:
                        self.database.save_position(position)
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"reconciliation failed: {exc}")
            logger.warning("Reconciliation failed: %s", exc)

    def _emit_metrics(self, summary: Dict[str, Any]) -> None:
        if self.monitoring is None:
            return
        try:
            self.monitoring.record_metric("signals_generated", summary["signals"])
            self.monitoring.record_metric("signals_approved", summary["approved"])
            self.monitoring.record_metric("signals_rejected", summary["rejected"])
            self.monitoring.record_metric("orders_submitted", summary["orders"])
            self.monitoring.health_check()
            self.state.touch_heartbeat()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Monitoring metric emission failed: %s", exc)

    def _record_system_event(self, event_type: str, detail: str) -> None:
        if self.database is None:
            return
        try:
            self.database.save_system_event({
                "type": event_type,
                "detail": detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Run loop entry point
    # ------------------------------------------------------------------
    def run_forever(self, callback_interval_seconds: float = 60.0) -> None:
        """Block running iterations until ``state`` is stopped."""
        self.validate()
        self.state.start(f"session-{uuid.uuid4().hex[:8]}")
        logger.info("Trading engine running (interval=%.1fs)", callback_interval_seconds)

        scheduler = Scheduler(
            self.state,
            interval_seconds=callback_interval_seconds,
        )
        try:
            scheduler.start(self.run_iteration)
            # Keep the main thread alive while the scheduler runs.
            while self.state.is_running() and scheduler.is_running():
                threading.Event().wait(1.0)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            scheduler.stop()
            self.state.stop()
            logger.info("Trading engine stopped")
