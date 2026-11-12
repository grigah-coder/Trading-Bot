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
from trading_bot.logging_config import get_logger
from trading_bot.order.lifecycle import OPEN_STATES
from trading_bot.scheduler import Scheduler
from trading_bot.state import AppState

logger = get_logger(__name__)


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
    def save_fill(
        self, fill: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]: ...
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
        # broker order id -> cumulative filled qty already processed
        self._processed_fills: Dict[str, float] = {}
        self._halt_reported = False

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
            "exits": 0,
            "failed": 0,
            "entries_blocked": 0,
            "ignored_sells": 0,
            "errors": [],
        }

        try:
            if not self.state.is_running():
                summary["errors"].append("application not running")
                return self._finish(summary)

            # Fail closed if risk/account/market data is uncertain.
            if self.risk_engine is not None and self.risk_engine.is_halted():
                reason = "circuit breaker halted; no new trades"
                summary["errors"].append(reason)
                self.state.set_data_stale(True)
                if not self._halt_reported:
                    self._halt_reported = True
                    self._save_risk_event({
                        "event_type": "halted",
                        "reason": reason,
                    })
                    self._alert(f"circuit breaker tripped: {reason}", level="critical")
                # Keep the health monitor's breaker in sync with the risk
                # engine (quietly; the alert above already notified).
                self._sync_monitoring_breaker(True, reason)
                return self._finish(summary)
            if self._halt_reported:
                self._halt_reported = False
                self._sync_monitoring_breaker(False)

            # 1. Market data
            bars: Dict[str, List[Dict[str, Any]]] = {}
            if self.data_provider is not None:
                data_stale = False
                for symbol in self.settings.watchlist:
                    try:
                        symbol_bars = self.data_provider.get_historical_bars(
                            symbol, self.settings.indicator_lookback,
                            self.settings.indicator_timeframe,
                        )
                        if symbol_bars is None or not self.data_provider.is_data_fresh(symbol):
                            summary["errors"].append(f"stale/missing data for {symbol}")
                            data_stale = True
                            continue
                        bars[symbol] = symbol_bars
                    except Exception as exc:  # noqa: BLE001
                        summary["errors"].append(f"data error for {symbol}: {exc}")
                        data_stale = True
                # Stale only when at least one symbol failed this pass.
                self.state.set_data_stale(data_stale)
            else:
                summary["errors"].append("no data provider")
                self.state.set_data_stale(True)
                return self._finish(summary)

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
            # Include symbols with open orders so concurrency/already-held
            # checks cannot be bypassed while an entry is still working.
            assess_positions = self._assessment_positions(positions, summary)
            closed_symbols: set[str] = set()

            approved: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
            entries_allowed = self._new_entries_allowed()
            if not entries_allowed:
                logger.info(
                    "New entries suppressed: market closed or opening blackout"
                )
            for signal in signals:
                side = str(signal.get("side", "buy")).strip().lower()
                if side == "sell":
                    # Long-only: a sell is an exit, never a new short entry.
                    # Exits are allowed even when entries are filtered out.
                    self._handle_sell(
                        signal, positions, assess_positions, closed_symbols,
                        summary,
                    )
                    continue
                if not entries_allowed:
                    summary["entries_blocked"] += 1
                    logger.debug(
                        "Blocked new entry for %s outside the entry window",
                        signal.get("symbol"),
                    )
                    continue
                try:
                    decision = self.risk_engine.assess(
                        signal, account, assess_positions,
                    )
                except RiskDataUnavailableError as exc:
                    summary["errors"].append(f"risk data unavailable: {exc}")
                    self.state.set_data_stale(True)
                    break  # fail closed — stop processing signals
                except Exception as exc:  # noqa: BLE001
                    summary["errors"].append(f"risk assessment error: {exc}")
                    logger.error("Risk assessment failed: %s", exc, exc_info=True)
                    continue
                if decision.get("approved"):
                    approved.append((signal, decision))
                    self._append_pending_position(
                        assess_positions, signal, decision,
                    )
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
                    self._save_risk_event({
                        "event_type": "rejected",
                        "symbol": signal.get("symbol"),
                        "side": side,
                        "quantity": signal.get("quantity"),
                        "price": signal.get("price"),
                        "reason": (
                            decision.get("rejection_reason")
                            or decision.get("reason")
                            or "rejected"
                        ),
                        "metadata": {"rules": decision.get("rules", [])},
                    })

            # 5-7. Order management -> broker execution -> fills
            for signal, decision in approved:
                order = self._build_order(signal, decision)
                try:
                    submitted = self._submit_order(order, summary)
                except Exception as exc:  # noqa: BLE001
                    summary["failed"] += 1
                    summary["errors"].append(f"order submission failed: {exc}")
                    logger.error("Order submission failed: %s", exc, exc_info=True)
                    continue
                if submitted is None:
                    summary["failed"] += 1
                    summary["errors"].append(
                        f"order submission failed for {order.get('symbol')}"
                    )
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
            self._alert(
                f"trading iteration crashed: {exc}", level="critical",
            )

        return self._finish(summary)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def reconcile_startup(self) -> None:
        """Reconcile broker positions/orders before the first iteration.

        Recovers fills and order state that changed while the bot was down,
        so the first trading decision is based on current broker truth.
        """
        if self.order_manager is None or self.broker_adapter is None:
            return
        summary: Dict[str, Any] = {"errors": []}
        try:
            self._reconcile(summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Startup reconciliation failed: %s", exc)
        for message in summary.get("errors", ()):
            logger.warning("Startup reconciliation: %s", message)

    def _finish(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Record iteration errors on shared state and return the summary."""
        recorder = getattr(self.state, "record_error", None)
        if callable(recorder):
            for message in summary.get("errors", ()):
                try:
                    recorder(message)
                except Exception:  # noqa: BLE001
                    pass
        return summary

    def _alert(self, message: str, level: str = "error") -> None:
        """Forward an alert to monitoring; never let it break the loop."""
        if self.monitoring is None:
            return
        try:
            self.monitoring.alert(message, level=level)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Monitoring alert failed: %s", exc)

    def _sync_monitoring_breaker(
        self, tripped: bool, reason: str = "",
    ) -> None:
        """Mirror the risk engine's breaker onto the health monitor."""
        if self.monitoring is None:
            return
        sync = getattr(self.monitoring, "set_circuit_breaker", None)
        if not callable(sync):
            return
        try:
            sync(tripped, reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Monitoring breaker sync failed: %s", exc)

    def _new_entries_allowed(self) -> bool:
        """Return True when the session/opening rules permit new entries.

        Exits and protection are never gated by this method. When the time
        filter is disabled, only the opening blackout remains in force.
        """
        provider = self.data_provider
        if provider is None:
            return True
        if bool(getattr(self.settings, "time_filter_enabled", True)):
            open_check = getattr(provider, "is_market_open", None)
            if callable(open_check):
                try:
                    if not bool(open_check()):
                        return False
                except Exception as exc:  # noqa: BLE001 - fail closed
                    logger.warning("Market-open check failed: %s", exc)
                    return False
        blackout_check = getattr(provider, "is_in_opening_blackout", None)
        if callable(blackout_check):
            try:
                if bool(blackout_check()):
                    return False
            except Exception as exc:  # noqa: BLE001 - fail closed
                logger.warning("Opening-blackout check failed: %s", exc)
                return False
        return True

    def _assessment_positions(
        self, positions: List[Dict[str, Any]], summary: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Positions plus synthetic entries for symbols with working orders."""
        result = [dict(position) for position in positions]
        seen = {
            str(position.get("symbol", "")).upper()
            for position in result
        }
        for symbol in self._open_order_symbols(summary):
            if symbol in seen:
                continue
            seen.add(symbol)
            result.append({
                "symbol": symbol,
                "qty": 1.0,
                "side": "long",
                "pending": True,
            })
        return result

    def _open_order_symbols(self, summary: Dict[str, Any]) -> set[str]:
        """Symbols that currently have a working entry order."""
        symbols: set[str] = set()
        if self.order_manager is not None:
            getter = getattr(self.order_manager, "get_active_orders", None)
            if callable(getter):
                try:
                    for order in getter() or ():
                        symbol = str(order.get("symbol", "")).upper()
                        if symbol:
                            symbols.add(symbol)
                except Exception as exc:  # noqa: BLE001
                    summary["errors"].append(
                        f"active order fetch failed: {exc}"
                    )
        if self.broker_adapter is not None:
            getter = getattr(self.broker_adapter, "get_orders", None)
            if callable(getter):
                try:
                    for order in getter() or ():
                        status = str(order.get("status", "")).lower()
                        if not (
                            order.get("is_open")
                            or status in OPEN_STATES
                        ):
                            continue
                        symbol = str(order.get("symbol", "")).upper()
                        if symbol:
                            symbols.add(symbol)
                except Exception as exc:  # noqa: BLE001
                    summary["errors"].append(
                        f"open order fetch failed: {exc}"
                    )
        return symbols

    @staticmethod
    def _append_pending_position(
        positions: List[Dict[str, Any]],
        signal: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> None:
        """Reserve a slot immediately so later signals see this approval."""
        symbol = str(signal.get("symbol", "")).upper()
        if not symbol:
            return
        if any(
            str(position.get("symbol", "")).upper() == symbol
            for position in positions
        ):
            return
        try:
            qty = abs(float(decision.get("qty") or 0.0))
        except (TypeError, ValueError):
            qty = 0.0
        positions.append({
            "symbol": symbol,
            "qty": qty if qty > 0 else 1.0,
            "side": "long",
            "pending": True,
        })

    def _handle_sell(
        self,
        signal: Dict[str, Any],
        positions: List[Dict[str, Any]],
        assess_positions: List[Dict[str, Any]],
        closed_symbols: set[str],
        summary: Dict[str, Any],
    ) -> None:
        """Long-only exit path: flatten a held symbol, ignore sells when flat."""
        symbol = str(signal.get("symbol", "")).upper()
        if not symbol or symbol in closed_symbols:
            summary["ignored_sells"] += 1
            logger.info("Ignoring sell signal for %s (no open position)", symbol)
            return
        position = next(
            (
                p for p in positions
                if str(p.get("symbol", "")).upper() == symbol
                and float(p.get("qty") or 0) != 0
            ),
            None,
        )
        if position is None:
            summary["ignored_sells"] += 1
            logger.info("Ignoring sell signal for flat symbol %s", symbol)
            return

        if self.order_manager is not None:
            cancel_legs = getattr(
                self.order_manager, "cancel_bracket_legs", None,
            )
            if callable(cancel_legs):
                try:
                    cancel_legs(symbol)
                except Exception as exc:  # noqa: BLE001
                    summary["errors"].append(
                        f"failed to cancel bracket legs for {symbol}: {exc}"
                    )
                    logger.error(
                        "Failed to cancel bracket legs for %s: %s",
                        symbol, exc, exc_info=True,
                    )
                    return

        qty = abs(float(position.get("qty") or 0))
        if qty <= 0:
            summary["ignored_sells"] += 1
            logger.info("Ignoring sell signal for flat symbol %s", symbol)
            return

        order = {
            "order_id": str(uuid.uuid4()),
            "symbol": symbol,
            "side": "sell",
            "quantity": qty,
            "order_type": "market",
            "signal_id": signal.get("signal_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            submitted = self._submit_order(order, summary)
        except Exception as exc:  # noqa: BLE001
            summary["failed"] += 1
            summary["errors"].append(f"exit order failed for {symbol}: {exc}")
            logger.error("Exit order failed for %s: %s", symbol, exc, exc_info=True)
            return
        if submitted is None:
            summary["failed"] += 1
            summary["errors"].append(f"exit order submission failed for {symbol}")
            return

        summary["orders"] += 1
        summary["exits"] += 1
        self.state.increment("orders_submitted")
        closed_symbols.add(symbol)
        assess_positions[:] = [
            p for p in assess_positions
            if str(p.get("symbol", "")).upper() != symbol
        ]
        if self.database is not None:
            try:
                self.database.save_order(submitted)
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to save exit order: %s", exc)
        logger.info("Closed long position %s (qty=%.4f)", symbol, qty)

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

    def _build_order(
        self, signal: Dict[str, Any], decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the broker order from the risk decision, not the raw signal."""
        return {
            "order_id": str(uuid.uuid4()),
            "symbol": signal.get("symbol"),
            "side": signal.get("side", "buy"),
            "quantity": decision.get("qty"),
            "order_type": "market",
            "stop_price": decision.get("stop_price"),
            "limit_price": decision.get("limit_price"),
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

    @staticmethod
    def _iter_broker_orders(
        orders: List[Dict[str, Any]],
    ):
        """Yield broker orders and their nested bracket legs."""
        for order in orders or ():
            if not isinstance(order, dict):
                continue
            yield order
            for leg in order.get("legs") or ():
                if isinstance(leg, dict):
                    yield leg

    def _reconcile(self, summary: Dict[str, Any]) -> None:
        if self.order_manager is None or self.broker_adapter is None:
            return
        try:
            positions = self.broker_adapter.get_positions() or []
            orders = self.broker_adapter.get_orders() or []

            # Detect new fills/terminal states before the manager reconciles
            # so partial-fill deltas are applied exactly once.
            self._process_fills(orders, summary)
            self.order_manager.reconcile(positions, orders)

            if self.database is not None:
                for order in self._iter_broker_orders(orders):
                    if not order.get("order_id"):
                        continue
                    try:
                        self.database.save_order(order)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("failed to save order state: %s", exc)
                for position in positions:
                    try:
                        self.database.save_position(position)
                    except Exception:  # noqa: BLE001
                        pass
                pruner = getattr(self.database, "prune_positions", None)
                if callable(pruner):
                    try:
                        pruner({
                            str(p.get("symbol", "")).upper()
                            for p in positions
                            if p.get("symbol")
                        })
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("failed to prune positions: %s", exc)

            # A realized loss can trip the breaker below; persist the event.
            if (
                self.risk_engine is not None
                and self.risk_engine.is_halted()
                and not self._halt_reported
            ):
                self._halt_reported = True
                self._save_risk_event({
                    "event_type": "halted",
                    "reason": "daily loss limit reached",
                })
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"reconciliation failed: {exc}")
            logger.warning("Reconciliation failed: %s", exc)

    def _process_fills(
        self, orders: List[Dict[str, Any]], summary: Dict[str, Any],
    ) -> None:
        """Record newly-filled quantities (including bracket exit legs)."""
        for order in self._iter_broker_orders(orders):
            order_id = str(order.get("order_id") or "")
            if not order_id:
                continue
            try:
                filled = float(order.get("filled_qty") or 0.0)
            except (TypeError, ValueError):
                continue
            if filled <= 0:
                continue
            already = self._processed_fills.get(order_id, 0.0)
            if filled <= already + 1e-9:
                continue
            try:
                price = float(order.get("filled_avg_price") or 0.0)
            except (TypeError, ValueError):
                price = 0.0
            symbol = str(order.get("symbol", "")).upper()
            side = str(order.get("side", "")).strip().lower()
            if not symbol or side not in {"buy", "sell"} or price <= 0:
                continue

            delta = filled - already
            self._processed_fills[order_id] = filled
            fill = {
                # Deterministic per cumulative fill so restarts and repeated
                # reconciles are idempotent in the DB and the trade ledger.
                "fill_id": f"{order_id}:{filled:g}",
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": delta,
                "price": price,
                "filled_at": (
                    order.get("filled_at")
                    or datetime.now(timezone.utc).isoformat()
                ),
            }

            # Persist the parent/leg order first: fills have a FK to orders.
            if self.database is not None:
                try:
                    self.database.save_order(order)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("failed to save filled order: %s", exc)

            if self.order_manager is not None:
                try:
                    self.order_manager.on_fill({
                        "order_id": order_id,
                        "client_order_id": order.get("client_order_id"),
                        "symbol": symbol,
                        "side": side,
                        "filled_qty": delta,
                        "fill_price": price,
                        "status": order.get("status"),
                    })
                except Exception as exc:  # noqa: BLE001
                    summary["errors"].append(f"fill handling failed: {exc}")
                    logger.warning("Fill handling failed for %s: %s", order_id, exc)

            ledger_result = None
            if self.database is not None:
                try:
                    ledger_result = self.database.save_fill(fill)
                except Exception as exc:  # noqa: BLE001
                    summary["errors"].append(f"fill persistence failed: {exc}")
                    logger.warning("Fill persistence failed for %s: %s", order_id, exc)

            self.state.increment("orders_filled")
            self._apply_realized_pl(fill, ledger_result)

    def _apply_realized_pl(
        self, fill: Dict[str, Any], ledger_result: Any,
    ) -> None:
        """Feed realized sell losses into the risk engine's daily counter."""
        if str(fill.get("side", "")).lower() != "sell":
            return
        if not isinstance(ledger_result, dict):
            return
        realized = ledger_result.get("realized_pl")
        if realized is None:
            return
        try:
            realized = float(realized)
        except (TypeError, ValueError):
            return
        if realized >= 0:
            return
        risk_engine = self.risk_engine
        recorder = getattr(risk_engine, "record_loss", None)
        if not callable(recorder):
            return
        try:
            recorder(abs(realized))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to record realized loss: %s", exc)

    def _save_risk_event(self, event: Dict[str, Any]) -> None:
        if self.database is None:
            return
        try:
            self.database.save_risk_event({
                **event,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to save risk event: %s", exc)

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
