"""Backtesting engine with realistic execution simulation.

Runs a strategy over historical bars bar-by-bar, applying the risk engine as
the sole gatekeeper, simulating order execution with slippage/spread/commission
and stop-loss / take-profit fills, and producing an equity curve plus
performance metrics.

Designed to prevent look-ahead bias: at each bar the strategy only sees bars
strictly before the current bar, and fills occur at the current bar's price
with slippage applied conservatively (buys filled higher, sells lower).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trading_bot.backtest.metrics import summary_metrics
from trading_bot.config.settings import Settings
from trading_bot.logging_config import configure_logging

logger = configure_logging(__name__).getChild("backtest")


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    initial_capital: float = 100_000.0
    commission_per_trade: float = 0.0          # flat fee per fill
    commission_pct: float = 0.0                # fraction of notional per fill
    slippage_pct: float = 0.0005               # 0.05% conservative slippage
    spread_pct: float = 0.0002                 # 0.02% half-spread
    periods_per_year: float = 252.0            # bars per year for annualised metrics
    max_concurrent_positions: Optional[int] = None  # override settings
    stop_loss_pct: Optional[float] = None      # override signal stop
    take_profit_pct: Optional[float] = None    # override signal take-profit


@dataclass
class BacktestTrade:
    """A completed (round-trip) trade."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    realised_pl: float
    entry_index: int
    exit_index: int
    entry_bar: Dict[str, Any] = field(default_factory=dict)
    exit_bar: Dict[str, Any] = field(default_factory=dict)

    @property
    def pnl(self) -> float:
        return self.realised_pl

    @property
    def is_win(self) -> bool:
        return self.realised_pl > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "realized_pl": self.realised_pl,
            "pnl": self.realised_pl,
            "entry_index": self.entry_index,
            "exit_index": self.exit_index,
        }


@dataclass
class BacktestResult:
    """Outcome of a backtest run."""

    config: BacktestConfig
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)  # per bar
    trades: List[BacktestTrade] = field(default_factory=list)
    signals_generated: int = 0
    signals_approved: int = 0
    signals_rejected: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return (
            self.equity_curve[-1]["equity"]
            if self.equity_curve
            else self.config.initial_capital
        )

    @property
    def total_return(self) -> float:
        if self.config.initial_capital <= 0:
            return 0.0
        return self.final_equity / self.config.initial_capital - 1.0


class BacktestEngine:
    """Simulates the trading loop over historical bars.

    Parameters
    ----------
    config:
        Execution parameters (capital, costs, slippage).
    settings:
        Optional :class:`Settings` for risk-control defaults.
    clock:
        Optional callable returning the current time (unused by backtest but
        kept for interface symmetry).

    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        settings: Optional[Settings] = None,
        clock=None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.settings = settings
        self._clock = clock

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(
        self,
        bars: Dict[str, List[Dict[str, Any]]],
        strategy: Any,
        risk_engine: Any,
    ) -> BacktestResult:
        """Run the backtest.

        ``bars`` maps symbol -> list of validated bar dicts (oldest-first).
        Each bar must have ``timestamp``, ``open``, ``high``, ``low``, ``close``,
        ``volume``.

        Returns a :class:`BacktestResult` with the equity curve, trades and
        metrics. Never raises — failures are recorded as errors.
        """
        config = self.config
        result = BacktestResult(config=config)

        # Normalise and sort bars per symbol (oldest-first).
        symbol_bars: Dict[str, List[Dict[str, Any]]] = {}
        for symbol, symbol_bar_list in (bars or {}).items():
            valid = [
                b for b in (symbol_bar_list or [])
                if self._valid_bar(b)
            ]
            valid.sort(key=lambda b: str(b.get("timestamp", "")))
            if valid:
                symbol_bars[str(symbol).upper()] = valid

        if not symbol_bars:
            logger.warning("Backtest received no usable bars")
            result.metrics = summary_metrics([], [], config.periods_per_year)
            return result

        # Common time axis: iterate by index. At index i, history = bars[:i]
        # (no look-ahead) and current = bars[i].
        max_len = max(len(v) for v in symbol_bars.values())

        # Portfolio state
        cash = config.initial_capital
        positions: Dict[str, Dict[str, Any]] = {}
        open_orders: List[Dict[str, Any]] = []
        order_counter = 0

        equity_curve: List[Dict[str, Any]] = []

        for i in range(max_len):
            # History strictly before current bar (no look-ahead).
            history: Dict[str, List[Dict[str, Any]]] = {}
            current: Dict[str, Dict[str, Any]] = {}
            for symbol, sb in symbol_bars.items():
                history[symbol] = sb[:i]
                if i < len(sb):
                    current[symbol] = sb[i]

            # 1. Strategy -> signals (only history up to previous bar).
            signals: List[Dict[str, Any]] = []
            if strategy is not None:
                try:
                    signals = strategy.generate_signals(history) or []
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Strategy failed at bar %d: %s", i, exc)
            result.signals_generated += len(signals)

            # 2. Risk approval.
            account = {
                "equity": cash + self._position_value(positions, current),
                "cash": cash,
                "status": "active",
                "account_blocked": False,
                "trading_blocked": False,
            }
            pos_list = list(positions.values())

            approved: List[Dict[str, Any]] = []
            for signal in signals:
                try:
                    decision = risk_engine.assess(signal, account, pos_list)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Risk assessment failed: %s", exc)
                    result.signals_rejected += 1
                    continue
                if decision.get("approved"):
                    approved.append(signal)
                    result.signals_approved += 1
                else:
                    result.signals_rejected += 1

            # 3. Submit approved orders (one entry per symbol).
            for signal in approved:
                symbol = str(signal.get("symbol", "")).upper()
                if symbol in positions:
                    continue  # already have a position in this symbol
                if config.max_concurrent_positions is not None:
                    if len(positions) >= config.max_concurrent_positions:
                        break
                order_counter += 1
                qty = float(
                    decision.get("qty")
                    or signal.get("quantity")
                    or 0
                )
                order = {
                    "order_id": f"bt-{order_counter}",
                    "symbol": symbol,
                    "side": str(signal.get("side", "buy")).lower(),
                    "quantity": qty,
                    "order_type": "market",
                    "stop_price": decision.get("stop_price"),
                    "limit_price": decision.get("limit_price"),
                    "signal_id": signal.get("signal_id"),
                    "status": "pending",
                }
                open_orders.append(order)
                result.orders_submitted += 1

            # 4. Check stop-loss / take-profit on open positions using current
            #    bar's high / low (conservative: worst price within the bar).
            closed: List[Dict[str, Any]] = []
            for symbol in list(positions):
                pos = positions[symbol]
                bar = current.get(symbol)
                if bar is None:
                    continue
                high = float(bar.get("high", bar.get("close")))
                low = float(bar.get("low", bar.get("close")))
                exit_price: Optional[float] = None
                exit_reason = ""
                if pos["side"] == "buy":
                    stop = pos.get("stop_price")
                    if stop is not None and low <= stop:
                        exit_price = min(stop, low)
                        exit_reason = "stop_loss"
                    tp = pos.get("limit_price")
                    if exit_price is None and tp is not None and high >= tp:
                        exit_price = max(tp, high)
                        exit_reason = "take_profit"
                else:  # sell
                    stop = pos.get("stop_price")
                    if stop is not None and high >= stop:
                        exit_price = max(stop, high)
                        exit_reason = "stop_loss"
                    tp = pos.get("limit_price")
                    if exit_price is None and tp is not None and low <= tp:
                        exit_price = min(tp, low)
                        exit_reason = "take_profit"
                if exit_price is not None:
                    closed.append((symbol, exit_price, exit_reason))

            for symbol, exit_price, _reason in closed:
                pos = positions.pop(symbol)
                pl = self._realised_pl(pos, exit_price)
                commission = self.commission_for(exit_price, pos["quantity"])
                net_pl = pl - commission
                result.trades.append(
                    BacktestTrade(
                        symbol=symbol,
                        side=pos["side"],
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        quantity=pos["quantity"],
                        realised_pl=net_pl,
                        entry_index=pos["entry_index"],
                        exit_index=i,
                        entry_bar=pos["entry_bar"],
                        exit_bar=dict(bar),
                    )
                )
                result.orders_filled += 1
                cash += pos["quantity"] * exit_price - commission
                logger.info(
                    "Backtest %s %s exit @ %.4f pl=%.2f",
                    symbol, pos["side"], exit_price, net_pl,
                )

            # 5. Fill new orders at current bar's close with slippage.
            still_open: List[Dict[str, Any]] = []
            for order in open_orders:
                symbol = order["symbol"]
                bar = current.get(symbol)
                if bar is None:
                    still_open.append(order)
                    continue
                close = float(bar.get("close", 0))
                if close <= 0:
                    still_open.append(order)
                    continue
                # Conservative slippage: buys fill higher, sells lower.
                side = order["side"]
                slip = config.slippage_pct
                fill_price = close * (1.0 + slip) if side == "buy" else close * (1.0 - slip)
                # Half-spread applied on top.
                fill_price *= (
                    1.0 + config.spread_pct
                    if side == "buy"
                    else 1.0 - config.spread_pct
                )
                qty = order["quantity"]
                notional = qty * fill_price
                commission = self.commission_for(fill_price, qty)
                if cash + 1e-9 < notional + commission:
                    # Not enough cash: keep order open, retry next bar.
                    still_open.append(order)
                    continue
                positions[symbol] = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": qty,
                    "entry_price": fill_price,
                    "stop_price": order.get("stop_price"),
                    "limit_price": order.get("limit_price"),
                    "entry_index": i,
                    "entry_bar": dict(bar),
                    "entry_time": bar.get("timestamp"),
                }
                cash -= notional + commission
                result.orders_filled += 1
                logger.info(
                    "Backtest %s %s entry @ %.4f qty=%.4f comm=%.2f",
                    symbol, side, fill_price, qty, commission,
                )

            open_orders = still_open

            # 6. Mark-to-market equity at current bar's close.
            port_value = self._position_value(positions, current)
            equity = cash + port_value
            equity_curve.append({
                "index": i,
                "timestamp": self._current_timestamp(current),
                "cash": cash,
                "positions_value": port_value,
                "equity": equity,
            })

        # Close any remaining open positions at the last available close.
        last_bar: Dict[str, Dict[str, Any]] = {}
        for symbol, sb in symbol_bars.items():
            if sb:
                last_bar[symbol] = sb[-1]
        for symbol in list(positions):
            pos = positions[symbol]
            bar = last_bar.get(symbol)
            if bar is None:
                continue
            exit_price = float(bar.get("close", pos["entry_price"]))
            pl = self._realised_pl(pos, exit_price)
            commission = self.commission_for(exit_price, pos["quantity"])
            net_pl = pl - commission
            result.trades.append(
                BacktestTrade(
                    symbol=symbol,
                    side=pos["side"],
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    quantity=pos["quantity"],
                    realised_pl=net_pl,
                    entry_index=pos["entry_index"],
                    exit_index=max_len - 1,
                    entry_bar=pos["entry_bar"],
                    exit_bar=dict(bar),
                )
            )
            result.orders_filled += 1
            cash += pos["quantity"] * exit_price - commission
            positions.pop(symbol, None)
            equity_curve.append({
                "index": max_len,
                "timestamp": self._current_timestamp(last_bar),
                "cash": cash,
                "positions_value": 0.0,
                "equity": cash,
            })

        result.equity_curve = equity_curve
        result.trades = [t for t in result.trades]
        result.metrics = summary_metrics(
            [e["equity"] for e in equity_curve],
            [t.to_dict() for t in result.trades],
            config.periods_per_year,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _valid_bar(bar: Any) -> bool:
        if not isinstance(bar, dict):
            return False
        try:
            float(bar.get("close", 0))
            float(bar.get("high", 0))
            float(bar.get("low", 0))
            float(bar.get("open", 0))
        except (TypeError, ValueError):
            return False
        return bool(bar.get("timestamp") or bar.get("symbol"))

    def commission_for(self, price: float, qty: float) -> float:
        config = self.config
        notional = abs(price * qty)
        return config.commission_per_trade + config.commission_pct * notional

    @staticmethod
    def _position_value(
        positions: Dict[str, Dict[str, Any]], current: Dict[str, Dict[str, Any]],
    ) -> float:
        total = 0.0
        for symbol, pos in positions.items():
            bar = current.get(symbol)
            price = (
                float(bar.get("close", pos["entry_price"]))
                if bar
                else pos["entry_price"]
            )
            qty = float(pos["quantity"])
            total += qty * price
        return total

    def _realised_pl(self, pos: Dict[str, Any], exit_price: float) -> float:
        qty = float(pos["quantity"])
        entry = float(pos["entry_price"])
        if pos["side"] == "buy":
            return (exit_price - entry) * qty
        return (entry - exit_price) * qty

    @staticmethod
    def _current_timestamp(current: Dict[str, Dict[str, Any]]) -> Optional[str]:
        for bar in current.values():
            ts = bar.get("timestamp")
            if ts:
                return str(ts)
        return None
