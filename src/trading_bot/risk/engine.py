"""Independent risk engine.

Approves or rejects every trade before it reaches the broker. Implements the
:class:`trading_bot.engine.RiskEngine` protocol consumed by
:class:`trading_bot.engine.TradingEngine`.

The engine owns all mutable risk state (circuit breaker, daily loss tracking)
and composes the pure rule functions in :mod:`trading_bot.risk.rules`. It
fails closed: when account/market state is uncertain it raises
:class:`trading_bot.exceptions.RiskDataUnavailableError` rather than guessing.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_bot.config.settings import Settings
from trading_bot.exceptions import RiskDataUnavailableError
from trading_bot.logging_config import configure_logging
from trading_bot.risk import rules

logger = configure_logging(__name__).getChild("risk")


class RiskEngine:
    """Independent risk engine — sole gatekeeper for all trade entries.

    Parameters
    ----------
    settings:
        Validated application configuration (risk controls live here).
    state:
        Optional :class:`trading_bot.state.AppState` for surfacing circuit
        breaker and stale-data flags. When omitted the engine tracks state
        internally.

    """

    def __init__(
        self,
        settings: Settings,
        state: Optional[Any] = None,
    ) -> None:
        self.settings = settings
        self.state = state
        self._lock = threading.RLock()
        self._halted = False
        self._halted_at: Optional[datetime] = None
        self._halt_reason = ""
        self._daily_loss_usd = 0.0

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------
    def is_halted(self) -> bool:
        """True when a circuit breaker has halted new entries."""
        with self._lock:
            return self._halted

    def assess(
        self,
        signal: Dict[str, Any],
        account: Dict[str, Any],
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assess a signal and return a risk decision.

        Returns a dict with at least an ``approved`` key. When approved the
        dict also carries ``order`` (the risk-adjusted order), ``qty``,
        ``stop_price`` and ``limit_price``. When rejected it carries
        ``rejection_reason``. Never raises for normal rejections; raises
        :class:`RiskDataUnavailableError` only when state is too uncertain to
        assess.
        """
        decision: Dict[str, Any] = {
            "approved": False,
            "reason": "",
            "rejection_reason": None,
            "order": None,
            "rules": [],
            "qty": None,
            "stop_price": None,
            "limit_price": None,
        }

        with self._lock:
            if self._halted:
                decision["reason"] = f"halted: {self._halt_reason}"
                decision["rejection_reason"] = self._halt_reason
                decision["rules"] = [{"name": "circuit_breaker", "passed": False,
                                      "reason": self._halt_reason}]
                return decision

        # Fail closed on uncertain account state.
        if not account:
            raise RiskDataUnavailableError(
                "risk assessment requires account state"
            )

        # 1. Account must be tradable.
        ok, reason, detail = rules.check_account_tradable(account)
        self._record_rule(decision, detail["name"] if "name" in detail else detail.get("rule"), ok, reason)
        if not ok:
            decision["reason"] = reason
            decision["rejection_reason"] = reason
            return decision

        # 2. Daily loss limit (may trip the circuit breaker).
        equity = rules._as_float(account.get("equity"))
        ok, reason, detail = rules.check_daily_loss_limit(
            self._daily_loss_usd, self.settings, equity,
        )
        self._record_rule(decision, detail["rule"], ok, reason)
        if not ok:
            self._halt(True, reason)
            decision["reason"] = reason
            decision["rejection_reason"] = reason
            return decision

        # 3. Concurrent position cap.
        ok, reason, detail = rules.check_concurrent_positions(
            positions, self.settings,
        )
        self._record_rule(decision, detail["rule"], ok, reason)
        if not ok:
            decision["reason"] = reason
            decision["rejection_reason"] = reason
            return decision

        # 4. Symbol not already held (avoid adding to an open position).
        ok, reason, detail = rules.check_symbol_not_already_held(
            signal, positions,
        )
        self._record_rule(decision, detail["rule"], ok, reason)
        if not ok:
            decision["reason"] = reason
            decision["rejection_reason"] = reason
            return decision

        # 5. Position sizing.
        qty, size_reason = rules.compute_position_size(signal, account, self.settings)
        if qty is None:
            decision["reason"] = size_reason
            decision["rejection_reason"] = size_reason
            decision["rules"].append({"name": "position_size", "passed": False,
                                      "reason": size_reason})
            return decision
        decision["qty"] = qty

        ok, reason, detail = rules.check_position_size(signal, account, self.settings)
        self._record_rule(decision, detail["rule"], ok, reason)
        # check_position_size recomputes size; trust compute_position_size.

        # 6. Portfolio risk (per-trade risk vs equity).
        ok, reason, detail = rules.check_portfolio_risk(
            signal, account, qty, self.settings,
        )
        self._record_rule(decision, detail["rule"], ok, reason)
        if not ok:
            decision["reason"] = reason
            decision["rejection_reason"] = reason
            return decision

        # 7. Stop loss band.
        ok, reason, detail = rules.check_stop_loss_band(signal, self.settings)
        self._record_rule(decision, detail["rule"], ok, reason)
        if not ok:
            decision["reason"] = reason
            decision["rejection_reason"] = reason
            return decision

        # 8. Reward / risk ratio.
        ok, reason, detail = rules.check_reward_risk(signal, self.settings)
        self._record_rule(decision, detail["rule"], ok, reason)
        if not ok:
            decision["reason"] = reason
            decision["rejection_reason"] = reason
            return decision

        # All checks passed — build the risk-adjusted order.
        price = rules._as_float(signal.get("price"))
        side = _normalize_side(signal.get("side", "buy"))
        stop_pct = rules._as_float(
            signal.get("stop_loss_pct"), self.settings.stop_loss_pct,
        )
        take_pct = rules._as_float(
            signal.get("take_profit_pct"), self.settings.take_profit_pct,
        )
        if side == "buy":
            stop_price = price * (1.0 - stop_pct)
            limit_price = price * take_pct
        else:
            stop_price = price * (1.0 + stop_pct)
            limit_price = price * (2.0 - take_pct)

        decision["approved"] = True
        decision["reason"] = "approved"
        decision["stop_price"] = round(stop_price, 4)
        decision["limit_price"] = round(limit_price, 4)
        decision["order"] = {
            "symbol": str(signal.get("symbol", "")).upper(),
            "side": side,
            "qty": round(qty, 4),
            "order_type": "market",
            "stop_price": round(stop_price, 4),
            "limit_price": round(limit_price, 4),
            "time_in_force": "day",
            "signal_id": signal.get("signal_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "risk_reason": "approved",
        }
        logger.info(
            "Risk approved %s %s qty=%.4f stop=%.4f limit=%.4f",
            side, signal.get("symbol"), qty, stop_price, limit_price,
        )
        return decision

    # ------------------------------------------------------------------
    # Mutable risk state (owned by engine)
    # ------------------------------------------------------------------
    def record_loss(self, usd: float) -> None:
        """Record a realized loss (USD) and trip the breaker if the daily
        limit is exceeded.
        """
        loss = abs(rules._as_float(usd))
        with self._lock:
            self._daily_loss_usd += loss
            equity = 0.0
            if self.state is not None:
                try:
                    equity = rules._as_float(
                        getattr(self.state, "equity", 0.0) or 0.0,
                    )
                except Exception:  # noqa: BLE001
                    equity = 0.0
            if self._should_halt_daily_loss(equity):
                self._halt(
                    True,
                    f"daily loss limit reached (${self._daily_loss_usd:.2f})",
                )

    def reset_daily(self) -> None:
        """Reset the in-session daily loss tracker (call at session start)."""
        with self._lock:
            self._daily_loss_usd = 0.0

    def trip_circuit_breaker(self, reason: str) -> None:
        """Trip the circuit breaker, halting new entries."""
        self._halt(True, reason)

    def reset_circuit_breaker(self) -> None:
        """Clear the circuit breaker (manual or scheduled reset)."""
        self._halt(False, "")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _halt(self, halted: bool, reason: str) -> None:
        with self._lock:
            self._halted = halted
            self._halt_reason = reason
            self._halted_at = (
                datetime.now(timezone.utc) if halted else None
            )
            if self.state is not None:
                try:
                    if halted:
                        self.state.trip_circuit_breaker(self._halted_at)
                    else:
                        self.state.reset_circuit_breaker()
                except Exception:  # noqa: BLE001
                    pass
        if halted:
            logger.warning("Circuit breaker tripped: %s", reason)
        else:
            logger.info("Circuit breaker reset")

    def _should_halt_daily_loss(self, equity: float) -> bool:
        ok, _, _ = rules.check_daily_loss_limit(
            self._daily_loss_usd, self.settings, equity,
        )
        return not ok

    @staticmethod
    def _record_rule(
        decision: Dict[str, Any], name: str, ok: bool, reason: str,
    ) -> None:
        decision.setdefault("rules", []).append({
            "name": name,
            "passed": ok,
            "reason": reason,
        })


def _normalize_side(value: Any) -> str:
    return str(value or "buy").strip().lower()
