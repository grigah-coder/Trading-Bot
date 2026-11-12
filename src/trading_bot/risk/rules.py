"""Pure risk rules.

Each rule is a pure function: given the signal, account, positions and
settings it returns ``(ok, reason, detail)``. Rules never mutate state and
never reach out to the broker or market data, so they are easy to unit test
and reason about.

The risk engine (``engine.py``) composes these rules and owns all mutable
state (circuit breaker, daily loss tracking).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

RuleResult = Tuple[bool, str, Dict[str, Any]]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _side(signal: Dict[str, Any]) -> str:
    return str(signal.get("side", "buy")).strip().lower()


def check_account_tradable(account: Dict[str, Any]) -> RuleResult:
    if not account:
        return False, "account unavailable", {"rule": "account_tradable"}
    if account.get("account_blocked"):
        return False, "account blocked", {"rule": "account_tradable"}
    if account.get("trading_blocked"):
        return False, "trading blocked", {"rule": "account_tradable"}
    status = str(account.get("status", "")).lower()
    if status not in {"active", "trading"}:
        return False, (
            f"account not tradable (status={status!r})"
        ), {"rule": "account_tradable"}
    return True, "account tradable", {"rule": "account_tradable"}


def check_concurrent_positions(
    positions: List[Dict[str, Any]], settings: Any,
) -> RuleResult:
    open_positions = [
        p for p in positions
        if _as_float(p.get("qty")) != 0
    ]
    if len(open_positions) >= settings.max_concurrent_positions:
        return False, (
            f"max concurrent positions reached "
            f"({len(open_positions)}/{settings.max_concurrent_positions})"
        ), {"rule": "concurrent_positions"}
    return True, "concurrent positions within limit", {
        "rule": "concurrent_positions",
    }


def check_symbol_not_already_held(
    signal: Dict[str, Any], positions: List[Dict[str, Any]],
) -> RuleResult:
    symbol = str(signal.get("symbol", "")).upper()
    for position in positions:
        if str(position.get("symbol", "")).upper() == symbol:
            return False, f"already holding position in {symbol}", {
                "rule": "symbol_not_held",
                "symbol": symbol,
            }
    return True, "symbol not already held", {"rule": "symbol_not_held"}


def compute_position_size(
    signal: Dict[str, Any], account: Dict[str, Any], settings: Any,
) -> Tuple[Optional[float], str]:
    """Compute the maximum allowable quantity for the signal.

    Returns ``(qty, reason)``. ``qty`` is ``None`` when the trade cannot be
    sized (e.g. no price or no equity).
    """
    price = _as_float(signal.get("price"))
    equity = _as_float(account.get("equity"))
    cash = _as_float(account.get("cash"))
    if price <= 0:
        return None, "no usable price in signal"
    if equity <= 0 and cash <= 0:
        return None, "no usable equity/cash in account"
    base = equity if equity > 0 else cash

    # Notional cap from position size pct of equity.
    max_by_pct = base * settings.position_size_pct / price
    # Absolute notional cap.
    max_by_notional = settings.max_position_notional / price
    # Share count cap.
    max_by_shares = float(settings.max_shares_per_trade)

    qty = min(max_by_pct, max_by_notional, max_by_shares)
    if qty <= 0:
        return None, "computed quantity is zero or negative"
    return qty, "sized"


def check_position_size(
    signal: Dict[str, Any], account: Dict[str, Any], settings: Any,
) -> RuleResult:
    qty, reason = compute_position_size(signal, account, settings)
    if qty is None:
        return False, reason, {"rule": "position_size"}
    return True, f"position sized to {qty:.4f} shares ({reason})", {
        "rule": "position_size",
        "qty": qty,
    }


def check_portfolio_risk(
    signal: Dict[str, Any], account: Dict[str, Any], qty: float,
    settings: Any,
) -> RuleResult:
    price = _as_float(signal.get("price"))
    equity = _as_float(account.get("equity"))
    if price <= 0 or equity <= 0 or qty <= 0:
        return False, "no usable price/equity/qty for portfolio risk", {
            "rule": "portfolio_risk",
        }
    stop_pct = _as_float(signal.get("stop_loss_pct"), settings.stop_loss_pct)
    stop_price = (
        price * (1.0 - stop_pct) if _side(signal) == "buy"
        else price * (1.0 + stop_pct)
    )
    risk_per_share = abs(price - stop_price)
    risk_usd = risk_per_share * qty
    max_risk_usd = equity * settings.max_portfolio_risk_pct
    if risk_usd > max_risk_usd:
        return False, (
            f"trade risk ${risk_usd:.2f} exceeds max ${max_risk_usd:.2f}"
        ), {"rule": "portfolio_risk"}
    return True, "portfolio risk within limit", {"rule": "portfolio_risk"}


def check_stop_loss_band(signal: Dict[str, Any], settings: Any) -> RuleResult:
    price = _as_float(signal.get("price"))
    stop_pct = _as_float(signal.get("stop_loss_pct"), settings.stop_loss_pct)
    if price <= 0:
        return False, "no usable price for stop loss check", {
            "rule": "stop_loss_band",
        }
    if stop_pct < settings.stop_loss_min_pct:
        return False, (
            f"stop loss {stop_pct:.3f} below minimum "
            f"{settings.stop_loss_min_pct:.3f}"
        ), {"rule": "stop_loss_band"}
    if stop_pct > settings.stop_loss_max_pct:
        return False, (
            f"stop loss {stop_pct:.3f} above maximum "
            f"{settings.stop_loss_max_pct:.3f}"
        ), {"rule": "stop_loss_band"}
    return True, "stop loss within band", {"rule": "stop_loss_band"}


def check_reward_risk(signal: Dict[str, Any], settings: Any) -> RuleResult:
    price = _as_float(signal.get("price"))
    stop_pct = _as_float(signal.get("stop_loss_pct"), settings.stop_loss_pct)
    take_pct = _as_float(signal.get("take_profit_pct"), settings.take_profit_pct)
    if price <= 0:
        return False, "no usable price for reward/risk check", {
            "rule": "reward_risk",
        }
    risk = price * stop_pct
    reward = price * take_pct
    if risk <= 0:
        return False, "non-positive risk", {"rule": "reward_risk"}
    ratio = reward / risk
    if ratio < settings.min_reward_risk_ratio:
        return False, (
            f"reward/risk {ratio:.2f} below minimum "
            f"{settings.min_reward_risk_ratio:.2f}"
        ), {"rule": "reward_risk"}
    return True, f"reward/risk {ratio:.2f} acceptable", {
        "rule": "reward_risk",
    }


def check_daily_loss_limit(
    daily_loss_usd: float, settings: Any, equity: float = 0.0,
) -> RuleResult:
    if not settings.daily_loss_limit_enabled:
        return True, "daily loss limit disabled", {"rule": "daily_loss_limit"}
    limit_usd = settings.daily_loss_limit_usd
    limit_pct_usd = equity * settings.daily_loss_limit_pct
    effective_limit = min(limit_usd, limit_pct_usd) if equity > 0 else limit_usd
    if daily_loss_usd >= effective_limit:
        return False, (
            f"daily loss limit reached (${daily_loss_usd:.2f} >= "
            f"${effective_limit:.2f})"
        ), {"rule": "daily_loss_limit"}
    return True, "daily loss limit not reached", {"rule": "daily_loss_limit"}
