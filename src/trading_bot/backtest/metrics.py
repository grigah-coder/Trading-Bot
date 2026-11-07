"""Performance metrics for backtesting.

Pure functions operating on equity curves and return series. No dependencies
beyond the standard library, so they are deterministic and easy to unit test.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _as_floats(values: Iterable[Optional[float]]) -> List[float]:
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def returns_from_equity(equity: Sequence[float]) -> List[float]:
    """Per-period simple returns from an equity curve.

    ``returns[i] = equity[i]/equity[i-1] - 1``. The first entry has no prior
    point and is omitted.
    """
    eq = _as_floats(equity)
    rets: List[float] = []
    for i in range(1, len(eq)):
        prev = eq[i - 1]
        if prev <= 0:
            rets.append(0.0)
        else:
            rets.append(eq[i] / prev - 1.0)
    return rets


def total_return(equity: Sequence[float]) -> float:
    eq = _as_floats(equity)
    if len(eq) < 2 or eq[0] <= 0:
        return 0.0
    return eq[-1] / eq[0] - 1.0


def cagr(equity: Sequence[float], periods_per_year: float = 252.0) -> float:
    """Compound annual growth rate from an equity curve."""
    eq = _as_floats(equity)
    if len(eq) < 2 or eq[0] <= 0 or periods_per_year <= 0:
        return 0.0
    years = (len(eq) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return (eq[-1] / eq[0]) ** (1.0 / years) - 1.0


def max_drawdown(equity: Sequence[float]) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (0..1)."""
    eq = _as_floats(equity)
    if not eq:
        return 0.0
    peak = eq[0]
    mdd = 0.0
    for value in eq:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak
            if dd > mdd:
                mdd = dd
    return mdd


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: Sequence[float], ddof: int = 1) -> float:
    n = len(values)
    if n <= ddof:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - ddof)
    return math.sqrt(max(var, 0.0))


def sharpe_ratio(
    returns: Sequence[float],
    periods_per_year: float = 252.0,
    risk_free_per_period: float = 0.0,
) -> float:
    """Annualised Sharpe ratio.

    Uses excess returns over a per-period risk-free rate. Returns 0 when there
    is insufficient data or zero volatility.
    """
    rets = [r - risk_free_per_period for r in _as_floats(returns)]
    if len(rets) < 2:
        return 0.0
    mean = _mean(rets)
    std = _std(rets)
    if std <= 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: Sequence[float],
    periods_per_year: float = 252.0,
    risk_free_per_period: float = 0.0,
    target_return_per_period: float = 0.0,
) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    rets = [r - risk_free_per_period for r in _as_floats(returns)]
    if len(rets) < 2:
        return 0.0
    mean = _mean(rets)
    downside = [
        min(r - target_return_per_period, 0.0) ** 2 for r in rets
    ]
    downside_std = math.sqrt(sum(downside) / len(downside)) if downside else 0.0
    if downside_std <= 0:
        return 0.0
    return (mean / downside_std) * math.sqrt(periods_per_year)


def _pnl(trade: Dict[str, Any]) -> Optional[float]:
    """Realised P/L for a trade, preferring ``realized_pl`` over ``pnl``.

    A zero P/L is preserved (it counts as breakeven); only missing values are
    dropped.
    """
    v = trade.get("realized_pl")
    if v is None:
        v = trade.get("pnl")
    return v


def win_rate(trades: Sequence[Dict[str, Any]]) -> float:
    """Fraction of trades with positive realised P/L."""
    pnls = _as_floats(_pnl(t) for t in trades)
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def expectancy(trades: Sequence[Dict[str, Any]]) -> float:
    """Average realised P/L per trade."""
    pnls = _as_floats(_pnl(t) for t in trades)
    if not pnls:
        return 0.0
    return sum(pnls) / len(pnls)


def profit_factor(trades: Sequence[Dict[str, Any]]) -> float:
    """Gross profit / gross loss. ``inf`` when there are no losses."""
    pnls = _as_floats(_pnl(t) for t in trades)
    if not pnls:
        return 0.0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(-p for p in pnls if p < 0)
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calmar_ratio(
    equity: Sequence[float],
    periods_per_year: float = 252.0,
) -> float:
    """CAGR / max drawdown. 0 when there is no drawdown or no data."""
    eq = _as_floats(equity)
    if len(eq) < 2:
        return 0.0
    mdd = max_drawdown(eq)
    if mdd <= 0:
        return 0.0
    return cagr(eq, periods_per_year) / mdd


def summary_metrics(
    equity: Sequence[float],
    trades: Sequence[Dict[str, Any]],
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    """Aggregate the most commonly-reported backtest metrics."""
    eq = _as_floats(equity)
    rets = returns_from_equity(eq)
    return {
        "total_return": total_return(eq),
        "cagr": cagr(eq, periods_per_year),
        "max_drawdown": max_drawdown(eq),
        "sharpe": sharpe_ratio(rets, periods_per_year),
        "sortino": sortino_ratio(rets, periods_per_year),
        "calmar": calmar_ratio(eq, periods_per_year),
        "win_rate": win_rate(trades),
        "expectancy": expectancy(trades),
        "profit_factor": profit_factor(trades),
        "trade_count": float(len(trades)),
    }
