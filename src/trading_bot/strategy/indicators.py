"""Pure technical indicators.

All functions are stateless and operate on plain lists of prices, returning a
list aligned to the input (``None`` where there is insufficient history). No
look-ahead: value at index ``i`` only depends on prices at ``<= i``.
"""

from __future__ import annotations

import math
from typing import List, Optional


def _clean(prices: List[Optional[float]]) -> List[float]:
    """Return finite prices, dropping None / non-finite entries (keeps order)."""
    out: List[float] = []
    for p in prices:
        if p is None:
            continue
        try:
            v = float(p)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def sma(prices: List[Optional[float]], period: int) -> List[Optional[float]]:
    """Simple moving average.

    ``result[i]`` is the mean of ``prices[max(0, i-period+1) .. i]`` when at
    least ``period`` finite prices are available up to ``i``, else ``None``.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period!r}")
    result: List[Optional[float]] = []
    window: List[float] = []
    window_sum = 0.0
    for p in prices:
        if p is None:
            result.append(None)
            continue
        try:
            v = float(p)
        except (TypeError, ValueError):
            result.append(None)
            continue
        if not math.isfinite(v):
            result.append(None)
            continue
        window.append(v)
        window_sum += v
        if len(window) > period:
            window_sum -= window.pop(0)
        if len(window) >= period:
            result.append(window_sum / len(window))
        else:
            result.append(None)
    return result


def ema(prices: List[Optional[float]], period: int) -> List[Optional[float]]:
    """Exponential moving average.

    Seed with the first ``period`` prices' SMA; thereafter recurse with
    ``alpha = 2 / (period + 1)``. Returns ``None`` until the seed window is
    full.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period!r}")
    alpha = 2.0 / (period + 1.0)
    result: List[Optional[float]] = []
    seeded = False
    buffer: List[float] = []
    prev: Optional[float]
    for p in prices:
        if p is None:
            result.append(None)
            continue
        try:
            v = float(p)
        except (TypeError, ValueError):
            result.append(None)
            continue
        if not math.isfinite(v):
            result.append(None)
            continue
        buffer.append(v)
        if not seeded:
            if len(buffer) < period:
                result.append(None)
                continue
            prev = sum(buffer) / period
            seeded = True
            result.append(prev)
            continue
        prev = alpha * v + (1.0 - alpha) * prev
        result.append(prev)
    return result


def rsi(prices: List[Optional[float]], period: int) -> List[Optional[float]]:
    """Relative Strength Index (Wilder's smoothing).

    Returns ``None`` until at least ``period + 1`` finite prices are available.
    Values are in ``[0, 100]``.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period!r}")
    result: List[Optional[float]] = []
    seeded = False
    avg_gain = 0.0
    avg_loss = 0.0
    prev_price: Optional[float] = None
    gains: List[float] = []
    losses: List[float] = []
    for p in prices:
        if p is None:
            result.append(None)
            prev_price = None
            continue
        try:
            v = float(p)
        except (TypeError, ValueError):
            result.append(None)
            prev_price = None
            continue
        if not math.isfinite(v):
            result.append(None)
            prev_price = None
            continue
        if prev_price is None:
            prev_price = v
            result.append(None)
            continue
        change = v - prev_price
        prev_price = v
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
        if not seeded:
            if len(gains) < period:
                result.append(None)
                continue
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            seeded = True
        else:
            avg_gain = (avg_gain * (period - 1) + gains[-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[-1]) / period
        if avg_loss == 0.0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100.0 - (100.0 / (1.0 + rs)))
    return result


def crossover(fast: List[Optional[float]], slow: List[Optional[float]]) -> List[bool]:
    """Element-wise strict crossover: ``fast[i] > slow[i] and fast[i-1] <= slow[i-1]``."""
    out: List[bool] = []
    for i in range(len(fast)):
        f_cur = fast[i]
        s_cur = slow[i] if i < len(slow) else None
        if f_cur is None or s_cur is None:
            out.append(False)
            continue
        if i == 0:
            out.append(False)
            continue
        f_prev = fast[i - 1]
        s_prev = slow[i - 1] if i - 1 < len(slow) else None
        if f_prev is None or s_prev is None:
            out.append(False)
            continue
        out.append(f_cur > s_cur and f_prev <= s_prev)
    return out


def crossunder(fast: List[Optional[float]], slow: List[Optional[float]]) -> List[bool]:
    """Element-wise strict crossunder: ``fast[i] < slow[i] and fast[i-1] >= slow[i-1]``."""
    out: List[bool] = []
    for i in range(len(fast)):
        f_cur = fast[i]
        s_cur = slow[i] if i < len(slow) else None
        if f_cur is None or s_cur is None:
            out.append(False)
            continue
        if i == 0:
            out.append(False)
            continue
        f_prev = fast[i - 1]
        s_prev = slow[i - 1] if i - 1 < len(slow) else None
        if f_prev is None or s_prev is None:
            out.append(False)
            continue
        out.append(f_cur < s_cur and f_prev >= s_prev)
    return out
