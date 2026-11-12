"""OHLCV data validation.

Validates market-data bars to prevent look-ahead bias and data leakage.
Every bar is checked for internal consistency and plausibility before it is
allowed into the strategy or risk engines.

A valid bar satisfies:
* ``open``, ``high``, ``low``, ``close`` are all finite and strictly positive,
* ``high`` >= max(open, close) and ``high`` >= low,
* ``low`` <= min(open, close) and ``low`` > 0,
* ``volume`` is finite and >= 0,
* the timestamp is timezone-aware and not in the future relative to the
  provider clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass(frozen=True)
class Bar:
    """A single validated OHLCV bar.

    Attributes are read-only to guarantee the bar cannot be mutated after
    validation (protects against downstream look-ahead bias).
    """

    timestamp: datetime  # tz-aware UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(f"bar timestamp must be timezone-aware: {self.timestamp!r}")
        if not math.isfinite(self.open) or self.open <= 0:
            raise ValueError(f"invalid open price {self.open!r} for {self.symbol}")
        if not math.isfinite(self.high) or self.high <= 0:
            raise ValueError(f"invalid high price {self.high!r} for {self.symbol}")
        if not math.isfinite(self.low) or self.low <= 0:
            raise ValueError(f"invalid low price {self.low!r} for {self.symbol}")
        if not math.isfinite(self.close) or self.close <= 0:
            raise ValueError(f"invalid close price {self.close!r} for {self.symbol}")
        if not math.isfinite(self.volume) or self.volume < 0:
            raise ValueError(f"invalid volume {self.volume!r} for {self.symbol}")
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"high {self.high} < max(open={self.open}, close={self.close}) "
                f"for {self.symbol} at {self.timestamp}"
            )
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"low {self.low} > min(open={self.open}, close={self.close}) "
                f"for {self.symbol} at {self.timestamp}"
            )
        if self.low > self.high:
            raise ValueError(f"low {self.low} > high {self.high} for {self.symbol}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        }


class DataValidationError(ValueError):
    """Raised when a bar fails validation."""


def validate_bar(
    timestamp: Any,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    symbol: str,
    timeframe: str,
    clock: Optional[callable] = None,  # type: ignore[name-defined]
) -> Bar:
    """Build and validate a :class:`Bar`.

    The optional ``clock`` (returns tz-aware ``datetime``) is used for the
    future-timestamp check. When a bar's timestamp is after the clock, it is
    rejected to prevent look-ahead bias.
    """
    try:
        ts = _normalize_timestamp(timestamp)
    except Exception as exc:
        raise DataValidationError(
            f"invalid timestamp {timestamp!r} for {symbol}: {exc}"
        ) from exc

    now = clock() if clock is not None else datetime.now(timezone.utc)
    if ts > now:
        raise DataValidationError(
            f"bar timestamp {ts.isoformat()} is in the future (now {now.isoformat()}) "
            f"for {symbol}; rejecting to prevent look-ahead bias"
        )

    try:
        return Bar(
            timestamp=ts,
            open=float(open_price),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(volume),
            symbol=str(symbol).upper(),
            timeframe=str(timeframe),
        )
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"invalid bar for {symbol} at {ts.isoformat()}: {exc}"
        ) from exc


def validate_bars(
    bars: list[dict[str, Any]],
    symbol: str,
    timeframe: str,
    clock: Optional[callable] = None,  # type: ignore[name-defined]
) -> list[Bar]:
    """Validate a list of raw bar dicts, returning :class:`Bar` objects.

    Invalid bars are skipped (not raised) so a single bad bar does not discard
    the entire batch; the count of skipped bars is logged by the caller via the
    returned list length being smaller than the input.
    """
    valid: list[Bar] = []
    for raw in bars:
        try:
            bar = validate_bar(
                timestamp=raw.get("timestamp") or raw.get("t"),
                open_price=raw.get("open") or raw.get("o"),
                high=raw.get("high") or raw.get("h"),
                low=raw.get("low") or raw.get("l"),
                close=raw.get("close") or raw.get("c"),
                volume=raw.get("volume") or raw.get("v", 0),
                symbol=symbol,
                timeframe=timeframe,
                clock=clock,
            )
            valid.append(bar)
        except DataValidationError:
            continue
    return valid


def _normalize_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, (int, float)):
        ts = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        ts = datetime.fromisoformat(value)
    else:
        raise ValueError(f"unsupported timestamp type: {type(value)!r}")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)
