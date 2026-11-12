"""Market session detection.

Determines whether the US equity market (NYSE/NASDAQ) is open at a given time,
accounts for the opening-volatility blackout window, and normalizes timestamps
to the configured market timezone (America/New_York by default).

No external market-calendar dependency is used: weekends are treated as closed
and the standard 09:30-16:00 ET session is assumed. This is deliberately
conservative — early closes and holidays are treated as open, which only makes
the system more cautious (it will still block new trades outside the session).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


class MarketCalendar:
    """US equity market session calendar.

    Assumes the regular session 09:30-16:00 America/New_York, Monday-Friday.
    """

    def __init__(
        self,
        market_timezone: str = "America/New_York",
        market_open_time: str = "09:30",
        market_close_time: str = "16:00",
        opening_blackout_minutes: int = 30,
    ) -> None:
        """Create a calendar for the given session and timezone."""
        self.tz = ZoneInfo(market_timezone)
        open_h, open_m = _parse_hhmm(market_open_time)
        close_h, close_m = _parse_hhmm(market_close_time)
        self.market_open = time(open_h, open_m)
        self.market_close = time(close_h, close_m)
        self.opening_blackout_minutes = max(0, int(opening_blackout_minutes))
        self._blackout_end = _add_minutes(self.market_open, self.opening_blackout_minutes)

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------
    def is_market_open(self, moment: Optional[datetime] = None) -> bool:
        """Return True when the market is in regular session at ``moment``."""
        moment = _to_market(moment or datetime.now(self.tz), self.tz)
        if moment.weekday() >= 5:  # Saturday/Sunday
            return False
        return self.market_open <= moment.time() < self.market_close

    def is_in_opening_blackout(self, moment: Optional[datetime] = None) -> bool:
        """Return True during the opening-volatility blackout after the open.

        New entries are suppressed here; protective exits are still allowed.
        """
        if self.opening_blackout_minutes <= 0:
            return False
        moment = _to_market(moment or datetime.now(self.tz), self.tz)
        if moment.weekday() >= 5:
            return False
        return self.market_open <= moment.time() < self._blackout_end

    def next_market_open(self, moment: Optional[datetime] = None) -> datetime:
        """Return the next market-open timestamp (market-local)."""
        moment = _to_market(moment or datetime.now(self.tz), self.tz)
        candidate = moment.replace(
            hour=self.market_open.hour,
            minute=self.market_open.minute,
            second=0,
            microsecond=0,
        )
        while not self.is_market_open(candidate):
            candidate += timedelta(days=1)
            candidate = candidate.replace(
                hour=self.market_open.hour,
                minute=self.market_open.minute,
                second=0,
                microsecond=0,
            )
        return candidate

    def next_market_close(self, moment: Optional[datetime] = None) -> datetime:
        """Return the next market-close timestamp (market-local)."""
        moment = _to_market(moment or datetime.now(self.tz), self.tz)
        candidate = moment.replace(
            hour=self.market_close.hour,
            minute=self.market_close.minute,
            second=0,
            microsecond=0,
        )
        if candidate <= moment:
            candidate += timedelta(days=1)
        while not self.is_market_open(candidate - timedelta(minutes=1)):
            candidate += timedelta(days=1)
            candidate = candidate.replace(
                hour=self.market_close.hour,
                minute=self.market_close.minute,
                second=0,
                microsecond=0,
            )
        return candidate

    def session_bounds(self, moment: Optional[datetime] = None) -> tuple[datetime, datetime]:
        """Return (open, close) for the session containing ``moment``."""
        moment = _to_market(moment or datetime.now(self.tz), self.tz)
        open_dt = moment.replace(
            hour=self.market_open.hour,
            minute=self.market_open.minute,
            second=0,
            microsecond=0,
        )
        close_dt = moment.replace(
            hour=self.market_close.hour,
            minute=self.market_close.minute,
            second=0,
            microsecond=0,
        )
        if not (open_dt <= moment < close_dt):
            # Moment is outside today's session; find the next session.
            open_dt = self.next_market_open(moment)
            close_dt = open_dt.replace(
                hour=self.market_close.hour,
                minute=self.market_close.minute,
            )
        return open_dt, close_dt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_hhmm(value: str) -> tuple[int, int]:
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"time must be HH:MM, got {value!r}")
    return int(parts[0]), int(parts[1])


def _add_minutes(t: time, minutes: int) -> time:
    dt = datetime(2000, 1, 1, t.hour, t.minute) + timedelta(minutes=minutes)
    return dt.time()


def _to_market(moment: datetime, tz: ZoneInfo) -> datetime:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=tz)
    return moment.astimezone(tz)


# Optional import of typing.Optional used above; define after use is fine but
# keep explicit for mypy/pyright.
from typing import Optional  # noqa: E402
