"""Synchronous market-session scheduler for periodic strategy evaluation."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

from core.market_calendar import MarketCalendar, MarketSession


class MarketScheduler:
    def __init__(
        self,
        calendar: MarketCalendar,
        *,
        timezone_name: str = "America/New_York",
        interval: timedelta = timedelta(minutes=1),
    ) -> None:
        if interval <= timedelta(0):
            raise ValueError("evaluation interval must be positive")
        self.calendar = calendar
        self.timezone = ZoneInfo(timezone_name)
        self.interval = interval
        self._active_session_date: date | None = None
        self._last_evaluation_at: datetime | None = None

    def run_pending(
        self,
        now: datetime,
        *,
        on_session_start: Callable[[MarketSession], None],
        evaluate: Callable[[], object],
    ) -> bool:
        """Start an open session once, then evaluate at the configured cadence."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")

        local_now = now.astimezone(self.timezone)
        session = self.calendar.session_for(local_now.date())
        if session is None:
            self._active_session_date = None
            self._last_evaluation_at = None
            return False

        instant = now.astimezone(timezone.utc)
        opens_at = session.opens_at.astimezone(timezone.utc)
        closes_at = session.closes_at.astimezone(timezone.utc)
        if not opens_at <= instant < closes_at:
            if instant >= closes_at:
                self._active_session_date = None
                self._last_evaluation_at = None
            return False

        if self._active_session_date != session.session_date:
            on_session_start(session)
            self._active_session_date = session.session_date
            self._last_evaluation_at = None

        if (
            self._last_evaluation_at is not None
            and instant - self._last_evaluation_at.astimezone(timezone.utc) < self.interval
        ):
            return False

        evaluate()
        self._last_evaluation_at = instant
        return True
