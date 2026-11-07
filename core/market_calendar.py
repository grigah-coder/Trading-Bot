"""Exchange session data backed by the exchange-calendars package."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Protocol

import exchange_calendars
import pandas as pd

from core.exceptions import MarketCalendarError


@dataclass(frozen=True, slots=True)
class MarketSession:
    session_date: date
    opens_at: datetime
    closes_at: datetime

    def __post_init__(self) -> None:
        if self.opens_at.tzinfo is None or self.closes_at.tzinfo is None:
            raise ValueError("Market session times must be timezone-aware")
        if self.opens_at.astimezone(timezone.utc) >= self.closes_at.astimezone(
            timezone.utc
        ):
            raise ValueError("Market session open must be before close")


class MarketCalendar(Protocol):
    def session_for(self, trading_date: date) -> MarketSession | None: ...


class ExchangeMarketCalendar:
    """Resolve regular and early-close sessions from an exchange calendar."""

    def __init__(self, calendar_name: str = "XNYS") -> None:
        self.calendar_name = calendar_name
        try:
            # exchange_calendars 4.13 currently relies on pandas' deprecated
            # timezone-aware Series.values during calendar construction.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Series.values returning an ndarray that drops timezone information",
                )
                self._calendar = exchange_calendars.get_calendar(
                    calendar_name, start="2000-01-01", end="2100-12-31"
                )
        except Exception as exc:
            raise MarketCalendarError(
                f"Unknown or unavailable exchange calendar: {calendar_name}",
                context={"calendar": calendar_name},
            ) from exc

    @property
    def timezone_name(self) -> str:
        return getattr(self._calendar.tz, "key", str(self._calendar.tz))

    def session_for(self, trading_date: date) -> MarketSession | None:
        label = pd.Timestamp(trading_date)
        try:
            if not self._calendar.is_session(label):
                return None
            schedule_row = self._calendar.schedule.loc[label]
            opens_at = schedule_row["open"].to_pydatetime()
            closes_at = schedule_row["close"].to_pydatetime()
        except Exception as exc:
            raise MarketCalendarError(
                f"Could not resolve {self.calendar_name} session for {trading_date}",
                context={
                    "calendar": self.calendar_name,
                    "session_date": trading_date.isoformat(),
                },
            ) from exc
        return MarketSession(trading_date, opens_at, closes_at)
