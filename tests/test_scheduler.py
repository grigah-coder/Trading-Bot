from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from core.market_calendar import ExchangeMarketCalendar, MarketSession
from core.scheduler import MarketScheduler


class FixedCalendar:
    def __init__(self, session: MarketSession | None) -> None:
        self.session = session

    def session_for(self, trading_date: date) -> MarketSession | None:
        if self.session and self.session.session_date == trading_date:
            return self.session
        return None


def test_scheduler_starts_session_and_evaluates_at_configured_intervals() -> None:
    zone = ZoneInfo("America/New_York")
    session = MarketSession(
        date(2026, 7, 2),
        datetime(2026, 7, 2, 9, 30, tzinfo=zone),
        datetime(2026, 7, 2, 16, 0, tzinfo=zone),
    )
    scheduler = MarketScheduler(
        FixedCalendar(session), timezone_name="America/New_York", interval=timedelta(minutes=1)
    )
    starts: list[date] = []
    evaluations: list[datetime] = []

    assert not scheduler.run_pending(
        datetime(2026, 7, 2, 13, 29, tzinfo=timezone.utc),
        on_session_start=lambda value: starts.append(value.session_date),
        evaluate=lambda: evaluations.append(datetime(2026, 7, 2, 13, 29, tzinfo=timezone.utc)),
    )
    assert scheduler.run_pending(
        datetime(2026, 7, 2, 13, 30, tzinfo=timezone.utc),
        on_session_start=lambda value: starts.append(value.session_date),
        evaluate=lambda: evaluations.append(datetime(2026, 7, 2, 13, 30, tzinfo=timezone.utc)),
    )
    assert not scheduler.run_pending(
        datetime(2026, 7, 2, 13, 30, 30, tzinfo=timezone.utc),
        on_session_start=lambda value: starts.append(value.session_date),
        evaluate=lambda: evaluations.append(datetime(2026, 7, 2, 13, 30, 30, tzinfo=timezone.utc)),
    )
    assert scheduler.run_pending(
        datetime(2026, 7, 2, 13, 31, tzinfo=timezone.utc),
        on_session_start=lambda value: starts.append(value.session_date),
        evaluate=lambda: evaluations.append(datetime(2026, 7, 2, 13, 31, tzinfo=timezone.utc)),
    )
    assert not scheduler.run_pending(
        datetime(2026, 7, 2, 20, 0, tzinfo=timezone.utc),
        on_session_start=lambda value: starts.append(value.session_date),
        evaluate=lambda: evaluations.append(datetime(2026, 7, 2, 20, 0, tzinfo=timezone.utc)),
    )

    assert starts == [date(2026, 7, 2)]
    assert len(evaluations) == 2


def test_exchange_calendar_handles_holiday_dst_and_early_close() -> None:
    calendar = ExchangeMarketCalendar("XNYS")

    assert calendar.session_for(date(2026, 7, 3)) is None
    summer_session = calendar.session_for(date(2026, 7, 2))
    winter_early_close = calendar.session_for(date(2026, 11, 27))

    assert summer_session is not None
    assert summer_session.opens_at.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M") == "09:30"
    assert summer_session.opens_at.hour == 13
    assert winter_early_close is not None
    assert winter_early_close.closes_at.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M") == "13:00"
    assert winter_early_close.opens_at.hour == 14
    assert calendar.session_for(date(2030, 1, 2)) is not None


def test_scheduler_rejects_naive_clock_values() -> None:
    scheduler = MarketScheduler(FixedCalendar(None))

    try:
        scheduler.run_pending(
            datetime(2026, 7, 2, 10),
            on_session_start=lambda _: None,
            evaluate=lambda: None,
        )
    except ValueError as error:
        assert "timezone-aware" in str(error)
    else:
        raise AssertionError("naive clock values must be rejected")
