from datetime import date, datetime, timedelta, timezone

import pytest

from core.market_calendar import ExchangeMarketCalendar
from data.history import HistoricalDataService
from data.models import Candle, Timeframe
from data.validation import CandleValidator, DataFreshnessError, DataQualityError

UTC = timezone.utc


def candle(at: datetime, *, open_: float = 10, high: float = 12, low: float = 9,
           close: float = 11, volume: float | None = 100) -> Candle:
    return Candle("AAPL", at, open_, high, low, close, volume)


def test_candle_normalizes_aware_timestamp_to_utc() -> None:
    local = datetime(2026, 11, 2, 9, 30, tzinfo=timezone(timedelta(hours=-5)))
    assert candle(local).timestamp == datetime(2026, 11, 2, 14, 30, tzinfo=UTC)


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        candle(datetime(2026, 11, 2, 14, 30))


@pytest.mark.parametrize("kwargs", [
    {"open_": -1}, {"high": 8}, {"low": 13}, {"close": 20}, {"volume": None},
    {"volume": -1}, {"volume": float("nan")},
])
def test_invalid_candle_values_are_rejected(kwargs: dict[str, object]) -> None:
    validator = CandleValidator()
    with pytest.raises(DataQualityError):
        validator.validate([candle(datetime(2026, 11, 2, 14, 30, tzinfo=UTC), **kwargs)])


def test_duplicate_and_out_of_order_candles_are_rejected() -> None:
    at = datetime(2026, 11, 2, 14, 30, tzinfo=UTC)
    validator = CandleValidator()
    with pytest.raises(DataQualityError, match="Duplicate"):
        validator.validate([candle(at), candle(at)])
    with pytest.raises(DataQualityError, match="out of order"):
        validator.validate([candle(at + timedelta(minutes=1)), candle(at)])


def test_missing_session_candle_is_detected_without_flagging_overnight_gap() -> None:
    calendar = ExchangeMarketCalendar()
    session = calendar.session_for(date(2026, 11, 2))
    assert session is not None
    bars = [candle(session.opens_at), candle(session.opens_at + timedelta(minutes=2))]
    with pytest.raises(DataQualityError, match="Missing candle"):
        CandleValidator(calendar).validate(bars, timeframe=Timeframe(1, "minute"))


def test_stale_latest_price_is_rejected() -> None:
    from data.models import LatestPrice
    validator = CandleValidator()
    now = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
    with pytest.raises(DataFreshnessError):
        validator.validate_latest(LatestPrice("AAPL", 10, now - timedelta(minutes=2)), now=now,
                                  max_age=timedelta(minutes=1))


class FakeProvider:
    def __init__(self, bars: dict[str, tuple[Candle, ...]]) -> None:
        self.bars = bars
        self.calls: list[datetime] = []

    def fetch_bars(self, symbols, timeframe, start, end):
        self.calls.append(end)
        return self.bars


def test_history_does_not_expose_candles_unavailable_at_decision_time() -> None:
    start = datetime(2026, 11, 2, 14, 30, tzinfo=UTC)
    decision = start + timedelta(minutes=2, seconds=30)
    provider = FakeProvider({"AAPL": (
        candle(start), candle(start + timedelta(minutes=1)),
        candle(start + timedelta(minutes=2)),
    )})
    service = HistoricalDataService(provider, CandleValidator(), cache_ttl_seconds=30)
    result = service.get_history(["AAPL"], Timeframe(1, "minute"), start, decision)
    assert [bar.timestamp for bar in result["AAPL"]] == [start, start + timedelta(minutes=1)]
    service.get_history(["AAPL"], Timeframe(1, "minute"), start, decision)
    assert len(provider.calls) == 1


def test_history_requires_aware_boundaries() -> None:
    service = HistoricalDataService(FakeProvider({}), CandleValidator())
    with pytest.raises(ValueError, match="timezone-aware"):
        service.get_history(["AAPL"], Timeframe(1, "minute"), datetime(2026, 11, 2),
                            datetime(2026, 11, 2, 1, tzinfo=UTC))


def test_historical_cache_can_be_invalidated() -> None:
    start = datetime(2026, 11, 2, 14, 30, tzinfo=UTC)
    decision = start + timedelta(minutes=3)
    provider = FakeProvider({"AAPL": (candle(start),)})
    service = HistoricalDataService(provider, CandleValidator(), cache_ttl_seconds=30)
    service.get_history(["AAPL"], Timeframe(1, "minute"), start, decision)
    service.invalidate()
    service.get_history(["AAPL"], Timeframe(1, "minute"), start, decision)
    assert len(provider.calls) == 2


def test_market_calendar_resolves_dst_aware_sessions() -> None:
    calendar = ExchangeMarketCalendar()
    winter = calendar.session_for(date(2026, 1, 5))
    summer = calendar.session_for(date(2026, 7, 6))
    assert winter is not None and summer is not None
    assert winter.opens_at.hour == 14
    assert summer.opens_at.hour == 13
    assert calendar.session_for(date(2026, 7, 4)) is None


def test_daily_candle_is_not_available_until_session_close() -> None:
    start = datetime(2026, 11, 2, 14, 30, tzinfo=UTC)
    decision = start + timedelta(hours=1)
    provider = FakeProvider({"AAPL": (candle(start),)})
    service = HistoricalDataService(provider, CandleValidator())
    result = service.get_history(["AAPL"], Timeframe(1, "day"), start, decision)
    assert result["AAPL"] == ()


def test_alpaca_provider_retries_api_failure_and_normalizes_bars() -> None:
    from types import SimpleNamespace
    from config.settings import BrokerSettings, MarketDataSettings
    from data.alpaca import AlpacaMarketDataProvider

    class Client:
        attempts = 0
        request = None
        def get_stock_bars(self, request):
            self.attempts += 1
            self.request = request
            if self.attempts == 1:
                raise ConnectionError("temporary")
            row = SimpleNamespace(timestamp=datetime(2026, 11, 2, 14, 30, tzinfo=UTC),
                                  open=10, high=12, low=9, close=11, volume=100)
            return SimpleNamespace(data={"AAPL": [row]})

    client = Client()
    provider = AlpacaMarketDataProvider(BrokerSettings("key", "secret"),
        MarketDataSettings(retry_attempts=2, retry_base_delay_seconds=0),
        historical_client=client)
    start = datetime(2026, 11, 2, 14, 30, tzinfo=UTC)
    result = provider.fetch_bars(["aapl"], Timeframe(1, "minute"), start, start + timedelta(minutes=2))
    assert client.attempts == 2
    assert result["AAPL"][0].close == 11
    assert client.request.symbol_or_symbols == ["AAPL"]


def test_alpaca_provider_rejects_naive_request_times() -> None:
    from config.settings import BrokerSettings
    from data.alpaca import AlpacaMarketDataProvider
    provider = AlpacaMarketDataProvider(BrokerSettings("key", "secret"), historical_client=object())
    with pytest.raises(ValueError, match="timezone-aware"):
        provider.fetch_bars(["AAPL"], Timeframe(1, "minute"), datetime(2026, 11, 2),
                            datetime(2026, 11, 2, 1, tzinfo=UTC))


def test_alpaca_latest_prices_are_normalized() -> None:
    from types import SimpleNamespace
    from config.settings import BrokerSettings
    from data.alpaca import AlpacaMarketDataProvider

    class Client:
        def get_stock_latest_trade(self, request):
            trade = SimpleNamespace(price=42.5, timestamp=datetime(2026, 11, 2, 14, 30, tzinfo=UTC))
            return SimpleNamespace(data={"AAPL": trade})

    provider = AlpacaMarketDataProvider(BrokerSettings("key", "secret"), historical_client=Client())
    result = provider.latest_prices(["aapl"])
    assert result["AAPL"].price == 42.5
    assert result["AAPL"].symbol == "AAPL"


def test_alpaca_stream_subscribes_normalized_symbols_and_stops() -> None:
    import asyncio
    from types import SimpleNamespace
    from config.settings import BrokerSettings
    from data.alpaca import AlpacaMarketDataProvider

    class FakeStream:
        def __init__(self, *args, **kwargs):
            self.handler = None
            self.symbols = ()
            self.stopped = False
        def subscribe_bars(self, handler, *symbols):
            self.handler, self.symbols = handler, symbols
        def run(self):
            row = SimpleNamespace(symbol="AAPL", timestamp=datetime(2026, 11, 2, 14, 30, tzinfo=UTC),
                                  open=10, high=11, low=9, close=10.5, volume=20)
            asyncio.run(self.handler(row))
        def stop(self):
            self.stopped = True

    stream = FakeStream()
    provider = AlpacaMarketDataProvider(BrokerSettings("key", "secret"),
        historical_client=object(), stream_factory=lambda *args, **kwargs: stream)
    seen = []
    provider.stream_bars(["aapl"], seen.append)
    assert stream.symbols == ("AAPL",)
    assert seen[0].close == 10.5
    provider.stop_stream()
    assert stream.stopped


def test_history_checks_for_missing_session_candles_by_default() -> None:
    calendar = ExchangeMarketCalendar()
    session = calendar.session_for(date(2026, 11, 2))
    assert session is not None
    first = session.opens_at
    provider = FakeProvider({"AAPL": (candle(first), candle(first + timedelta(minutes=2)))})
    service = HistoricalDataService(provider, CandleValidator(), calendar=calendar)
    with pytest.raises(DataQualityError, match="Missing candle"):
        service.get_history(["AAPL"], Timeframe(1, "minute"), first,
                            first + timedelta(minutes=3))


def test_alpaca_timeout_is_applied_to_sdk_http_requests(monkeypatch) -> None:
    import requests
    from config.settings import BrokerSettings, MarketDataSettings
    from data.alpaca import AlpacaMarketDataProvider

    seen = {}
    def fake_request(session, method, url, **kwargs):
        seen.update(kwargs)
        return object()
    monkeypatch.setattr(requests.Session, "request", fake_request)
    provider = AlpacaMarketDataProvider(BrokerSettings("key", "secret"),
        MarketDataSettings(request_timeout_seconds=4.25))
    provider._client._session.request("GET", "https://example.invalid")
    assert seen["timeout"] == 4.25


def test_history_does_not_fabricate_a_missing_symbol() -> None:
    start = datetime(2026, 11, 2, 14, 30, tzinfo=UTC)
    provider = FakeProvider({"AAPL": (candle(start),)})
    result = HistoricalDataService(provider, CandleValidator()).get_history(
        ["AAPL", "MSFT"], Timeframe(1, "minute"), start, start + timedelta(minutes=2)
    )
    assert result["MSFT"] == ()


def test_history_cache_result_cannot_be_mutated_by_consumer() -> None:
    start = datetime(2026, 11, 2, 14, 30, tzinfo=UTC)
    service = HistoricalDataService(FakeProvider({"AAPL": (candle(start),)}), CandleValidator())
    result = service.get_history(["AAPL"], Timeframe(1, "minute"), start,
                                start + timedelta(minutes=2))
    with pytest.raises(TypeError):
        result["MSFT"] = ()  # type: ignore[index]
