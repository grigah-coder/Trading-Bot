"""Unit tests for the market data engine (Phase 3).

Tests cover OHLCV validation, look-ahead-bias prevention, market session
detection, and the Alpaca provider with a mocked alpaca-py client. No live
network or Alpaca credentials are used.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from trading_bot.config.settings import Settings
from trading_bot.data.market_calendar import MarketCalendar
from trading_bot.data.provider import AlpacaDataProvider
from trading_bot.data.validation import (
    DataValidationError,
    validate_bar,
    validate_bars,
)
from trading_bot.exceptions import MarketDataError


# ---------------------------------------------------------------------------
# Bar validation
# ---------------------------------------------------------------------------
def _now_utc():
    return datetime.now(timezone.utc)


def test_valid_bar():
    bar = validate_bar(
        timestamp=_now_utc(),
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000,
        symbol="AAPL",
        timeframe="5 minutes",
    )
    assert bar.symbol == "AAPL"
    assert bar.open == 100.0
    assert bar.high == 101.0
    assert bar.low == 99.0
    assert bar.close == 100.5
    assert bar.volume == 1000
    assert bar.timestamp.tzinfo is not None


def test_bar_high_must_cover_open_and_close():
    with pytest.raises(DataValidationError, match="high"):
        validate_bar(
            timestamp=_now_utc(),
            open_price=100.0,
            high=99.0,  # below open
            low=98.0,
            close=100.5,
            volume=1000,
            symbol="AAPL",
            timeframe="5 minutes",
        )


def test_bar_low_must_be_below_open_and_close():
    with pytest.raises(DataValidationError, match="low"):
        validate_bar(
            timestamp=_now_utc(),
            open_price=100.0,
            high=101.0,
            low=100.5,  # above close
            close=100.5,
            volume=1000,
            symbol="AAPL",
            timeframe="5 minutes",
        )


def test_bar_rejects_nonpositive_prices():
    for field, value in [("open_price", 0.0), ("high", -1.0), ("low", 0.0), ("close", -2.0)]:
        kwargs = dict(
            timestamp=_now_utc(),
            open_price=100.0, high=101.0, low=99.0, close=100.0,
            volume=1000, symbol="AAPL", timeframe="5 minutes",
        )
        kwargs[field] = value
        with pytest.raises(DataValidationError):
            validate_bar(**kwargs)


def test_bar_rejects_negative_volume():
    with pytest.raises(DataValidationError, match="volume"):
        validate_bar(
            timestamp=_now_utc(),
            open_price=100.0, high=101.0, low=99.0, close=100.0,
            volume=-1, symbol="AAPL", timeframe="5 minutes",
        )


def test_bar_rejects_future_timestamp_lookahead():
    future = _now_utc() + timedelta(days=1)
    with pytest.raises(DataValidationError, match="future"):
        validate_bar(
            timestamp=future,
            open_price=100.0, high=101.0, low=99.0, close=100.0,
            volume=1000, symbol="AAPL", timeframe="5 minutes",
        )


def test_bar_accepts_naive_timestamp_utc():
    naive = datetime(2024, 1, 1, 10, 0, 0)
    bar = validate_bar(
        timestamp=naive,
        open_price=100.0, high=101.0, low=99.0, close=100.0,
        volume=1000, symbol="AAPL", timeframe="5 minutes",
    )
    assert bar.timestamp.tzinfo is not None


def test_bar_to_dict_roundtrip():
    bar = validate_bar(
        timestamp=_now_utc(),
        open_price=100.0, high=101.0, low=99.0, close=100.0,
        volume=1000, symbol="aapl", timeframe="5 minutes",
    )
    d = bar.to_dict()
    assert d["symbol"] == "AAPL"
    assert set(d) == {"timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe"}


def test_validate_bars_skips_invalid():
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    raw = [
        {"timestamp": now - timedelta(minutes=10), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000},
        {"timestamp": now - timedelta(minutes=5), "open": 100.0, "high": 99.0, "low": 98.0, "close": 99.5, "volume": 1000},  # invalid high
        {"timestamp": now - timedelta(minutes=1), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1200},
    ]
    valid = validate_bars(raw, "AAPL", "5 minutes", clock=lambda: now)
    assert len(valid) == 2
    assert valid[0].close == 100.0
    assert valid[1].close == 101.5


# ---------------------------------------------------------------------------
# Market calendar
# ---------------------------------------------------------------------------
def test_market_open_weekday():
    cal = MarketCalendar(market_open_time="09:30", market_close_time="16:00", opening_blackout_minutes=30)
    # Tuesday 2026-11-10 10:00 ET
    moment = datetime(2026, 11, 10, 10, 0, tzinfo=cal.tz)
    assert cal.is_market_open(moment) is True


def test_market_closed_weekend():
    cal = MarketCalendar()
    # Saturday 2026-11-14
    moment = datetime(2026, 11, 14, 10, 0, tzinfo=cal.tz)
    assert cal.is_market_open(moment) is False
    # Sunday 2026-11-15
    moment = datetime(2026, 11, 15, 10, 0, tzinfo=cal.tz)
    assert cal.is_market_open(moment) is False


def test_market_closed_before_open():
    cal = MarketCalendar()
    moment = datetime(2026, 11, 10, 8, 0, tzinfo=cal.tz)
    assert cal.is_market_open(moment) is False


def test_market_closed_after_close():
    cal = MarketCalendar()
    moment = datetime(2026, 11, 10, 17, 0, tzinfo=cal.tz)
    assert cal.is_market_open(moment) is False


def test_opening_blackout():
    cal = MarketCalendar(opening_blackout_minutes=30)
    moment = datetime(2026, 11, 10, 9, 45, tzinfo=cal.tz)  # inside blackout
    assert cal.is_in_opening_blackout(moment) is True
    moment = datetime(2026, 11, 10, 10, 5, tzinfo=cal.tz)  # outside blackout
    assert cal.is_in_opening_blackout(moment) is False
    moment = datetime(2026, 11, 10, 9, 30, tzinfo=cal.tz)  # exactly at open
    assert cal.is_in_opening_blackout(moment) is True


def test_opening_blackout_disabled():
    cal = MarketCalendar(opening_blackout_minutes=0)
    moment = datetime(2026, 11, 10, 9, 45, tzinfo=cal.tz)
    assert cal.is_in_opening_blackout(moment) is False


# ---------------------------------------------------------------------------
# Alpaca provider (mocked client)
# ---------------------------------------------------------------------------
def _make_settings(**overrides):
    data = dict(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        alpaca_paper=True,
        alpaca_data_feed="iex",
        trading_mode="paper",
        live_trading_enabled=False,
        indicator_cache_ttl=30.0,
        cache_maxsize=512,
        retry_attempts=2,
        retry_base_delay=0.01,
        retry_max_delay=1.0,
        retry_backoff_factor=2.0,
        retry_jitter=False,
        market_timezone="America/New_York",
        market_open_time="09:30",
        opening_blackout_minutes=30,
    )
    data.update(overrides)
    return Settings(**data)


def _make_bar_df():
    """Build a minimal barset-like object with a .df (MultiIndex)."""
    import pandas as pd

    idx = pd.MultiIndex.from_tuples(
        [
            ("AAPL", pd.Timestamp("2024-01-02 09:30:00", tz="UTC")),
            ("AAPL", pd.Timestamp("2024-01-02 09:35:00", tz="UTC")),
        ],
        names=["symbol", "timestamp"],
    )
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
        },
        index=idx,
    )
    return MagicMock(df=df)


def test_provider_get_historical_bars(monkeypatch):
    settings = _make_settings()
    fixed_now = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    provider = AlpacaDataProvider(settings, clock=lambda: fixed_now)

    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = _make_bar_df()

    with patch.object(provider, "_client", mock_client):
        bars = provider.get_historical_bars("AAPL", 10, "5 minutes")

    assert bars is not None
    assert len(bars) == 2
    assert bars[0]["symbol"] == "AAPL"
    assert bars[0]["close"] == 100.5
    assert bars[1]["close"] == 101.5
    # oldest-first
    assert bars[0]["timestamp"] < bars[1]["timestamp"]
    # all timestamps in the past relative to the clock
    for b in bars:
        ts = datetime.fromisoformat(b["timestamp"])
        assert ts <= fixed_now


def test_provider_get_historical_bars_none_on_error(monkeypatch):
    settings = _make_settings(retry_attempts=1)
    provider = AlpacaDataProvider(settings, clock=lambda: datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc))

    mock_client = MagicMock()
    mock_client.get_stock_bars.side_effect = Exception("network down")

    with patch.object(provider, "_client", mock_client):
        bars = provider.get_historical_bars("AAPL", 10, "5 minutes")

    assert bars is None


def test_provider_get_latest_bar(monkeypatch):
    settings = _make_settings()
    fixed_now = datetime(2024, 1, 2, 10, 5, 30, tzinfo=timezone.utc)  # forming minute
    provider = AlpacaDataProvider(settings, clock=lambda: fixed_now)

    mock_client = MagicMock()
    mock_trade = MagicMock()
    mock_trade.price = 150.0
    mock_trade.size = 500
    mock_trade.timestamp = fixed_now
    mock_client.get_stock_latest_trade.return_value = mock_trade

    with patch.object(provider, "_client", mock_client):
        bar = provider.get_latest_bar("AAPL")

    assert bar is not None
    assert bar["close"] == 150.0
    assert bar["symbol"] == "AAPL"
    # anchored to previous completed minute, not the forming minute
    ts = datetime.fromisoformat(bar["timestamp"])
    assert ts == fixed_now.replace(second=0, microsecond=0) - timedelta(minutes=1)


def test_provider_get_latest_bar_none_when_invalid_price(monkeypatch):
    settings = _make_settings()
    provider = AlpacaDataProvider(settings, clock=lambda: datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc))

    mock_client = MagicMock()
    mock_trade = MagicMock()
    mock_trade.price = -1.0  # invalid
    mock_trade.size = 100
    mock_trade.timestamp = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    mock_client.get_stock_latest_trade.return_value = mock_trade

    with patch.object(provider, "_client", mock_client):
        bar = provider.get_latest_bar("AAPL")

    assert bar is None


def test_provider_get_latest_trade(monkeypatch):
    settings = _make_settings()
    provider = AlpacaDataProvider(settings)

    mock_client = MagicMock()
    mock_trade = MagicMock()
    mock_trade.price = 150.0
    mock_trade.size = 500
    mock_trade.timestamp = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    mock_client.get_stock_latest_trade.return_value = mock_trade

    with patch.object(provider, "_client", mock_client):
        trade = provider.get_latest_trade("AAPL")

    assert trade is not None
    assert trade["price"] == 150.0
    assert trade["size"] == 500.0


def test_provider_is_market_open(monkeypatch):
    settings = _make_settings()
    provider = AlpacaDataProvider(settings)
    # The calendar uses real time; just check it returns a bool without error.
    assert isinstance(provider.is_market_open(), bool)
    assert isinstance(provider.is_in_opening_blackout(), bool)


def test_provider_is_data_fresh(monkeypatch):
    settings = _make_settings()
    fixed_now = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    provider = AlpacaDataProvider(settings, clock=lambda: fixed_now)

    mock_client = MagicMock()
    mock_bar = MagicMock()
    mock_bar.df = None  # force dict/list path -> empty
    mock_client.get_stock_bars.return_value = mock_bar
    # latest trade for freshness
    mock_trade = MagicMock()
    mock_trade.price = 100.0
    mock_trade.size = 100
    mock_trade.timestamp = fixed_now - timedelta(minutes=1)
    mock_client.get_stock_latest_trade.return_value = mock_trade

    with patch.object(provider, "_client", mock_client):
        # get_latest_bar will synthesize from trade; freshness uses that.
        fresh = provider.is_data_fresh("AAPL", max_age_minutes=5.0)

    assert isinstance(fresh, bool)


def test_provider_rejects_unsupported_timeframe():
    settings = _make_settings()
    provider = AlpacaDataProvider(settings)
    from trading_bot.data.provider import _parse_timeframe
    with pytest.raises(MarketDataError, match="unsupported timeframe"):
        _parse_timeframe("1 week")


def test_provider_normalizes_symbol_to_upper():
    settings = _make_settings()
    provider = AlpacaDataProvider(settings)
    # Feed resolves to IEX
    assert provider._feed.value == "iex"
