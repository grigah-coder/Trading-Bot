"""Unit tests for configuration management (Phase 2)."""

import os

import pytest

from trading_bot.config.settings import (
    BrokerMode,
    Settings,
    TradingMode,
    _load_dotenv,
    load_settings,
)


def test_defaults_are_paper_only():
    settings = Settings()
    assert settings.trading_mode == TradingMode.PAPER
    assert settings.live_trading_enabled is False
    assert settings.alpaca_paper is True
    assert settings.is_paper() is True


def test_live_trading_enabled_is_rejected():
    with pytest.raises(ValueError, match="LIVE_TRADING_ENABLED"):
        Settings(live_trading_enabled=True)


def test_trading_mode_live_is_rejected():
    with pytest.raises(ValueError, match="TRADING_MODE=live"):
        Settings(trading_mode=TradingMode.LIVE)


def test_alpaca_paper_false_is_rejected():
    with pytest.raises(ValueError, match="paper trading is required"):
        Settings(alpaca_paper=False)


def test_stop_band_validation():
    with pytest.raises(ValueError, match="STOP_LOSS_MIN_PCT must not exceed"):
        Settings(stop_loss_min_pct=0.02, stop_loss_max_pct=0.01)


def test_env_file_is_not_overwritten_by_existing_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("ALPACA_API_KEY=from_file\nALPACA_DATA_FEED=sip\n", encoding="utf-8")
    monkeypatch.setenv("ALPACA_API_KEY", "from_env")
    applied = _load_dotenv(env_file)
    assert "ALPACA_API_KEY" not in applied  # existing env wins
    assert os.environ["ALPACA_API_KEY"] == "from_env"
    # key not in env gets loaded from file
    assert os.environ["ALPACA_DATA_FEED"] == "sip"


def test_load_settings_uses_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key123")
    monkeypatch.setenv("ALPACA_API_SECRET", "secret456")
    monkeypatch.setenv("WATCHLIST", "AAPL,TSLA")
    monkeypatch.setenv("MAX_CONCURRENT_POSITIONS", "5")
    settings = load_settings()
    assert settings.alpaca_api_key == "key123"
    assert settings.alpaca_api_secret == "secret456"
    assert settings.watchlist == ["AAPL", "TSLA"]
    assert settings.max_concurrent_positions == 5


def test_load_settings_defaults_when_no_env(monkeypatch):
    # Clear relevant env
    for k in ["ALPACA_API_KEY", "ALPACA_API_SECRET", "WATCHLIST"]:
        monkeypatch.delenv(k, raising=False)
    settings = load_settings()
    assert settings.is_paper() is True
    assert settings.watchlist == [
        "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "AMD", "QQQ",
    ]


def test_feed_validation():
    with pytest.raises(ValueError, match="must be 'iex' or 'sip'"):
        Settings(alpaca_data_feed="otc")


def test_market_open_validation():
    # Non-zero-padded times are accepted and normalized.
    assert Settings(market_open_time="9:30").market_open_time == "09:30"
    assert Settings(market_open_time="9:3").market_open_time == "09:03"
    with pytest.raises(ValueError):
        Settings(market_open_time="25:00")
    with pytest.raises(ValueError):
        Settings(market_open_time="abc")
    with pytest.raises(ValueError):
        Settings(market_open_time="9:30:00")


def test_broker_mode_enum():
    assert BrokerMode.PAPER == "paper"
    assert BrokerMode.LIVE == "live"
