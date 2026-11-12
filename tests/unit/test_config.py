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


def test_pct_values_are_fractions_between_zero_and_one():
    """Defaults and validators enforce the fraction convention."""
    settings = Settings()
    assert 0.0 < settings.stop_loss_pct < 1.0
    assert 0.0 < settings.take_profit_pct < 1.0
    assert settings.stop_loss_min_pct <= settings.stop_loss_pct
    assert settings.stop_loss_pct <= settings.stop_loss_max_pct

    for field in ("stop_loss_min_pct", "stop_loss_max_pct",
                  "stop_loss_pct", "take_profit_pct"):
        with pytest.raises(ValueError):
            Settings(**{field: 1.0})
        with pytest.raises(ValueError):
            Settings(**{field: 0.0})


def test_env_file_is_not_overwritten_by_existing_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("ALPACA_API_KEY=from_file\nALPACA_DATA_FEED=sip\n", encoding="utf-8")
    monkeypatch.setenv("ALPACA_API_KEY", "from_env")
    # Ensure the file value can load: delete any polluted env from prior tests.
    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
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
    # Keep the test hermetic: do not inherit pct values from the host env.
    for key in ("STOP_LOSS_MIN_PCT", "STOP_LOSS_MAX_PCT",
                "STOP_LOSS_PCT", "TAKE_PROFIT_PCT"):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings()
    assert settings.alpaca_api_key == "key123"
    assert settings.alpaca_api_secret == "secret456"
    assert settings.watchlist == ["AAPL", "TSLA"]
    assert settings.max_concurrent_positions == 5


def test_load_settings_defaults_when_no_env(monkeypatch):
    # Clear relevant env
    for k in ["ALPACA_API_KEY", "ALPACA_API_SECRET", "WATCHLIST",
              "STOP_LOSS_MIN_PCT", "STOP_LOSS_MAX_PCT",
              "STOP_LOSS_PCT", "TAKE_PROFIT_PCT"]:
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


def test_default_dotenv_path_is_the_repo_root():
    """The default ``.env`` is resolved at the repository root."""
    from pathlib import Path

    from trading_bot.config import settings as settings_module

    assert settings_module._REPO_ROOT == Path(
        settings_module.__file__
    ).resolve().parents[3]
    assert (settings_module._REPO_ROOT / "pyproject.toml").exists()


def test_load_settings_reads_explicit_tmp_env_file(monkeypatch, tmp_path):
    """An explicit env file is loaded without touching the repo ``.env``."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ALPACA_API_KEY=tmp-key\n"
        "ALPACA_API_SECRET=tmp-secret\n"
        "MAX_CONCURRENT_POSITIONS=7\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT_POSITIONS", raising=False)
    # The host shell may export legacy multiplier-style values; keep this
    # test independent of the ambient environment.
    for key in ("STOP_LOSS_PCT", "TAKE_PROFIT_PCT",
                "STOP_LOSS_MIN_PCT", "STOP_LOSS_MAX_PCT"):
        monkeypatch.delenv(key, raising=False)

    settings = load_settings(env_file=env_file)
    assert settings.alpaca_api_key == "tmp-key"
    assert settings.alpaca_api_secret == "tmp-secret"
    assert settings.max_concurrent_positions == 7


def test_logging_helpers_use_module_names():
    """``get_logger`` is a name-based helper, not a config entry point."""
    from trading_bot.logging_config import configure_logging, get_logger

    assert get_logger("trading_bot.engine").name == "trading_bot.engine"
    # A module name must never be accepted as the positional ``level``.
    with pytest.raises(TypeError):
        configure_logging("trading_bot.engine")


def test_log_file_default_matches_env_example():
    """The code default for ``LOG_FILE`` matches ``.env.example``."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    env_example = (repo / ".env.example").read_text(encoding="utf-8")
    assert f"LOG_FILE={Settings().log_file}" in env_example
