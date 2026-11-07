import pytest

from config.settings import (
    BrokerSettings,
    ConfigurationError,
    DatabaseSettings,
    LoggingSettings,
    RiskSettings,
    Settings,
    StrategySettings,
    TradingSettings,
)


def test_settings_are_split_into_validated_domains() -> None:
    settings = Settings.from_env(
        {
            "ALPACA_API_KEY": "key",
            "ALPACA_API_SECRET": "secret",
            "TRADING_TIMEZONE": "America/Los_Angeles",
            "STRATEGY_NAME": "example",
            "RISK_MAX_OPEN_POSITIONS": "3",
            "DATABASE_URL": "sqlite:///trades.db",
            "LOG_LEVEL": "debug",
        }
    )

    assert isinstance(settings.broker, BrokerSettings)
    assert isinstance(settings.trading, TradingSettings)
    assert isinstance(settings.strategy, StrategySettings)
    assert isinstance(settings.risk, RiskSettings)
    assert isinstance(settings.database, DatabaseSettings)
    assert isinstance(settings.logging, LoggingSettings)
    assert settings.broker.paper is True
    assert settings.trading.timezone == "America/Los_Angeles"
    assert settings.strategy.name == "example"
    assert settings.risk.max_open_positions == 3
    assert settings.database.url == "sqlite:///trades.db"
    assert settings.logging.level == "DEBUG"


@pytest.mark.parametrize(
    "environment",
    [
        {"ALPACA_PAPER": "false"},
        {"TRADING_TIMEZONE": "Mars/Olympus"},
        {"TRADING_EVALUATION_INTERVAL_SECONDS": "0"},
        {"RISK_MAX_OPEN_POSITIONS": "0"},
        {"RISK_MAX_POSITION_FRACTION": "1.5"},
        {"LOG_LEVEL": "TRACE"},
    ],
)
def test_invalid_environment_configuration_is_rejected(environment: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env(environment)


def test_direct_invalid_risk_settings_are_rejected() -> None:
    with pytest.raises(ConfigurationError):
        RiskSettings(max_position_fraction=0)


def test_secret_values_are_not_in_configuration_repr() -> None:
    settings = Settings.from_env(
        {"ALPACA_API_KEY": "private-key", "ALPACA_API_SECRET": "private-secret"}
    )

    assert "private-key" not in repr(settings)
    assert "private-secret" not in repr(settings)


def test_non_integer_position_limit_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        RiskSettings(max_open_positions=1.5)  # type: ignore[arg-type]


def test_market_data_settings_are_validated_from_environment() -> None:
    from config.settings import MarketDataSettings
    settings = Settings.from_env({"MARKET_DATA_FEED": "sip", "MARKET_DATA_REQUEST_TIMEOUT_SECONDS": "4.5"})
    assert settings.market_data.feed == "sip"
    assert settings.market_data.request_timeout_seconds == 4.5
    assert isinstance(settings.market_data, MarketDataSettings)


def test_strategy_parameters_load_from_environment() -> None:
    settings = Settings.from_env({
        "STRATEGY_NAME": "trend_momentum",
        "STRATEGY_PARAM_FAST_PERIOD": "8",
        "STRATEGY_PARAM_MAXIMUM_ATR_FRACTION": "0.04",
    })
    assert settings.strategy.name == "trend_momentum"
    assert settings.strategy.parameters == {"fast_period": "8", "maximum_atr_fraction": "0.04"}


def test_invalid_market_data_settings_are_rejected() -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env({"MARKET_DATA_REQUEST_TIMEOUT_SECONDS": "0"})


def test_broker_request_reliability_settings_are_validated_from_environment() -> None:
    settings = Settings.from_env({"ALPACA_API_KEY": "PK-key", "ALPACA_API_SECRET": "secret",
                                 "ALPACA_REQUEST_TIMEOUT_SECONDS": "4.5",
                                 "ALPACA_RETRY_ATTEMPTS": "5"})
    assert settings.broker.request_timeout_seconds == 4.5
    assert settings.broker.retry_attempts == 5


def test_invalid_broker_timeout_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env({"ALPACA_REQUEST_TIMEOUT_SECONDS": "0"})


def test_all_documented_risk_limits_load_from_environment() -> None:
    settings = Settings.from_env({
        "RISK_MAX_RISK_PER_TRADE_FRACTION": "0.004",
        "RISK_MAX_DRAWDOWN_FRACTION": "0.08",
        "RISK_MAX_PORTFOLIO_EXPOSURE_FRACTION": "0.35",
        "RISK_MAX_ORDER_NOTIONAL_FRACTION": "0.05",
        "RISK_MAX_SECTOR_EXPOSURE_FRACTION": "0.20",
        "RISK_MINIMUM_AVERAGE_DAILY_DOLLAR_VOLUME": "2000000",
        "RISK_MAXIMUM_SPREAD_FRACTION": "0.003",
        "RISK_MAXIMUM_DAILY_VOLATILITY_FRACTION": "0.07",
        "RISK_MARKET_DATA_MAX_AGE_SECONDS": "45",
        "RISK_REGULAR_SESSION_ONLY": "true",
    })
    risk = settings.risk
    assert risk.max_risk_per_trade_fraction == 0.004
    assert risk.max_drawdown_fraction == 0.08
    assert risk.max_portfolio_exposure_fraction == 0.35
    assert risk.max_order_notional_fraction == 0.05
    assert risk.max_sector_exposure_fraction == 0.20
    assert risk.minimum_average_daily_dollar_volume == 2_000_000
    assert risk.maximum_spread_fraction == 0.003
    assert risk.maximum_daily_volatility_fraction == 0.07
    assert risk.market_data_max_age_seconds == 45
    assert risk.regular_session_only


@pytest.mark.parametrize("environment", [
    {"RISK_MAX_RISK_PER_TRADE_FRACTION": "0"},
    {"RISK_MAX_DRAWDOWN_FRACTION": "1.1"},
    {"RISK_MAX_PORTFOLIO_EXPOSURE_FRACTION": "-0.1"},
    {"RISK_MAX_ORDER_NOTIONAL_FRACTION": "nan"},
    {"RISK_MINIMUM_AVERAGE_DAILY_DOLLAR_VOLUME": "-10"},
])
def test_invalid_extended_risk_configuration_prevents_startup(environment: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env(environment)
