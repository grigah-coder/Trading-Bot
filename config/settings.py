"""Strongly validated, environment-driven application configuration."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.exceptions import ConfigurationError


def _boolean(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


def _integer(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _number(name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ConfigurationError(f"{name} must be finite")
    return parsed


@dataclass(frozen=True, slots=True)
class BrokerSettings:
    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)
    paper: bool = True
    request_timeout_seconds: float = 10.0
    retry_attempts: int = 3
    retry_base_delay_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not isinstance(self.api_secret, str):
            raise ConfigurationError("Alpaca credentials must be strings")
        if not isinstance(self.paper, bool):
            raise ConfigurationError("ALPACA_PAPER must be a boolean")
        if bool(self.api_key) != bool(self.api_secret):
            raise ConfigurationError("Alpaca API key and secret must be supplied together")
        if not self.paper:
            raise ConfigurationError("Only Alpaca paper mode is supported in this phase")
        if (not isinstance(self.request_timeout_seconds, (int, float))
                or isinstance(self.request_timeout_seconds, bool)
                or not math.isfinite(self.request_timeout_seconds)
                or self.request_timeout_seconds <= 0):
            raise ConfigurationError("ALPACA_REQUEST_TIMEOUT_SECONDS must be positive and finite")
        if (not isinstance(self.retry_attempts, int) or isinstance(self.retry_attempts, bool)
                or self.retry_attempts <= 0):
            raise ConfigurationError("ALPACA_RETRY_ATTEMPTS must be positive")
        if (not isinstance(self.retry_base_delay_seconds, (int, float))
                or isinstance(self.retry_base_delay_seconds, bool)
                or not math.isfinite(self.retry_base_delay_seconds)
                or self.retry_base_delay_seconds < 0):
            raise ConfigurationError("ALPACA_RETRY_BASE_DELAY_SECONDS must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class TradingSettings:
    calendar: str = "XNYS"
    timezone: str = "America/New_York"
    evaluation_interval_seconds: int = 60
    scheduler_poll_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.calendar, str) or not isinstance(self.timezone, str):
            raise ConfigurationError("Trading calendar and timezone must be strings")
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ConfigurationError("TRADING_TIMEZONE is not a valid timezone") from exc
        if not self.calendar.strip():
            raise ConfigurationError("TRADING_CALENDAR cannot be empty")
        if (
            not isinstance(self.evaluation_interval_seconds, int)
            or isinstance(self.evaluation_interval_seconds, bool)
            or self.evaluation_interval_seconds <= 0
        ):
            raise ConfigurationError("TRADING_EVALUATION_INTERVAL_SECONDS must be positive")
        if (
            not isinstance(self.scheduler_poll_seconds, (int, float))
            or isinstance(self.scheduler_poll_seconds, bool)
            or not math.isfinite(self.scheduler_poll_seconds)
            or self.scheduler_poll_seconds <= 0
        ):
            raise ConfigurationError("TRADING_SCHEDULER_POLL_SECONDS must be positive")


@dataclass(frozen=True, slots=True)
class MarketDataSettings:
    feed: str = "iex"
    request_timeout_seconds: float = 10.0
    stale_after_seconds: float = 60.0
    cache_ttl_seconds: float = 30.0
    retry_attempts: int = 3
    retry_base_delay_seconds: float = 0.5
    stream_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.feed, str) or self.feed.lower() not in {"iex", "sip", "delayed_sip", "otc", "boats", "overnight"}:
            raise ConfigurationError("MARKET_DATA_FEED is not a supported Alpaca feed")
        object.__setattr__(self, "feed", self.feed.lower())
        for name, value in (("MARKET_DATA_REQUEST_TIMEOUT_SECONDS", self.request_timeout_seconds),
                            ("MARKET_DATA_STALE_AFTER_SECONDS", self.stale_after_seconds),
                            ("MARKET_DATA_CACHE_TTL_SECONDS", self.cache_ttl_seconds),
                            ("MARKET_DATA_STREAM_TIMEOUT_SECONDS", self.stream_timeout_seconds)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be positive and finite")
        delay = self.retry_base_delay_seconds
        if not isinstance(delay, (int, float)) or isinstance(delay, bool) or not math.isfinite(delay) or delay < 0:
            raise ConfigurationError("MARKET_DATA_RETRY_BASE_DELAY_SECONDS must be non-negative and finite")
        if not isinstance(self.retry_attempts, int) or isinstance(self.retry_attempts, bool) or self.retry_attempts <= 0:
            raise ConfigurationError("MARKET_DATA_RETRY_ATTEMPTS must be positive")


@dataclass(frozen=True, slots=True)
class StrategySettings:
    name: str = "unconfigured"
    parameters: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("STRATEGY_NAME cannot be empty")
        if not isinstance(self.parameters, Mapping):
            raise ConfigurationError("Strategy parameters must be a mapping")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.parameters.items()
        ):
            raise ConfigurationError("Strategy parameters must map strings to strings")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class RiskSettings:
    """Explicit, validated portfolio and market risk boundaries.

    Defaults are documented in risk/README.md and mirrored in .env.example.
    """

    max_position_fraction: float = 0.03
    daily_loss_limit_fraction: float = 0.0025
    max_open_positions: int = 2
    max_risk_per_trade_fraction: float = 0.005
    max_drawdown_fraction: float = 0.05
    max_portfolio_exposure_fraction: float = 0.25
    max_order_notional_fraction: float = 0.03
    max_sector_exposure_fraction: float = 0.15
    minimum_average_daily_dollar_volume: float = 1_000_000.0
    maximum_spread_fraction: float = 0.005
    maximum_daily_volatility_fraction: float = 0.08
    market_data_max_age_seconds: float = 60.0
    regular_session_only: bool = True

    def __post_init__(self) -> None:
        fractions = {
            "RISK_MAX_POSITION_FRACTION": self.max_position_fraction,
            "RISK_DAILY_LOSS_LIMIT_FRACTION": self.daily_loss_limit_fraction,
            "RISK_MAX_RISK_PER_TRADE_FRACTION": self.max_risk_per_trade_fraction,
            "RISK_MAX_DRAWDOWN_FRACTION": self.max_drawdown_fraction,
            "RISK_MAX_PORTFOLIO_EXPOSURE_FRACTION": self.max_portfolio_exposure_fraction,
            "RISK_MAX_ORDER_NOTIONAL_FRACTION": self.max_order_notional_fraction,
            "RISK_MAX_SECTOR_EXPOSURE_FRACTION": self.max_sector_exposure_fraction,
            "RISK_MAXIMUM_SPREAD_FRACTION": self.maximum_spread_fraction,
            "RISK_MAXIMUM_DAILY_VOLATILITY_FRACTION": self.maximum_daily_volatility_fraction,
        }
        for name, value in fractions.items():
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or not 0 < value <= 1):
                raise ConfigurationError(f"{name} must be greater than 0 and at most 1")
        if (not isinstance(self.max_open_positions, int) or isinstance(self.max_open_positions, bool)
                or self.max_open_positions <= 0):
            raise ConfigurationError("RISK_MAX_OPEN_POSITIONS must be positive")
        for name, value in (("RISK_MINIMUM_AVERAGE_DAILY_DOLLAR_VOLUME", self.minimum_average_daily_dollar_volume),
                            ("RISK_MARKET_DATA_MAX_AGE_SECONDS", self.market_data_max_age_seconds)):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0):
                raise ConfigurationError(f"{name} must be positive and finite")
        if not isinstance(self.regular_session_only, bool):
            raise ConfigurationError("RISK_REGULAR_SESSION_ONLY must be a boolean")


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    url: str = field(default="sqlite:///data/trading_bot.sqlite", repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.url, str):
            raise ConfigurationError("DATABASE_URL must be a string")
        parsed = urlparse(self.url)
        if not parsed.scheme:
            raise ConfigurationError("DATABASE_URL must include a scheme")


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    level: str = "INFO"
    file: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.level, str):
            raise ConfigurationError("LOG_LEVEL must be a string")
        if self.file is not None and not isinstance(self.file, Path):
            raise ConfigurationError("LOG_FILE must be a filesystem path")
        normalized = self.level.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigurationError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        object.__setattr__(self, "level", normalized)


@dataclass(frozen=True, slots=True)
class Settings:
    broker: BrokerSettings = field(default_factory=BrokerSettings)
    trading: TradingSettings = field(default_factory=TradingSettings)
    market_data: MarketDataSettings = field(default_factory=MarketDataSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    def __post_init__(self) -> None:
        domains = (
            ("broker", self.broker, BrokerSettings),
            ("trading", self.trading, TradingSettings),
            ("market_data", self.market_data, MarketDataSettings),
            ("strategy", self.strategy, StrategySettings),
            ("risk", self.risk, RiskSettings),
            ("database", self.database, DatabaseSettings),
            ("logging", self.logging, LoggingSettings),
        )
        for name, value, expected_type in domains:
            if not isinstance(value, expected_type):
                raise ConfigurationError(f"Settings.{name} has the wrong settings type")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Load and validate settings without mutating process-wide state."""
        source = os.environ if environ is None else environ
        paper = _boolean("ALPACA_PAPER", source.get("ALPACA_PAPER", "true"))
        log_file = source.get("LOG_FILE", "").strip()
        strategy_parameters = {
            key.removeprefix("STRATEGY_PARAM_").lower(): value
            for key, value in source.items()
            if key.startswith("STRATEGY_PARAM_")
        }
        if any(not key for key in strategy_parameters):
            raise ConfigurationError("Strategy parameter names cannot be empty")
        return cls(
            broker=BrokerSettings(
                api_key=source.get("ALPACA_API_KEY", "").strip(),
                api_secret=source.get("ALPACA_API_SECRET", "").strip(),
                paper=paper,
                request_timeout_seconds=_number("ALPACA_REQUEST_TIMEOUT_SECONDS", source.get("ALPACA_REQUEST_TIMEOUT_SECONDS", "10")),
                retry_attempts=_integer("ALPACA_RETRY_ATTEMPTS", source.get("ALPACA_RETRY_ATTEMPTS", "3")),
                retry_base_delay_seconds=_number("ALPACA_RETRY_BASE_DELAY_SECONDS", source.get("ALPACA_RETRY_BASE_DELAY_SECONDS", "0.5")),
            ),
            trading=TradingSettings(
                calendar=source.get("TRADING_CALENDAR", "XNYS").strip(),
                timezone=source.get("TRADING_TIMEZONE", "America/New_York").strip(),
                evaluation_interval_seconds=_integer(
                    "TRADING_EVALUATION_INTERVAL_SECONDS",
                    source.get("TRADING_EVALUATION_INTERVAL_SECONDS", "60"),
                ),
                scheduler_poll_seconds=_number(
                    "TRADING_SCHEDULER_POLL_SECONDS",
                    source.get("TRADING_SCHEDULER_POLL_SECONDS", "1"),
                ),
            ),
            market_data=MarketDataSettings(
                feed=source.get("MARKET_DATA_FEED", "iex").strip(),
                request_timeout_seconds=_number("MARKET_DATA_REQUEST_TIMEOUT_SECONDS", source.get("MARKET_DATA_REQUEST_TIMEOUT_SECONDS", "10")),
                stale_after_seconds=_number("MARKET_DATA_STALE_AFTER_SECONDS", source.get("MARKET_DATA_STALE_AFTER_SECONDS", "60")),
                cache_ttl_seconds=_number("MARKET_DATA_CACHE_TTL_SECONDS", source.get("MARKET_DATA_CACHE_TTL_SECONDS", "30")),
                retry_attempts=_integer("MARKET_DATA_RETRY_ATTEMPTS", source.get("MARKET_DATA_RETRY_ATTEMPTS", "3")),
                retry_base_delay_seconds=_number("MARKET_DATA_RETRY_BASE_DELAY_SECONDS", source.get("MARKET_DATA_RETRY_BASE_DELAY_SECONDS", "0.5")),
                stream_timeout_seconds=_number("MARKET_DATA_STREAM_TIMEOUT_SECONDS", source.get("MARKET_DATA_STREAM_TIMEOUT_SECONDS", "30")),
            ),
            strategy=StrategySettings(
                name=source.get("STRATEGY_NAME", "unconfigured").strip(),
                parameters=strategy_parameters,
            ),
            risk=RiskSettings(
                max_position_fraction=_number("RISK_MAX_POSITION_FRACTION", source.get("RISK_MAX_POSITION_FRACTION", "0.03")),
                daily_loss_limit_fraction=_number("RISK_DAILY_LOSS_LIMIT_FRACTION", source.get("RISK_DAILY_LOSS_LIMIT_FRACTION", "0.0025")),
                max_open_positions=_integer("RISK_MAX_OPEN_POSITIONS", source.get("RISK_MAX_OPEN_POSITIONS", "2")),
                max_risk_per_trade_fraction=_number("RISK_MAX_RISK_PER_TRADE_FRACTION", source.get("RISK_MAX_RISK_PER_TRADE_FRACTION", "0.005")),
                max_drawdown_fraction=_number("RISK_MAX_DRAWDOWN_FRACTION", source.get("RISK_MAX_DRAWDOWN_FRACTION", "0.05")),
                max_portfolio_exposure_fraction=_number("RISK_MAX_PORTFOLIO_EXPOSURE_FRACTION", source.get("RISK_MAX_PORTFOLIO_EXPOSURE_FRACTION", "0.25")),
                max_order_notional_fraction=_number("RISK_MAX_ORDER_NOTIONAL_FRACTION", source.get("RISK_MAX_ORDER_NOTIONAL_FRACTION", "0.03")),
                max_sector_exposure_fraction=_number("RISK_MAX_SECTOR_EXPOSURE_FRACTION", source.get("RISK_MAX_SECTOR_EXPOSURE_FRACTION", "0.15")),
                minimum_average_daily_dollar_volume=_number("RISK_MINIMUM_AVERAGE_DAILY_DOLLAR_VOLUME", source.get("RISK_MINIMUM_AVERAGE_DAILY_DOLLAR_VOLUME", "1000000")),
                maximum_spread_fraction=_number("RISK_MAXIMUM_SPREAD_FRACTION", source.get("RISK_MAXIMUM_SPREAD_FRACTION", "0.005")),
                maximum_daily_volatility_fraction=_number("RISK_MAXIMUM_DAILY_VOLATILITY_FRACTION", source.get("RISK_MAXIMUM_DAILY_VOLATILITY_FRACTION", "0.08")),
                market_data_max_age_seconds=_number("RISK_MARKET_DATA_MAX_AGE_SECONDS", source.get("RISK_MARKET_DATA_MAX_AGE_SECONDS", "60")),
                regular_session_only=_boolean("RISK_REGULAR_SESSION_ONLY", source.get("RISK_REGULAR_SESSION_ONLY", "true")),
            ),
            database=DatabaseSettings(
                url=source.get("DATABASE_URL", "sqlite:///data/trading_bot.sqlite").strip()
            ),
            logging=LoggingSettings(
                level=source.get("LOG_LEVEL", "INFO").strip(),
                file=Path(log_file) if log_file else None,
            ),
        )
