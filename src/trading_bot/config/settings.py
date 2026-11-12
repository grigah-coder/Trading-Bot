"""Application settings.

Validated configuration loaded from environment variables (optionally from a
``.env`` file). Every operational value lives here; nothing is hardcoded in
trading logic.

Default operating mode is paper trading only. Live trading is disabled and the
settings validator rejects any attempt to enable it.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

# Repo root: <root>/src/trading_bot/config/settings.py -> parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class BrokerMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


def _load_dotenv(path: Optional[str | Path] = None) -> Dict[str, str]:
    """Load ``KEY=value`` pairs from a ``.env`` file into ``os.environ``.

    Defaults to the repo-root ``.env`` (not ``src/.env``). Existing
    environment variables are never overwritten. Returns the mapping that was
    applied (only keys not already present in the environment).
    """
    if path is None:
        path = _REPO_ROOT / ".env"
    path = Path(path)
    if not path.exists():
        return {}

    before = set(os.environ)
    try:
        load_dotenv(dotenv_path=path, override=False, encoding="utf-8")
    except Exception:  # noqa: BLE001 - unreadable .env is non-fatal
        return {}
    return {
        key: os.environ[key]
        for key in os.environ.keys() - before
    }


def _env_list(value: Optional[str], default: List[str]) -> List[str]:
    if default is None:
        default = []
    if not value:
        return list(default)
    items = [item.strip().upper() for item in value.replace(";", ",").split(",")]
    return [item for item in items if item] or list(default)


class Settings(BaseModel):
    """Validated application configuration.

    All fields are populated from environment variables. Use
    :func:`load_settings` to construct an instance (it also loads ``.env``).
    """

    # ------------------------------------------------------------------
    # Alpaca credentials
    # ------------------------------------------------------------------
    alpaca_api_key: str = Field(default="")
    alpaca_api_secret: str = Field(default="")
    alpaca_paper: bool = Field(default=True)
    alpaca_data_feed: str = Field(default="iex")

    # ------------------------------------------------------------------
    # Operating mode (paper-only enforced)
    # ------------------------------------------------------------------
    trading_mode: TradingMode = Field(default=TradingMode.PAPER)
    live_trading_enabled: bool = Field(default=False)

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------
    watchlist: List[str] = Field(
        default_factory=lambda: [
            "AAPL",
            "NVDA",
            "TSLA",
            "MSFT",
            "AMZN",
            "META",
            "GOOGL",
            "NFLX",
            "AMD",
            "QQQ",
        ]
    )

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    sleep_time: str = Field(default="1M")

    # ------------------------------------------------------------------
    # Strategy / indicators
    # ------------------------------------------------------------------
    indicator_timeframe: str = Field(default="5 minutes")
    indicator_lookback: int = Field(default=200, ge=1)

    # ------------------------------------------------------------------
    # Market session / time filter (America/New_York)
    # ------------------------------------------------------------------
    time_filter_enabled: bool = Field(default=True)
    market_open_time: str = Field(default="09:30")
    opening_blackout_minutes: int = Field(default=30, ge=0)
    market_timezone: str = Field(default="America/New_York")

    # ------------------------------------------------------------------
    # Risk controls
    # ------------------------------------------------------------------
    position_size_pct: float = Field(default=0.03, ge=0, le=1)
    max_position_notional: float = Field(default=2500.0, ge=0)
    max_portfolio_risk_pct: float = Field(default=0.01, ge=0, le=1)
    max_shares_per_trade: int = Field(default=100, ge=0)
    max_concurrent_positions: int = Field(default=2, ge=0)
    # All percentage values are fractions of price (0.012 == 1.2%).
    stop_loss_min_pct: float = Field(default=0.010, gt=0, lt=1)
    stop_loss_max_pct: float = Field(default=0.015, gt=0, lt=1)
    stop_loss_pct: float = Field(default=0.012, gt=0, lt=1)
    take_profit_pct: float = Field(default=0.04, gt=0, lt=1)
    min_reward_risk_ratio: float = Field(default=1.0, ge=0)
    daily_loss_limit_enabled: bool = Field(default=True)
    daily_loss_limit_usd: float = Field(default=250.0, ge=0)
    daily_loss_limit_pct: float = Field(default=0.0025, ge=0, le=1)

    # ------------------------------------------------------------------
    # Caching (seconds / entries)
    # ------------------------------------------------------------------
    indicator_cache_ttl: float = Field(default=30.0, gt=0)
    cache_maxsize: int = Field(default=512, ge=1)

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------
    alert_workers: int = Field(default=2, ge=1)

    # ------------------------------------------------------------------
    # Retry / backoff
    # ------------------------------------------------------------------
    retry_attempts: int = Field(default=4, ge=1)
    retry_base_delay: float = Field(default=1.0, gt=0)
    retry_max_delay: float = Field(default=30.0, gt=0)
    retry_backoff_factor: float = Field(default=2.0, gt=1)
    retry_jitter: bool = Field(default=True)

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------
    telegram_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    telegram_enabled: bool = Field(default=False)
    telegram_timeout: float = Field(default=5.0, gt=0)
    telegram_retry_attempts: int = Field(default=3, ge=1)

    # ------------------------------------------------------------------
    # Persistence file paths (relative to project dir)
    # ------------------------------------------------------------------
    risk_state_file: str = Field(default="logs/risk_state.json")
    trade_ledger_file: str = Field(default="logs/trade_ledger.json")
    db_path: str = Field(default="data/trading_bot.sqlite")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")
    log_file: str = Field(default="tradingbot.log")
    log_max_bytes: int = Field(default=5_000_000, gt=0)
    log_backup_count: int = Field(default=5, ge=0)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("alpaca_data_feed")
    @classmethod
    def _validate_feed(cls, value: str) -> str:
        feed = value.strip().lower()
        if feed not in {"iex", "sip"}:
            raise ValueError(f"ALPACA_DATA_FEED must be 'iex' or 'sip', got {value!r}")
        return feed

    @field_validator("market_open_time")
    @classmethod
    def _validate_market_open(cls, value: str) -> str:
        parts = value.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"MARKET_OPEN_TIME must be HH:MM, got {value!r}")
        try:
            hour, minute = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"MARKET_OPEN_TIME must be HH:MM, got {value!r}") from exc
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"MARKET_OPEN_TIME out of range: {value!r}")
        return f"{hour:02d}:{minute:02d}"

    @model_validator(mode="after")
    def _enforce_paper_mode(self) -> "Settings":
        """Enforce paper trading only; reject any attempt to enable live mode."""
        if self.live_trading_enabled:
            raise ValueError(
                "LIVE_TRADING_ENABLED is true; live trading is disabled by design"
            )
        if self.trading_mode == TradingMode.LIVE:
            raise ValueError(
                "TRADING_MODE=live; live trading is disabled by design"
            )
        if not self.alpaca_paper:
            raise ValueError(
                "ALPACA_PAPER is false; paper trading is required"
            )
        return self

    @model_validator(mode="after")
    def _validate_stop_band(self) -> "Settings":
        if self.stop_loss_min_pct > self.stop_loss_max_pct:
            raise ValueError(
                "STOP_LOSS_MIN_PCT must not exceed STOP_LOSS_MAX_PCT"
            )
        return self

    def alpaca_config(self) -> Dict[str, Any]:
        """Return the Alpaca client configuration dict."""
        return {
            "api_key": self.alpaca_api_key,
            "api_secret": self.alpaca_api_secret,
            "paper": self.alpaca_paper,
            "feed": self.alpaca_data_feed,
        }

    def is_paper(self) -> bool:
        """True when the system is configured for paper trading."""
        return (
            self.trading_mode == TradingMode.PAPER
            and not self.live_trading_enabled
            and self.alpaca_paper
        )


def load_settings(env_file: Optional[str | Path] = None) -> Settings:
    """Load settings from environment (and optional ``.env`` file).

    Loads ``.env`` first so environment variables take precedence, then builds
    and validates the :class:`Settings` model.
    """
    _load_dotenv(env_file)
    return Settings(
        alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
        alpaca_api_secret=os.getenv("ALPACA_API_SECRET", ""),
        alpaca_paper=os.getenv("ALPACA_PAPER", "true").strip().lower() in {
            "1", "true", "yes", "y", "on",
        },
        alpaca_data_feed=os.getenv("ALPACA_DATA_FEED", "iex"),
        trading_mode=TradingMode(
            os.getenv("TRADING_MODE", "paper").strip().lower()
        ),
        live_trading_enabled=os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() in {
            "1", "true", "yes", "y", "on",
        },
        watchlist=_env_list(os.getenv("WATCHLIST"), None) or [
            "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "AMD", "QQQ",
        ],
        sleep_time=os.getenv("SLEEPTIME", "1M"),
        indicator_timeframe=os.getenv("INDICATOR_TIMEFRAME", "5 minutes"),
        indicator_lookback=int(os.getenv("INDICATOR_LOOKBACK", "200")),
        time_filter_enabled=os.getenv("TIME_FILTER_ENABLED", "true").strip().lower() in {
            "1", "true", "yes", "y", "on",
        },
        market_open_time=os.getenv("MARKET_OPEN_TIME", "09:30"),
        opening_blackout_minutes=int(os.getenv("OPENING_BLACKOUT_MINUTES", "30")),
        market_timezone=os.getenv("MARKET_TIMEZONE", "America/New_York"),
        position_size_pct=float(os.getenv("POSITION_SIZE_PCT", "0.03")),
        max_position_notional=float(os.getenv("MAX_POSITION_NOTIONAL", "2500")),
        max_portfolio_risk_pct=float(os.getenv("MAX_PORTFOLIO_RISK_PCT", "0.01")),
        max_shares_per_trade=int(os.getenv("MAX_SHARES_PER_TRADE", "100")),
        max_concurrent_positions=int(os.getenv("MAX_CONCURRENT_POSITIONS", "2")),
        stop_loss_min_pct=float(os.getenv("STOP_LOSS_MIN_PCT", "0.010")),
        stop_loss_max_pct=float(os.getenv("STOP_LOSS_MAX_PCT", "0.015")),
        stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.012")),
        take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.04")),
        min_reward_risk_ratio=float(os.getenv("MIN_REWARD_RISK_RATIO", "1.0")),
        daily_loss_limit_enabled=os.getenv("DAILY_LOSS_LIMIT_ENABLED", "true").strip().lower() in {
            "1", "true", "yes", "y", "on",
        },
        daily_loss_limit_usd=float(os.getenv("DAILY_LOSS_LIMIT_USD", "250")),
        daily_loss_limit_pct=float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.0025")),
        indicator_cache_ttl=float(os.getenv("INDICATOR_CACHE_TTL", "30")),
        cache_maxsize=int(os.getenv("CACHE_MAXSIZE", "512")),
        alert_workers=int(os.getenv("ALERT_WORKERS", "2")),
        retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "4")),
        retry_base_delay=float(os.getenv("RETRY_BASE_DELAY", "1.0")),
        retry_max_delay=float(os.getenv("RETRY_MAX_DELAY", "30.0")),
        retry_backoff_factor=float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0")),
        retry_jitter=os.getenv("RETRY_JITTER", "true").strip().lower() in {
            "1", "true", "yes", "y", "on",
        },
        telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        telegram_enabled=os.getenv("TELEGRAM_ENABLED", "false").strip().lower() in {
            "1", "true", "yes", "y", "on",
        },
        telegram_timeout=float(os.getenv("TELEGRAM_TIMEOUT", "5")),
        telegram_retry_attempts=int(os.getenv("TELEGRAM_RETRY_ATTEMPTS", "3")),
        risk_state_file=os.getenv("RISK_STATE_FILE", "logs/risk_state.json"),
        trade_ledger_file=os.getenv("TRADE_LEDGER_FILE", "logs/trade_ledger.json"),
        db_path=os.getenv("DB_PATH", "data/trading_bot.sqlite"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_dir=os.getenv("LOG_DIR", "logs"),
        log_file=os.getenv("LOG_FILE", "tradingbot.log"),
        log_max_bytes=int(os.getenv("LOG_MAX_BYTES", "5000000")),
        log_backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
    )
