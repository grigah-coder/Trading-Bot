"""Structured exceptions shared across application layers."""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar, Mapping


class TradingBotError(Exception):
    """Base exception with a stable code and safe diagnostic context."""

    code: ClassVar[str] = "trading_bot_error"

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None):
        super().__init__(message)
        self.context = MappingProxyType(dict(context or {}))


class ConfigurationError(TradingBotError):
    code = "invalid_configuration"


class StateTransitionError(TradingBotError):
    code = "invalid_state_transition"


class ComponentHealthError(TradingBotError):
    code = "component_health_check_failed"


class EngineCycleError(TradingBotError):
    code = "engine_cycle_failed"


class EngineStateError(TradingBotError):
    code = "engine_not_running"


class EngineShutdownError(TradingBotError):
    code = "engine_shutdown_failed"


class ApplicationStartupError(TradingBotError):
    code = "application_startup_failed"


class ApplicationShutdownError(TradingBotError):
    code = "application_shutdown_failed"


class RecoveryError(TradingBotError):
    code = "recovery_exhausted"


class MarketCalendarError(TradingBotError):
    code = "market_calendar_error"


class RiskGateRequired(TradingBotError):
    code = "risk_gate_required"


class TradingBlocked(TradingBotError):
    code = "trading_blocked"
