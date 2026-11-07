"""Structured failures from risk assessment and circuit breakers."""

from __future__ import annotations

from core.exceptions import TradingBotError


class RiskError(TradingBotError):
    code = "risk_error"


class RiskRejected(RiskError):
    code = "trade_rejected_by_risk"


class CircuitBreakerActive(RiskError):
    code = "risk_circuit_breaker_active"


class RiskConfigurationError(RiskError):
    code = "invalid_risk_configuration"
