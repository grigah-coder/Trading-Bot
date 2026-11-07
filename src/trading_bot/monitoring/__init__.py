"""Monitoring and operational readiness package.

Provides structured logging, health checks, broker/market-data/position/risk
monitoring, error alerts, circuit-breaker alerts, crash notifications and a
notification interface supporting Telegram or another provider.

Implemented in Phase 10 (Monitoring).
"""

__all__ = ["health", "notifier"]
