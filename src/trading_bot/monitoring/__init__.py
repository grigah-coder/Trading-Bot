"""Monitoring and operational readiness package.

Provides structured logging, health checks, broker/market-data/position/risk
monitoring, error alerts, circuit-breaker alerts, crash notifications and a
notification interface supporting Telegram or another provider.

Implemented in Phase 11 (Monitoring and Operational Readiness).
Intentionally empty of trading logic in Phase 1.
"""

__all__ = ["health", "notifier"]
