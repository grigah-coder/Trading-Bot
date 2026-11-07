"""Health monitoring and alerting.

Provides :class:`HealthMonitor`, which implements the
:class:`trading_bot.engine.Monitoring` protocol:

* :meth:`health_check` — snapshot of component health (broker, data, risk,
  order manager, database) plus circuit-breaker and stale-data flags.
* :meth:`alert` — emit a levelled alert (logged; forwarded to a notifier when
  configured).
* :meth:`record_metric` — record a named numeric metric (gauge/counter).

The monitor is dependency-light: it aggregates health from the components it
is given and never reaches out to the network, so the trading loop can call it
every iteration without risk.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from trading_bot.logging_config import configure_logging
from trading_bot.monitoring.notifier import Notifier

logger = configure_logging(__name__).getChild("health")

# Maximum number of recent alerts to keep in memory.
_MAX_ALERTS = 500
# Maximum number of metric samples to keep per name.
_MAX_METRIC_SAMPLES = 1000


class HealthMonitor:
    """Aggregates component health and emits levelled alerts.

    Parameters
    ----------
    notifier:
        Optional :class:`Notifier` (e.g. Telegram) alerts are forwarded to.
    components:
        Optional mapping of component name -> health-check callable returning a
        dict with at least an ``ok`` key. Called on every :meth:`health_check`.

    """

    def __init__(
        self,
        notifier: Optional[Notifier] = None,
        components: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.notifier = notifier
        self.components = dict(components or {})
        self._lock = threading.RLock()
        self._metrics: Dict[str, Deque[float]] = {}
        self._metric_counts: Dict[str, int] = {}
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=_MAX_ALERTS)
        self._circuit_breaker_tripped = False
        self._circuit_breaker_reason = ""
        self._data_stale = False
        self._last_heartbeat: Optional[datetime] = None
        self._started_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Monitoring protocol
    # ------------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """Return a snapshot of system health.

        The dict always contains ``ok``, ``status``, ``uptime_seconds``,
        ``circuit_breaker_tripped``, ``data_stale``, ``alerts`` (recent count)
        and a ``components`` mapping. ``ok`` is False when any component is
        unhealthy or the circuit breaker is tripped.
        """
        component_health: Dict[str, Any] = {}
        all_ok = True
        for name, check in self.components.items():
            try:
                result = check() if callable(check) else dict(check or {})
                if isinstance(result, dict):
                    health = result
                else:
                    health = {"ok": bool(result)}
            except Exception as exc:  # noqa: BLE001
                health = {"ok": False, "error": str(exc)}
                logger.warning("Health check %s failed: %s", name, exc)
            ok = bool(health.get("ok", False))
            if not ok:
                all_ok = False
            component_health[name] = health

        ok = all_ok and not self._circuit_breaker_tripped and not self._data_stale
        status = "ok" if ok else ("halted" if self._circuit_breaker_tripped else "degraded")

        with self._lock:
            recent_alerts = list(self._alerts)
            alert_count_24h = len(recent_alerts)

        return {
            "ok": ok,
            "status": status,
            "uptime_seconds": (
                (datetime.now(timezone.utc) - self._started_at).total_seconds()
            ),
            "circuit_breaker_tripped": self._circuit_breaker_tripped,
            "circuit_breaker_reason": self._circuit_breaker_reason,
            "data_stale": self._data_stale,
            "last_heartbeat": (
                self._last_heartbeat.isoformat() if self._last_heartbeat else None
            ),
            "alerts_recent": alert_count_24h,
            "components": component_health,
            "metrics": {
                name: {
                    "count": self._metric_counts.get(name, 0),
                    "latest": samples[-1] if samples else None,
                }
                for name, samples in self._metrics.items()
            },
        }

    def alert(self, message: str, level: str = "error") -> None:
        """Emit a levelled alert.

        Logs the message, stores it in the recent-alerts ring buffer, and
        forwards it to the configured notifier (when alerting is enabled).
        Never raises.
        """
        level = (level or "error").strip().lower()
        entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._alerts.append(entry)
        if level in {"critical", "error"}:
            logger.error("ALERT [%s]: %s", level.upper(), message)
        elif level == "warning":
            logger.warning("ALERT [%s]: %s", level.upper(), message)
        else:
            logger.info("ALERT [%s]: %s", level.upper(), message)
        if self.notifier is not None:
            try:
                self.notifier.notify(message, level=level)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Notifier failed: %s", exc)

    def record_metric(self, name: str, value: float) -> None:
        """Record a named numeric metric (keeps the most recent samples)."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        with self._lock:
            samples = self._metrics.setdefault(name, deque(maxlen=_MAX_METRIC_SAMPLES))
            samples.append(v)
            self._metric_counts[name] = self._metric_counts.get(name, 0) + 1

    # ------------------------------------------------------------------
    # State mutation (used by engine / risk / order manager)
    # ------------------------------------------------------------------
    def trip_circuit_breaker(self, reason: str) -> None:
        with self._lock:
            self._circuit_breaker_tripped = True
            self._circuit_breaker_reason = reason
        logger.warning("Circuit breaker tripped: %s", reason)
        self.alert(f"circuit breaker tripped: {reason}", level="critical")

    def reset_circuit_breaker(self) -> None:
        with self._lock:
            self._circuit_breaker_tripped = False
            self._circuit_breaker_reason = ""
        logger.info("Circuit breaker reset")
        self.alert("circuit breaker reset", level="info")

    def set_data_stale(self, stale: bool) -> None:
        with self._lock:
            self._data_stale = bool(stale)

    def touch_heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat = datetime.now(timezone.utc)

    def add_component(self, name: str, check: Any) -> None:
        """Register a component health-check callable."""
        with self._lock:
            self.components[name] = check

    def remove_component(self, name: str) -> None:
        with self._lock:
            self.components.pop(name, None)

    def recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._alerts)[-limit:]

    def snapshot(self) -> Dict[str, Any]:
        """Alias for :meth:`health_check` (kept for engine symmetry)."""
        return self.health_check()


__all__ = ["HealthMonitor"]
