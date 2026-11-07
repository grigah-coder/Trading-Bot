"""Notification backends for alerts.

Provides :class:`Notifier` (base interface), :class:`ConsoleNotifier` (logs
alerts — the default, network-free fallback) and :class:`TelegramNotifier`
(sends alerts via the Telegram Bot API when ``TELEGRAM_ENABLED=true``).

The :func:`build_notifier` factory reads :class:`trading_bot.config.settings.Settings`
and returns the appropriate backend. All notifiers are best-effort: network
failures are logged, never raised, so alerting can never crash the trading
loop.
"""

from __future__ import annotations

import logging
from typing import Any

try:  # requests is a declared runtime dependency
    import requests  # pyright: ignore[reportMissingModuleSource]
except Exception:  # pragma: no cover - requests always present in this project
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Telegram Bot API: sendMessage endpoint.
_TELEGRAM_SEND_MESSAGE = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier:
    """Base interface for alert notifiers."""

    def notify(self, message: str, level: str = "error") -> None:
        """Send ``message`` at ``level``. Best-effort; never raises."""
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    """Logs alerts to the application logger. No network, always safe."""

    def __init__(self, name: str = "monitoring.notifier") -> None:
        self._logger = logging.getLogger(name)

    def notify(self, message: str, level: str = "error") -> None:
        level = (level or "error").strip().lower()
        if level == "critical":
            self._logger.critical("ALERT: %s", message)
        elif level == "error":
            self._logger.error("ALERT: %s", message)
        elif level == "warning":
            self._logger.warning("ALERT: %s", message)
        else:
            self._logger.info("ALERT: %s", message)


class TelegramNotifier(Notifier):
    """Sends alerts to a Telegram chat via the Bot API.

    Uses ``requests`` with a short timeout so a slow network cannot block the
    trading loop. Failures are logged and swallowed.
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        timeout: float = 5.0,
        retry_attempts: int = 1,
    ) -> None:
        self.token = token.strip()
        self.chat_id = chat_id.strip()
        self.timeout = max(1.0, float(timeout))
        self.retry_attempts = max(1, int(retry_attempts))
        self._session = requests.Session() if requests is not None else None

    def notify(self, message: str, level: str = "error") -> None:
        if not self.token or not self.chat_id:
            logger.warning("Telegram notifier not configured; dropping alert")
            return
        if self._session is None:
            logger.warning("requests unavailable; dropping alert")
            return
        url = _TELEGRAM_SEND_MESSAGE.format(token=self.token)
        payload = {
            "chat_id": self.chat_id,
            "text": f"*[{level.upper()}]* {message}",
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return
                logger.warning(
                    "Telegram alert failed (HTTP %s): %s",
                    resp.status_code,
                    resp.text[:200],
                )
            except Exception:  # noqa: BLE001
                logger.debug("Telegram alert attempt %d failed", attempt)
        logger.warning("Telegram alert could not be delivered after %d attempt(s)",
                       self.retry_attempts)


def build_notifier(settings: Any) -> Notifier:
    """Build a notifier from application settings.

    Returns :class:`TelegramNotifier` when ``telegram_enabled`` is true and
    token/chat_id are set; otherwise :class:`ConsoleNotifier`.
    """
    telegram_enabled = bool(getattr(settings, "telegram_enabled", False))
    token = str(getattr(settings, "telegram_token", "") or "").strip()
    chat_id = str(getattr(settings, "telegram_chat_id", "") or "").strip()

    if telegram_enabled and token and chat_id:
        logger.info("Telegram notifier enabled")
        return TelegramNotifier(
            token=token,
            chat_id=chat_id,
            timeout=float(getattr(settings, "telegram_timeout", 5.0)),
            retry_attempts=int(getattr(settings, "telegram_retry_attempts", 1)),
        )
    logger.info("Console notifier enabled (Telegram not configured)")
    return ConsoleNotifier()


__all__ = ["Notifier", "ConsoleNotifier", "TelegramNotifier", "build_notifier"]
