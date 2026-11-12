"""Alpaca broker adapter (paper trading only).

Implements the :class:`BrokerAdapter` protocol using alpaca-py's
``TradingClient`` against the paper environment. Live trading is disabled by
design: the constructor rejects any non-paper configuration.

Responsibilities:
* account information (equity, cash, buying power)
* position and order retrieval
* market / limit order submission
* order cancellation and replacement
* order status monitoring (canonical states)
* partial-fill handling (filled_qty, filled_avg_price)
* broker error handling, idempotency (client_order_id), state reconciliation

Safety rules:
* Paper trading only — ``ALPACA_PAPER=true`` and ``LIVE_TRADING_ENABLED=false``
  are enforced at construction.
* Never blindly retry an order after an uncertain API timeout: before a retry,
  the adapter queries the broker for a matching existing order (idempotency
  via ``client_order_id``).
* The broker is the authoritative source for execution and position state.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from alpaca.common.exceptions import APIError  # pyright: ignore[reportMissingImports]
from alpaca.trading.client import TradingClient  # pyright: ignore[reportMissingImports]
from alpaca.trading.enums import (  # pyright: ignore[reportMissingImports]
    OrderClass,
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.requests import (  # pyright: ignore[reportMissingImports]
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from trading_bot.config.settings import Settings
from trading_bot.exceptions import (
    BrokerError,
    BrokerUnavailableError,
    OrderRejectedError,
)
from trading_bot.logging_config import get_logger
from trading_bot.retry import call_with_retry, is_network_error, is_retryable

logger = get_logger(__name__)

# Canonical order states (normalized from Alpaca's varying strings).
OPEN_STATES = frozenset(
    {
        "pending",
        "pending_new",
        "pending_replace",
        "submitted",
        "new",
        "accepted",
        "accepted_for_bidding",
        "open",
        "held",
        "partially_filled",
    }
)
TERMINAL_STATES = frozenset(
    {"filled", "canceled", "cancelled", "expired", "rejected", "error", "done_for_day"}
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_side(value: Any) -> str:
    return str(getattr(value, "value", value) or value).lower()


def _normalize_status(value: Any) -> str:
    return str(getattr(value, "value", value) or value).lower()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> Optional[float]:
    """Return ``None`` for absent prices instead of a fake ``0.0``."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_duplicate_client_order_id(exc: Exception) -> bool:
    """True when the broker rejected an order as a duplicate client id."""
    if not isinstance(exc, APIError) or getattr(exc, "status_code", None) != 422:
        return False
    text = str(exc).lower()
    return "client_order_id" in text and (
        "unique" in text or "duplicate" in text or "already" in text
    )


def _api_error_to_exception(exc: Exception, context: str) -> Exception:
    """Translate broker/network failures; pass other errors through.

    Only ``APIError`` responses and network-level failures become broker
    exceptions. Programming errors (``ValueError``, ``TypeError``, ...) are
    returned unchanged so bugs are not silently reported as broker outages.
    """
    if isinstance(exc, APIError):
        code = getattr(exc, "status_code", None)
        message = str(exc)
        if code in {401, 403}:
            return OrderRejectedError(f"{context}: unauthorized/forbidden ({message})")
        if code == 422:
            return OrderRejectedError(f"{context}: invalid request ({message})")
        if code == 404:
            return BrokerError(f"{context}: not found ({message})")
        if code in {408, 429, 500, 502, 503, 504} or is_retryable(exc):
            return BrokerUnavailableError(f"{context}: broker unavailable ({message})")
        return BrokerError(f"{context}: alpaca error ({message})")
    if is_network_error(exc):
        return BrokerUnavailableError(f"{context}: {exc}")
    return exc


class AlpacaBrokerAdapter:
    """Alpaca paper-trading broker adapter."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # --- Paper-only enforcement ---
        if not settings.is_paper():
            raise ValueError(
                "AlpacaBrokerAdapter requires paper trading only "
                "(TRADING_MODE=paper, LIVE_TRADING_ENABLED=false, ALPACA_PAPER=true)"
            )
        if not settings.alpaca_api_key or not settings.alpaca_api_secret:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_API_SECRET are required for the broker adapter"
            )

        self._client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_api_secret,
            paper=settings.alpaca_paper,
        )
        self._feed = settings.alpaca_data_feed

    # ------------------------------------------------------------------
    # BrokerAdapter protocol
    # ------------------------------------------------------------------
    def is_connected(self) -> bool:
        """True when the broker account endpoint responds."""
        try:
            self._call(self._client.get_account, description="broker_is_connected")
            return True
        except Exception:  # noqa: BLE001
            return False

    def get_account(self) -> Optional[dict[str, Any]]:
        """Return a normalized account snapshot (equity, cash, buying power, ...)."""
        try:
            account = self._call(self._client.get_account, description="get_account")
        except Exception as exc:  # noqa: BLE001
            raise _api_error_to_exception(exc, "get_account")
        if account is None:
            return None
        return self._normalize_account(account)

    def get_positions(self) -> list[dict[str, Any]]:
        """Return all open positions as normalized dicts."""
        try:
            raw = self._call(
                self._client.get_all_positions, description="get_all_positions"
            ) or []
        except Exception as exc:  # noqa: BLE001
            raise _api_error_to_exception(exc, "get_all_positions")
        return [self._normalize_position(p) for p in raw]

    def get_position(self, symbol: str) -> Optional[dict[str, Any]]:
        """Return the open position for ``symbol`` (or ``None``)."""
        try:
            position = self._call(
                lambda: self._client.get_open_position(symbol),
                description=f"get_open_position[{symbol}]",
            )
        except Exception as exc:  # noqa: BLE001
            # 404 -> no position (not an error).
            if isinstance(exc, APIError) and getattr(exc, "status_code", None) == 404:
                return None
            raise _api_error_to_exception(exc, f"get_open_position[{symbol}]")
        if position is None:
            return None
        return self._normalize_position(position)

    def get_orders(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        """Return orders, optionally filtered by canonical status."""
        status_value = status.lower() if isinstance(status, str) else status
        request = GetOrdersRequest(
            status=QueryOrderStatus(status_value) if status_value else QueryOrderStatus.ALL
        )
        try:
            raw = self._call(
                lambda: self._client.get_orders(filter=request),
                description="get_orders",
            ) or []
        except Exception as exc:  # noqa: BLE001
            raise _api_error_to_exception(exc, "get_orders")
        return [self._normalize_order(o) for o in raw]

    def get_order(self, order_id: str) -> Optional[dict[str, Any]]:
        """Return a single order by id (or ``None`` when not found)."""
        try:
            order = self._call(
                lambda: self._client.get_order_by_id(order_id),
                description=f"get_order_by_id[{order_id}]",
            )
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, APIError) and getattr(exc, "status_code", None) == 404:
                return None
            raise _api_error_to_exception(exc, f"get_order_by_id[{order_id}]")
        if order is None:
            return None
        return self._normalize_order(order)

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        *,
        client_order_id: Optional[str] = None,
        time_in_force: str = "day",
        extended_hours: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Submit a market order with idempotency via ``client_order_id``.

        ``client_order_id`` is mandatory for idempotency: a retry with the same
        id will be detected as a duplicate by the broker and not double-filled.
        """
        if client_order_id is None:
            client_order_id = f"tb-{uuid.uuid4().hex}"
        return self._submit_order(
            MarketOrderRequest(
                symbol=symbol.upper(),
                qty=_as_float(qty),
                side=OrderSide(side.lower()),
                type=OrderType.MARKET,
                time_in_force=TimeInForce(time_in_force),
                client_order_id=client_order_id,
                extended_hours=extended_hours,
            ),
            client_order_id=client_order_id,
            context="submit_market_order",
        )

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        *,
        client_order_id: Optional[str] = None,
        time_in_force: str = "day",
        extended_hours: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Submit a limit order with idempotency via ``client_order_id``."""
        if client_order_id is None:
            client_order_id = f"tb-{uuid.uuid4().hex}"
        return self._submit_order(
            LimitOrderRequest(
                symbol=symbol.upper(),
                qty=_as_float(qty),
                side=OrderSide(side.lower()),
                type=OrderType.LIMIT,
                limit_price=_as_float(limit_price),
                time_in_force=TimeInForce(time_in_force),
                client_order_id=client_order_id,
                extended_hours=extended_hours,
            ),
            client_order_id=client_order_id,
            context="submit_limit_order",
        )

    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        take_profit_price: Optional[float] = None,
        *,
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        time_in_force: str = "day",
    ) -> Optional[dict[str, Any]]:
        """Submit a bracket order with attached stop-loss and take-profit legs.

        Quantities are floored to whole shares; fractional or zero-share
        brackets are rejected rather than rounded up. ``limit_price`` is
        accepted as an alias for ``take_profit_price``.
        """
        if take_profit_price is None:
            take_profit_price = limit_price
        whole_qty = math.floor(_as_float(qty))
        if whole_qty < 1:
            raise OrderRejectedError(
                f"bracket order for {symbol.upper()} requires at least 1 whole "
                f"share, got {qty!r}"
            )
        if stop_price is None or take_profit_price is None:
            raise OrderRejectedError(
                f"bracket order for {symbol.upper()} requires both stop and "
                "take-profit prices"
            )
        if client_order_id is None:
            client_order_id = f"tb-{uuid.uuid4().hex}"
        return self._submit_order(
            MarketOrderRequest(
                symbol=symbol.upper(),
                qty=float(whole_qty),
                side=OrderSide(side.lower()),
                type=OrderType.MARKET,
                time_in_force=TimeInForce(time_in_force),
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(
                    limit_price=_as_float(take_profit_price),
                ),
                stop_loss=StopLossRequest(
                    stop_price=_as_float(stop_price),
                ),
                client_order_id=client_order_id,
            ),
            client_order_id=client_order_id,
            context="submit_bracket_order",
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True when cancelled / already gone."""
        try:
            self._call(
                lambda: self._client.cancel_order_by_id(order_id),
                description=f"cancel_order[{order_id}]",
            )
            return True
        except Exception as exc:  # noqa: BLE001
            # 404 / 422 -> order already closed; treat as success.
            if isinstance(exc, APIError) and getattr(exc, "status_code", None) in {
                404,
                422,
            }:
                logger.debug("Order %s already closed: %s", order_id, exc)
                return True
            raise _api_error_to_exception(exc, f"cancel_order[{order_id}]")

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns the number cancelled."""
        try:
            self._call(self._client.cancel_orders, description="cancel_orders")
            return 0  # Alpaca does not return a count; caller re-queries if needed
        except Exception as exc:  # noqa: BLE001
            raise _api_error_to_exception(exc, "cancel_all_orders")

    def replace_order(
        self, order_id: str, *, qty: float, limit_price: Optional[float] = None
    ) -> Optional[dict[str, Any]]:
        """Replace an open order (qty and/or limit price). Alpaca returns a new order id.

        The order side cannot be changed on a replacement.
        """
        fields: dict[str, Any] = {"qty": _as_float(qty)}
        if limit_price is not None:
            fields["limit_price"] = _as_float(limit_price)
        try:
            order = self._call(
                lambda: self._client.replace_order_by_id(
                    order_id, ReplaceOrderRequest(**fields)
                ),
                description=f"replace_order[{order_id}]",
            )
        except Exception as exc:  # noqa: BLE001
            raise _api_error_to_exception(exc, f"replace_order[{order_id}]")
        return self._normalize_order(order) if order is not None else None

    def close_position(self, symbol: str) -> Optional[dict[str, Any]]:
        """Close (liquidate) a single position at market."""
        try:
            order = self._call(
                lambda: self._client.close_position(symbol),
                description=f"close_position[{symbol}]",
            )
        except Exception as exc:  # noqa: BLE001
            raise _api_error_to_exception(exc, f"close_position[{symbol}]")
        return self._normalize_order(order) if order is not None else None

    def close_all_positions(self, cancel_orders: bool = True) -> list[dict[str, Any]]:
        """Close all open positions. Returns close responses."""
        try:
            response = self._call(
                lambda: self._client.close_all_positions(cancel_orders=cancel_orders),
                description="close_all_positions",
            ) or []
        except Exception as exc:  # noqa: BLE001
            raise _api_error_to_exception(exc, "close_all_positions")
        if isinstance(response, list):
            return [self._normalize_order(r) if hasattr(r, "status") else r for r in response]
        return [response] if response else []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _call(self, func, description: str, attempts: Optional[int] = None):
        """Wrap an alpaca-py call with retry.

        Re-raises the raw exception (alpaca-py ``APIError`` or network error);
        translation to our exception types happens at each call site so errors
        are translated exactly once.
        """
        attempts = attempts or self.settings.retry_attempts
        return call_with_retry(
            func,
            attempts=attempts,
            base_delay=self.settings.retry_base_delay,
            max_delay=self.settings.retry_max_delay,
            backoff_factor=self.settings.retry_backoff_factor,
            jitter=self.settings.retry_jitter,
            description=description,
        )

    def _submit_order(self, order_request, client_order_id: str, context: str):
        """Submit an order once, resolving uncertain outcomes by lookup.

        Submission itself is **never retried**: a timeout may have placed the
        order already, and a blind retry would either double-fill or fail with
        a duplicate-client-order-id 422. Instead, after an unavailable or
        duplicate response we look the order up by ``client_order_id`` and
        return it when it exists.
        """
        # Idempotency check: does this client_order_id already exist?
        existing = self._get_order_by_client_id(client_order_id)
        if existing is not None:
            logger.info(
                "[%s] order with client_order_id %s already exists; not resubmitting",
                order_request.symbol,
                client_order_id,
            )
            return existing

        try:
            order = self._call(
                lambda: self._client.submit_order(order_request),
                description=f"{context}[{order_request.symbol}]",
                attempts=1,
            )
        except Exception as exc:  # noqa: BLE001
            broker_exc = _api_error_to_exception(
                exc, f"{context}[{order_request.symbol}]"
            )
            uncertain = isinstance(broker_exc, BrokerUnavailableError)
            duplicate = _is_duplicate_client_order_id(exc)
            if uncertain or duplicate:
                existing = self._get_order_by_client_id(client_order_id)
                if existing is not None:
                    logger.warning(
                        "[%s] submission failed (%s) but order %s exists; "
                        "treating as accepted",
                        order_request.symbol,
                        "duplicate" if duplicate else "uncertain timeout",
                        client_order_id,
                    )
                    return existing
            if broker_exc is exc:
                raise
            raise broker_exc

        return self._normalize_order(order) if order is not None else None

    def _get_order_by_client_id(self, client_order_id: str) -> Optional[dict[str, Any]]:
        """Return the order with ``client_order_id`` (``None`` when absent).

        Uses the broker's dedicated lookup endpoint: one direct request instead
        of two capped list queries. A 404 means "not found", which is the
        normal case before submission.
        """
        try:
            order = self._call(
                lambda: self._client.get_order_by_client_id(client_order_id),
                description=f"get_order_by_client_id[{client_order_id}]",
            )
        except Exception as exc:  # noqa: BLE001 - lookup is best effort
            if isinstance(exc, APIError) and getattr(exc, "status_code", None) == 404:
                return None
            logger.warning(
                "Could not look up client_order_id %s: %s", client_order_id, exc,
            )
            return None
        if order is None:
            return None
        return self._normalize_order(order)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def _normalize_account(self, account: Any) -> dict[str, Any]:
        return {
            "account_id": str(getattr(account, "account_id", "")),
            "status": _normalize_status(getattr(account, "status", "")),
            "currency": getattr(account, "currency", "USD"),
            "cash": _as_float(getattr(account, "cash", None)),
            "equity": _as_float(getattr(account, "equity", None)),
            "portfolio_value": _as_float(getattr(account, "portfolio_value", None)),
            "buying_power": _as_float(getattr(account, "buying_power", None)),
            "long_market_value": _as_float(getattr(account, "long_market_value", None)),
            "short_market_value": _as_float(getattr(account, "short_market_value", None)),
            "account_blocked": bool(getattr(account, "account_blocked", False)),
            "trading_blocked": bool(getattr(account, "trading_blocked", False)),
            "transfers_blocked": bool(getattr(account, "transfers_blocked", False)),
            "day_trade_count": int(getattr(account, "daytrade_count", 0) or 0),
            "last_equity": _as_float(getattr(account, "last_equity", None)),
            "day_pl": None,
            "day_pl_pct": None,
            "raw": account,
        }

    def _normalize_position(self, position: Any) -> dict[str, Any]:
        symbol = str(getattr(position, "symbol", "")).upper()
        qty = _as_float(getattr(position, "qty", None))
        avg_entry = _as_float(getattr(position, "avg_entry_price", None))
        current = _as_float(getattr(position, "current_price", None))
        market_value = _as_float(getattr(position, "market_value", None))
        unrealized_pl = _as_float(getattr(position, "unrealized_pl", None))
        unrealized_plpc = _as_float(getattr(position, "unrealized_plpc", None))
        return {
            "symbol": symbol,
            "asset_id": str(getattr(position, "asset_id", "")),
            "exchange": getattr(position, "exchange", ""),
            "asset_class": getattr(position, "asset_class", ""),
            "qty": qty,
            "avg_entry_price": avg_entry,
            "current_price": current,
            "market_value": market_value,
            "cost_basis": _as_float(getattr(position, "cost_basis", None)),
            "unrealized_pl": unrealized_pl,
            "unrealized_plpc": unrealized_plpc,
            "unrealized_intraday_pl": _as_float(
                getattr(position, "unrealized_intraday_pl", None)
            ),
            "unrealized_intraday_plpc": _as_float(
                getattr(position, "unrealized_intraday_plpc", None)
            ),
            "current_date": _to_iso(getattr(position, "current_date", None)),
            "lastday_price": _as_float(getattr(position, "lastday_price", None)),
            "change_today": _as_float(getattr(position, "change_today", None)),
            "side": "long" if qty >= 0 else "short",
            "raw": position,
        }

    def _normalize_order(self, order: Any) -> dict[str, Any]:
        filled_qty = _as_float(getattr(order, "filled_qty", None))
        filled_avg_price = _as_float(getattr(order, "filled_avg_price", None))
        submitted_at = getattr(order, "submitted_at", None)
        filled_at = getattr(order, "filled_at", None)
        status = _normalize_status(getattr(order, "status", ""))
        raw_legs = getattr(order, "legs", None) or []
        return {
            "order_id": str(getattr(order, "id", "")),
            "client_order_id": str(getattr(order, "client_order_id", "")),
            "symbol": str(getattr(order, "symbol", "")).upper(),
            "asset_id": str(getattr(order, "asset_id", "")),
            "side": _normalize_side(getattr(order, "side", "")),
            "qty": _as_float(getattr(order, "qty", None)),
            "filled_qty": filled_qty,
            "filled_avg_price": filled_avg_price,
            "order_type": _normalize_side(getattr(order, "order_type", "")),
            "order_class": _normalize_side(getattr(order, "order_class", "")),
            "time_in_force": _normalize_side(getattr(order, "time_in_force", "")),
            "status": status,
            "is_open": status in OPEN_STATES,
            "is_terminal": status in TERMINAL_STATES,
            "is_filled": status == "filled",
            "is_cancelled": status in {"canceled", "cancelled"},
            "is_rejected": status == "rejected",
            "limit_price": _as_optional_float(
                getattr(order, "limit_price", None)
            ),
            "stop_price": _as_optional_float(
                getattr(order, "stop_price", None)
            ),
            "submitted_at": _to_iso(submitted_at),
            "filled_at": _to_iso(filled_at),
            "expired_at": _to_iso(getattr(order, "expired_at", None)),
            "canceled_at": _to_iso(getattr(order, "canceled_at", None)),
            "failed_at": _to_iso(getattr(order, "failed_at", None)),
            "replaced_at": _to_iso(getattr(order, "replaced_at", None)),
            "replaced_by": str(getattr(order, "replaced_by", "")) or None,
            "replaces": str(getattr(order, "replaces", "")) or None,
            "legs": [self._normalize_order(leg) for leg in raw_legs],
            "raw": order,
        }


def _to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)
