"""Alpaca Trading API adapter, locked to the paper environment."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import requests
from alpaca.common.enums import Sort
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import QueryOrderStatus, TimeInForce as AlpacaTimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
)

from config.settings import BrokerSettings, Settings
from core.retry import call_with_retry
from execution.broker import (
    AccountInfo,
    Broker,
    BrokerAuthenticationError,
    BrokerConnectionError,
    BrokerError,
    BrokerOperationUnknown,
    BrokerOrder,
    BrokerRateLimitError,
    BrokerRejectedOrder,
    BrokerRequestTimeout,
    IdempotencyConflict,
    OrderNotFound,
    OrderSide,
    OrderStatus,
    OrderSubmissionUnknown,
    Position,
    TimeInForce,
)


class _TimeoutSession(requests.Session):
    def __init__(self, timeout_seconds: float) -> None:
        super().__init__()
        self._default_timeout = timeout_seconds

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self._default_timeout)
        return super().request(method, url, **kwargs)


class AlpacaBrokerAdapter(Broker):
    """Official Alpaca SDK adapter with paper-only execution and reconciliation."""

    def __init__(
        self,
        settings: BrokerSettings,
        *,
        client_factory: Callable[..., Any] = TradingClient,
        retry: Callable[..., Any] = call_with_retry,
    ) -> None:
        if not settings.paper:
            raise ValueError("AlpacaBrokerAdapter only permits paper trading")
        if not settings.api_key or not settings.api_secret:
            raise ValueError("Alpaca API key and secret are required")
        self.settings = settings
        self._client = client_factory(settings.api_key, settings.api_secret, paper=True)
        configured_url = getattr(self._client, "_base_url", None)
        endpoint = getattr(configured_url, "value", str(configured_url))
        if configured_url is not None and endpoint.rstrip("/") != "https://paper-api.alpaca.markets":
            raise ValueError("Alpaca client must target the paper endpoint")
        if hasattr(self._client, "_session"):
            self._client._session = _TimeoutSession(settings.request_timeout_seconds)
        self._retry = retry
        self._submission_lock = threading.RLock()

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        client_factory: Callable[..., Any] = TradingClient,
        retry: Callable[..., Any] = call_with_retry,
    ) -> AlpacaBrokerAdapter:
        """Load credentials through validated environment configuration."""
        return cls(Settings.from_env(environ).broker, client_factory=client_factory, retry=retry)

    def get_account(self) -> AccountInfo:
        raw = self._read(self._client.get_account)
        return AccountInfo(
            account_id=str(raw.id), status=_text(raw.status), currency=str(raw.currency),
            cash=_decimal(raw.cash), buying_power=_decimal(raw.buying_power),
            equity=_decimal(raw.equity), portfolio_value=_decimal(raw.portfolio_value),
            day_trade_count=int(getattr(raw, "daytrade_count", 0)),
            trading_blocked=bool(getattr(raw, "trading_blocked", False)),
        )

    def get_open_positions(self) -> Sequence[Position]:
        return tuple(self._position(raw) for raw in self._read(self._client.get_all_positions))

    def get_open_orders(self) -> Sequence[BrokerOrder]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, direction=Sort.ASC)
        raw_orders = self._read(self._client.get_orders, filter=request)
        return tuple(self._order(raw) for raw in raw_orders)

    def get_historical_orders(self, start: datetime, end: datetime) -> Sequence[BrokerOrder]:
        start_utc = _aware_utc(start, "start")
        end_utc = _aware_utc(end, "end")
        if start_utc >= end_utc:
            raise ValueError("start must be before end")
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL, after=start_utc, until=end_utc,
            limit=500, direction=Sort.ASC,
        )
        raw_orders = self._read(self._client.get_orders, filter=request)
        return tuple(self._order(raw) for raw in raw_orders)

    def submit_market_order(
        self, symbol: str, quantity: Decimal | float | int, side: OrderSide, *,
        client_order_id: str, time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> BrokerOrder:
        normalized_symbol = _symbol(symbol)
        qty = _positive_decimal(quantity, "quantity")
        intent = _client_order_id(client_order_id)
        order_data = MarketOrderRequest(
            symbol=normalized_symbol, qty=float(qty), side=_side(side),
            time_in_force=_tif(time_in_force), client_order_id=intent,
        )
        return self._submit_idempotently(order_data, intent, normalized_symbol, qty,
                                         side, "market", None, time_in_force)

    def submit_limit_order(
        self, symbol: str, quantity: Decimal | float | int, side: OrderSide,
        limit_price: Decimal | float, *, client_order_id: str,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> BrokerOrder:
        normalized_symbol = _symbol(symbol)
        qty = _positive_decimal(quantity, "quantity")
        price = _positive_decimal(limit_price, "limit_price")
        intent = _client_order_id(client_order_id)
        order_data = LimitOrderRequest(
            symbol=normalized_symbol, qty=float(qty), limit_price=float(price),
            side=_side(side), time_in_force=_tif(time_in_force), client_order_id=intent,
        )
        return self._submit_idempotently(order_data, intent, normalized_symbol, qty,
                                         side, "limit", price, time_in_force)

    def _submit_idempotently(
        self, request: Any, client_id: str, symbol: str, quantity: Decimal,
        side: OrderSide, order_type: str, limit_price: Decimal | None,
        time_in_force: TimeInForce,
    ) -> BrokerOrder:
        with self._submission_lock:
            try:
                existing = self.get_order_by_client_order_id(client_id)
            except OrderNotFound:
                existing = None
            if existing is not None:
                if not _same_intent(existing, symbol, quantity, side, order_type, limit_price, time_in_force):
                    raise IdempotencyConflict(
                        "client_order_id is already associated with a different order",
                        context={"client_order_id": client_id},
                    )
                return existing

            try:
                # Never apply generic network retries to an ambiguous submission.
                # Retry only explicit 429 responses, which confirm rate limiting.
                raw = self._retry(
                    self._client.submit_order, request,
                    attempts=self.settings.retry_attempts,
                    base_delay=self.settings.retry_base_delay_seconds,
                    retry_on=lambda error: _status_code(error) == 429,
                    description="Alpaca order submission",
                )
                return self._order(raw)
            except Exception as exc:
                if _is_ambiguous_submission_failure(exc) or _is_duplicate_client_id_error(exc):
                    try:
                        reconciled = self.get_order_by_client_order_id(client_id)
                    except Exception as reconcile_error:
                        raise OrderSubmissionUnknown(
                            "Order submission outcome is unknown; reconcile by client_order_id before retrying",
                            context={"client_order_id": client_id},
                        ) from reconcile_error
                    if not _same_intent(reconciled, symbol, quantity, side, order_type, limit_price, time_in_force):
                        raise IdempotencyConflict(
                            "Broker order with this client_order_id does not match the requested intent",
                            context={"client_order_id": client_id},
                        ) from exc
                    return reconciled
                raise _translate_error(exc, operation="submit order", client_order_id=client_id) from exc

    def cancel_order(self, order_id: str) -> BrokerOrder:
        normalized_id = _required_text(order_id, "order_id")
        try:
            self._retry(
                self._client.cancel_order_by_id, normalized_id,
                attempts=self.settings.retry_attempts,
                base_delay=self.settings.retry_base_delay_seconds,
                retry_on=lambda error: _status_code(error) == 429,
                description="Alpaca order cancellation",
            )
        except Exception as exc:
            if _is_ambiguous_submission_failure(exc):
                try:
                    current = self.get_order(normalized_id)
                    if current.status is OrderStatus.CANCELED:
                        return current
                except Exception:
                    pass
                raise BrokerOperationUnknown(
                    "Cancellation outcome is unknown; retrieve order status before retrying",
                    context={"order_id": normalized_id},
                ) from exc
            raise _translate_error(exc, operation="cancel order", order_id=normalized_id) from exc
        try:
            return self.get_order(normalized_id)
        except Exception as exc:
            raise BrokerOperationUnknown(
                "Cancellation was sent but its final broker status could not be confirmed",
                context={"order_id": normalized_id},
            ) from exc

    def replace_order(
        self, order_id: str, *, client_order_id: str,
        quantity: Decimal | float | int | None = None,
        limit_price: Decimal | float | None = None,
    ) -> BrokerOrder:
        normalized_id = _required_text(order_id, "order_id")
        intent = _client_order_id(client_order_id)
        qty = _positive_decimal(quantity, "quantity") if quantity is not None else None
        price = _positive_decimal(limit_price, "limit_price") if limit_price is not None else None
        if qty is not None and qty != qty.to_integral_value():
            raise ValueError("Alpaca order replacement quantity must be a whole number")
        request = ReplaceOrderRequest(
            qty=int(qty) if qty is not None else None,
            limit_price=float(price) if price is not None else None,
            client_order_id=intent,
        )
        try:
            existing = self.get_order_by_client_order_id(intent)
        except OrderNotFound:
            existing = None
        if existing is not None:
            original = self.get_order(normalized_id)
            if not _same_replacement_intent(existing, original, qty, price):
                raise IdempotencyConflict(
                    "client_order_id is already associated with a different replacement",
                    context={"client_order_id": intent, "order_id": normalized_id},
                )
            return existing
        if qty is None and price is None:
            raise ValueError("replace_order requires quantity or limit_price")
        try:
            raw = self._retry(
                self._client.replace_order_by_id, normalized_id, request,
                attempts=self.settings.retry_attempts,
                base_delay=self.settings.retry_base_delay_seconds,
                retry_on=lambda error: _status_code(error) == 429,
                description="Alpaca order replacement",
            )
            return self._order(raw)
        except Exception as exc:
            if _is_ambiguous_submission_failure(exc) or _is_duplicate_client_id_error(exc):
                try:
                    reconciled = self.get_order_by_client_order_id(intent)
                except Exception as reconcile_error:
                    raise BrokerOperationUnknown(
                        "Replacement outcome is unknown; reconcile by client_order_id before retrying",
                        context={"order_id": normalized_id, "client_order_id": intent},
                    ) from reconcile_error
                original = self.get_order(normalized_id)
                if not _same_replacement_intent(reconciled, original, qty, price):
                    raise IdempotencyConflict(
                        "Broker replacement with this client_order_id does not match the requested intent",
                        context={"client_order_id": intent, "order_id": normalized_id},
                    ) from exc
                return reconciled
            raise _translate_error(exc, operation="replace order", order_id=normalized_id) from exc

    def get_order(self, order_id: str) -> BrokerOrder:
        normalized_id = _required_text(order_id, "order_id")
        raw = self._read(self._client.get_order_by_id, normalized_id)
        return self._order(raw)

    def get_order_by_client_order_id(self, client_order_id: str) -> BrokerOrder:
        normalized_id = _client_order_id(client_order_id)
        raw = self._read(self._client.get_order_by_client_id, normalized_id)
        return self._order(raw)

    def reconcile_order(self, client_order_id: str) -> BrokerOrder | None:
        """Return current broker state, or None only after a broker 404 response."""
        try:
            return self.get_order_by_client_order_id(client_order_id)
        except OrderNotFound:
            return None

    def _read(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return self._retry(
                operation, *args, attempts=self.settings.retry_attempts,
                base_delay=self.settings.retry_base_delay_seconds,
                description=getattr(operation, "__name__", "Alpaca request"), **kwargs,
            )
        except BrokerError:
            raise
        except Exception as exc:
            raise _translate_error(exc, operation=getattr(operation, "__name__", "broker request")) from exc

    @staticmethod
    def _position(raw: Any) -> Position:
        return Position(
            symbol=str(raw.symbol).upper(), quantity=_decimal(raw.qty), side=_text(raw.side),
            market_value=_decimal(raw.market_value), cost_basis=_decimal(raw.cost_basis),
            average_entry_price=_decimal(raw.avg_entry_price),
            unrealized_pl=_decimal(raw.unrealized_pl),
            unrealized_pl_fraction=_decimal(raw.unrealized_plpc),
        )

    @staticmethod
    def _order(raw: Any) -> BrokerOrder:
        status_text = _text(raw.status).lower()
        try:
            status = OrderStatus(status_text)
        except ValueError:
            status = OrderStatus.UNKNOWN
        side_text = _text(getattr(raw, "side", "")).lower()
        try:
            side = OrderSide(side_text)
        except ValueError:
            side = None
        return BrokerOrder(
            order_id=str(raw.id), client_order_id=str(getattr(raw, "client_order_id", "")),
            symbol=str(getattr(raw, "symbol", "")).upper(),
            quantity=_decimal(getattr(raw, "qty", 0)),
            filled_quantity=_decimal(getattr(raw, "filled_qty", 0)),
            side=side, order_type=_text(getattr(raw, "type", "unknown")).lower(),
            time_in_force=_text(getattr(raw, "time_in_force", "unknown")).lower(),
            status=status, raw_status=status_text,
            limit_price=_optional_decimal(getattr(raw, "limit_price", None)),
            filled_average_price=_optional_decimal(getattr(raw, "filled_avg_price", None)),
            submitted_at=_optional_aware(getattr(raw, "submitted_at", None)),
            updated_at=_optional_aware(getattr(raw, "updated_at", None)),
        )


def _same_intent(order: BrokerOrder, symbol: str, quantity: Decimal, side: OrderSide,
                 order_type: str, limit_price: Decimal | None, tif: TimeInForce) -> bool:
    return (
        order.symbol == symbol and order.quantity == quantity and order.side is side
        and order.order_type == order_type and order.limit_price == limit_price
        and order.time_in_force == tif.value
    )


def _same_replacement_intent(
    replacement: BrokerOrder, original: BrokerOrder,
    quantity: Decimal | None, limit_price: Decimal | None,
) -> bool:
    expected_quantity = quantity if quantity is not None else original.quantity
    expected_price = limit_price if limit_price is not None else original.limit_price
    return (
        replacement.symbol == original.symbol
        and replacement.side is original.side
        and replacement.order_type == original.order_type
        and replacement.quantity == expected_quantity
        and replacement.limit_price == expected_price
    )


def _translate_error(exc: Exception, *, operation: str, **context: object):
    status = _status_code(exc)
    details = {key: value for key, value in context.items() if value is not None}
    if status == 404:
        return OrderNotFound(f"Order was not found during {operation}", context=details)
    if status in {401, 403}:
        return BrokerAuthenticationError("Alpaca authentication or authorization failed", context=details)
    if status == 429:
        return BrokerRateLimitError("Alpaca rate limit remained after retry attempts", context=details)
    if status == 422:
        return BrokerRejectedOrder("Alpaca rejected the order", context=details)
    if isinstance(exc, (requests.Timeout, TimeoutError)):
        return BrokerRequestTimeout(f"Alpaca request timed out during {operation}", context=details)
    if isinstance(exc, (requests.ConnectionError, ConnectionError, OSError)):
        return BrokerConnectionError(f"Alpaca connection failed during {operation}", context=details)
    if status is not None and status >= 400:
        return BrokerError(f"Alpaca API request failed during {operation} (HTTP {status})", context=details)
    return BrokerError(f"Alpaca request failed during {operation}", context=details)


def _is_ambiguous_submission_failure(exc: Exception) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status >= 500
    try:
        from alpaca.common.exceptions import RetryException
    except ImportError:  # pragma: no cover - SDK dependency normally supplies it
        RetryException = None
    retry_exceptions = (RetryException,) if isinstance(RetryException, type) else ()
    return isinstance(
        exc, (requests.Timeout, requests.ConnectionError, TimeoutError, ConnectionError, OSError) + retry_exceptions
    )


def _is_duplicate_client_id_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "client_order_id" in text and any(word in text for word in ("duplicate", "unique", "already exists"))


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _optional_aware(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Alpaca returned an invalid order timestamp")
    return value.astimezone(timezone.utc)


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Broker returned a non-numeric account or order value") from exc
    if not result.is_finite():
        raise ValueError("Broker returned a non-finite account or order value")
    return result


def _optional_decimal(value: Any) -> Decimal | None:
    return None if value is None else _decimal(value)


def _positive_decimal(value: Decimal | float | int, name: str) -> Decimal:
    result = _decimal(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _symbol(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("symbol cannot be empty")
    return value.strip().upper()


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value.strip()


def _client_order_id(value: str) -> str:
    result = _required_text(value, "client_order_id")
    if len(result) > 48:
        raise ValueError("client_order_id cannot exceed 48 characters")
    return result


def _side(value: OrderSide) -> AlpacaOrderSide:
    try:
        return AlpacaOrderSide(value.value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("side must be OrderSide.BUY or OrderSide.SELL") from exc


def _tif(value: TimeInForce) -> AlpacaTimeInForce:
    try:
        return AlpacaTimeInForce(value.value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("Unsupported time in force") from exc


def _text(value: Any) -> str:
    return str(getattr(value, "value", value))
