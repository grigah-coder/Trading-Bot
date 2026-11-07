"""Broker-neutral interface and normalized account/order models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Sequence

from core.exceptions import TradingBotError


class BrokerError(TradingBotError):
    code = "broker_error"


class BrokerConnectionError(BrokerError):
    code = "broker_connection_failed"


class BrokerRequestTimeout(BrokerError):
    code = "broker_request_timeout"


class BrokerRateLimitError(BrokerError):
    code = "broker_rate_limited"


class BrokerAuthenticationError(BrokerError):
    code = "broker_authentication_failed"


class BrokerRejectedOrder(BrokerError):
    code = "broker_order_rejected"


class OrderNotFound(BrokerError):
    code = "broker_order_not_found"


class OrderSubmissionUnknown(BrokerError):
    code = "broker_order_submission_unknown"


class BrokerOperationUnknown(BrokerError):
    code = "broker_operation_unknown"


class IdempotencyConflict(BrokerError):
    code = "broker_idempotency_conflict"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    PENDING_REVIEW = "pending_review"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    STOPPED = "stopped"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"
    HELD = "held"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AccountInfo:
    account_id: str
    status: str
    currency: str
    cash: Decimal
    buying_power: Decimal
    equity: Decimal
    portfolio_value: Decimal
    day_trade_count: int
    trading_blocked: bool


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    quantity: Decimal
    side: str
    market_value: Decimal
    cost_basis: Decimal
    average_entry_price: Decimal
    unrealized_pl: Decimal
    unrealized_pl_fraction: Decimal


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    order_id: str
    client_order_id: str
    symbol: str
    quantity: Decimal
    filled_quantity: Decimal
    side: OrderSide | None
    order_type: str
    time_in_force: str
    status: OrderStatus
    raw_status: str
    limit_price: Decimal | None
    filled_average_price: Decimal | None
    submitted_at: datetime | None
    updated_at: datetime | None

    @property
    def is_partially_filled(self) -> bool:
        return 0 < self.filled_quantity < self.quantity


class Broker(ABC):
    """Provider-independent broker operations used by application services."""

    @abstractmethod
    def get_account(self) -> AccountInfo: ...

    @abstractmethod
    def get_open_positions(self) -> Sequence[Position]: ...

    @abstractmethod
    def get_open_orders(self) -> Sequence[BrokerOrder]: ...

    @abstractmethod
    def get_historical_orders(self, start: datetime, end: datetime) -> Sequence[BrokerOrder]: ...

    @abstractmethod
    def submit_market_order(self, symbol: str, quantity: Decimal | float | int, side: OrderSide,
                            *, client_order_id: str, time_in_force: TimeInForce = TimeInForce.DAY) -> BrokerOrder: ...

    @abstractmethod
    def submit_limit_order(self, symbol: str, quantity: Decimal | float | int, side: OrderSide,
                           limit_price: Decimal | float, *, client_order_id: str,
                           time_in_force: TimeInForce = TimeInForce.DAY) -> BrokerOrder: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> BrokerOrder: ...

    @abstractmethod
    def replace_order(self, order_id: str, *, client_order_id: str,
                      quantity: Decimal | float | int | None = None,
                      limit_price: Decimal | float | None = None) -> BrokerOrder: ...

    @abstractmethod
    def get_order(self, order_id: str) -> BrokerOrder: ...

    @abstractmethod
    def get_order_by_client_order_id(self, client_order_id: str) -> BrokerOrder: ...
