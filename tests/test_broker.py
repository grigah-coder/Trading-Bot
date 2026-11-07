from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from config.settings import BrokerSettings, ConfigurationError
from execution.broker import (
    BrokerConnectionError,
    BrokerRejectedOrder,
    IdempotencyConflict,
    OrderSide,
    OrderStatus,
    OrderSubmissionUnknown,
)
from execution.alpaca_broker import AlpacaBrokerAdapter

UTC = timezone.utc
NOW = datetime(2026, 11, 6, 15, 0, tzinfo=UTC)


def order(*, client_id: str = "intent-1", order_id: str = "order-1", status: str = "new",
          symbol: str = "AAPL", qty: str = "2", filled_qty: str = "0",
          side: str = "buy", order_type: str = "market", limit_price: str | None = None) -> Any:
    return SimpleNamespace(
        id=order_id, client_order_id=client_id, symbol=symbol, qty=qty,
        filled_qty=filled_qty, side=side, type=order_type, status=status,
        time_in_force="day", limit_price=limit_price, filled_avg_price="11.25",
        submitted_at=NOW, updated_at=NOW,
    )


class FakeClient:
    def __init__(self) -> None:
        self.account = SimpleNamespace(id="acct-1", status="ACTIVE", currency="USD", cash="1000",
            buying_power="2000", equity="1500", portfolio_value="1500", daytrade_count=0,
            trading_blocked=False)
        self.positions = [SimpleNamespace(symbol="AAPL", qty="2", side="long", market_value="22",
            cost_basis="20", avg_entry_price="10", unrealized_pl="2", unrealized_plpc="0.1")]
        self.orders: list[Any] = []
        self.by_client: dict[str, Any] = {}
        self.by_id: dict[str, Any] = {}
        self.submissions: list[Any] = []
        self.fail_submit: BaseException | None = None
        self.fail_get: BaseException | None = None
        self.fail_cancel: BaseException | None = None
        self.fail_replace: BaseException | None = None
        self.last_filter = None

    def get_account(self): return self.account
    def get_all_positions(self): return self.positions
    def get_orders(self, *, filter):
        self.last_filter = filter
        if getattr(filter.status, "value", filter.status) == "open":
            return [o for o in self.orders if o.status not in {"filled", "canceled", "rejected"}]
        return self.orders
    def get_order_by_client_id(self, client_id):
        if self.fail_get:
            raise self.fail_get
        if client_id not in self.by_client:
            raise FakeAPIError(404, "not found")
        return self.by_client[client_id]
    def get_order_by_id(self, order_id):
        if order_id not in self.by_id:
            raise FakeAPIError(404, "not found")
        return self.by_id[order_id]
    def submit_order(self, order_data):
        self.submissions.append(order_data)
        if self.fail_submit:
            raise self.fail_submit
        placed = order(client_id=order_data.client_order_id, symbol=order_data.symbol,
            qty=str(order_data.qty), side=order_data.side.value,
            order_type=order_data.type.value,
            limit_price=str(getattr(order_data, "limit_price", "")) or None)
        self.by_client[placed.client_order_id] = placed
        self.by_id[placed.id] = placed
        return placed
    def cancel_order_by_id(self, order_id):
        if self.fail_cancel:
            raise self.fail_cancel
        existing = self.by_id[order_id]
        existing.status = "canceled"
    def replace_order_by_id(self, order_id, order_data):
        if self.fail_replace:
            raise self.fail_replace
        existing = self.by_id[order_id]
        replaced = order(client_id=order_data.client_order_id, order_id="order-2",
            status="new", symbol=existing.symbol,
            qty=str(order_data.qty or existing.qty), side=existing.side,
            order_type=existing.type,
            limit_price=str(order_data.limit_price or existing.limit_price or "") or None)
        self.by_client[replaced.client_order_id] = replaced
        self.by_id[replaced.id] = replaced
        return replaced


class FakeAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def direct_retry(operation, *args, **kwargs):
    for key in ("attempts", "base_delay", "description", "jitter", "retry_on", "max_delay", "backoff_factor"):
        kwargs.pop(key, None)
    return operation(*args, **kwargs)


def make_broker(client: FakeClient | None = None) -> tuple[AlpacaBrokerAdapter, FakeClient]:
    fake = client or FakeClient()
    factory = lambda key, secret, *, paper: fake
    return AlpacaBrokerAdapter(BrokerSettings("PK-paper", "paper-secret"),
                               client_factory=factory, retry=direct_retry), fake


def test_authentication_loads_environment_and_forces_paper_client() -> None:
    constructed = {}
    def factory(key, secret, *, paper):
        constructed.update(key=key, secret=secret, paper=paper)
        return FakeClient()
    AlpacaBrokerAdapter.from_env({"ALPACA_API_KEY": "PK-paper", "ALPACA_API_SECRET": "secret"},
                                 client_factory=factory, retry=direct_retry)
    assert constructed == {"key": "PK-paper", "secret": "secret", "paper": True}


def test_live_mode_configuration_is_rejected_before_client_creation() -> None:
    with pytest.raises(ConfigurationError):
        AlpacaBrokerAdapter.from_env({"ALPACA_API_KEY": "live-key", "ALPACA_API_SECRET": "secret",
                                      "ALPACA_PAPER": "false"})


def test_account_exposes_buying_power_and_equity() -> None:
    broker, _ = make_broker()
    account = broker.get_account()
    assert account.buying_power == Decimal("2000")
    assert account.equity == Decimal("1500")
    assert account.cash == Decimal("1000")


def test_open_positions_and_orders_are_normalized() -> None:
    broker, client = make_broker()
    client.orders = [order(status="new")]
    assert broker.get_open_positions()[0].symbol == "AAPL"
    assert broker.get_open_orders()[0].status is OrderStatus.NEW


def test_historical_orders_pass_aware_date_range() -> None:
    broker, client = make_broker()
    client.orders = [order()]
    result = broker.get_historical_orders(NOW - timedelta(days=1), NOW)
    assert result[0].client_order_id == "intent-1"
    assert client.last_filter.after == NOW - timedelta(days=1)
    assert client.last_filter.until == NOW


def test_market_and_limit_orders_use_official_request_models() -> None:
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
    broker, client = make_broker()
    market = broker.submit_market_order("aapl", Decimal("2"), OrderSide.BUY,
                                        client_order_id="market-1")
    limit = broker.submit_limit_order("msft", Decimal("3"), OrderSide.SELL,
                                      Decimal("42.50"), client_order_id="limit-1")
    assert market.status is OrderStatus.NEW
    assert isinstance(client.submissions[0], MarketOrderRequest)
    assert isinstance(client.submissions[1], LimitOrderRequest)
    assert client.submissions[1].limit_price == 42.5
    assert limit.symbol == "MSFT"


def test_rejected_order_is_reported_as_broker_rejection() -> None:
    broker, client = make_broker()
    client.fail_submit = FakeAPIError(422, "insufficient buying power")
    with pytest.raises(BrokerRejectedOrder):
        broker.submit_market_order("AAPL", 1, OrderSide.BUY, client_order_id="reject-1")


def test_partial_fill_and_unknown_status_are_preserved() -> None:
    broker, client = make_broker()
    client.by_id["partial"] = order(order_id="partial", status="partially_filled", filled_qty="1")
    client.by_id["mystery"] = order(order_id="mystery", status="brand_new_state")
    partial = broker.get_order("partial")
    unknown = broker.get_order("mystery")
    assert partial.status is OrderStatus.PARTIALLY_FILLED
    assert partial.filled_quantity == Decimal("1")
    assert partial.is_partially_filled
    assert unknown.status is OrderStatus.UNKNOWN
    assert unknown.raw_status == "brand_new_state"


def test_timeout_reconciles_by_client_id_without_resubmitting() -> None:
    broker, client = make_broker()
    def accepted_then_timed_out(order_data):
        client.submissions.append(order_data)
        placed = order(client_id=order_data.client_order_id)
        client.by_client[order_data.client_order_id] = placed
        raise requests.Timeout("read timed out")
    client.submit_order = accepted_then_timed_out
    result = broker.submit_market_order("AAPL", 2, OrderSide.BUY, client_order_id="uncertain-1")
    assert result.order_id == "order-1"
    assert len(client.submissions) == 1


def test_timeout_without_visible_order_is_unknown_and_not_retried() -> None:
    broker, client = make_broker()
    client.fail_submit = requests.Timeout("read timed out")
    with pytest.raises(OrderSubmissionUnknown):
        broker.submit_market_order("AAPL", 2, OrderSide.BUY, client_order_id="uncertain-2")
    assert len(client.submissions) == 1


def test_repeated_idempotency_key_returns_existing_order() -> None:
    broker, client = make_broker()
    client.by_client["intent-1"] = order(client_id="intent-1")
    result = broker.submit_market_order("AAPL", 2, OrderSide.BUY, client_order_id="intent-1")
    assert result.order_id == "order-1"
    assert client.submissions == []


def test_idempotency_key_cannot_be_reused_for_different_order() -> None:
    broker, client = make_broker()
    client.by_client["intent-1"] = order(client_id="intent-1")
    with pytest.raises(IdempotencyConflict):
        broker.submit_market_order("MSFT", 2, OrderSide.BUY, client_order_id="intent-1")


def test_cancel_replace_and_order_retrieval() -> None:
    broker, client = make_broker()
    original = order()
    client.by_id[original.id] = original
    canceled = broker.cancel_order(original.id)
    assert canceled.status is OrderStatus.CANCELED
    replaced = broker.replace_order(original.id, client_order_id="replace-1", quantity=4,
                                    limit_price=Decimal("10.50"))
    assert replaced.order_id == "order-2"
    assert broker.get_order(replaced.order_id).client_order_id == "replace-1"


def test_connection_failures_are_structured() -> None:
    broker, client = make_broker()
    client.fail_get = requests.ConnectionError("offline")
    with pytest.raises(BrokerConnectionError):
        broker.get_order_by_client_order_id("pending-intent")


def test_rate_limited_submission_retries_only_the_definitive_429() -> None:
    from core.retry import call_with_retry
    broker, client = make_broker()
    calls = 0
    def first_rate_limited(order_data):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FakeAPIError(429, "rate limited")
        return FakeClient.submit_order(client, order_data)
    client.submit_order = first_rate_limited
    broker._retry = lambda operation, *args, **kwargs: call_with_retry(
        operation, *args, jitter=False, **kwargs
    )
    result = broker.submit_market_order("AAPL", 1, OrderSide.BUY, client_order_id="rate-1")
    assert result.client_order_id == "rate-1"
    assert calls == 2


def test_non_transport_programming_failure_is_not_marked_ambiguous() -> None:
    broker, client = make_broker()
    client.fail_submit = ValueError("bad local request")
    with pytest.raises(Exception) as error:
        broker.submit_market_order("AAPL", 1, OrderSide.BUY, client_order_id="bad-1")
    assert not isinstance(error.value, OrderSubmissionUnknown)
    assert len(client.submissions) == 1


def test_alpaca_request_timeout_is_configured(monkeypatch) -> None:
    import requests
    called = {}
    def fake_request(session, method, url, **kwargs):
        called.update(kwargs)
        return object()
    monkeypatch.setattr(requests.Session, "request", fake_request)
    broker = AlpacaBrokerAdapter(BrokerSettings("PK-paper", "secret"), retry=direct_retry)
    broker._client._session.request("GET", "https://example.invalid")
    assert called["timeout"] == 10.0


def test_cancel_timeout_reconciles_order_without_resending():
    broker, client = make_broker()
    current = order()
    client.by_id[current.id] = current
    def cancel_then_timeout(order_id):
        current.status = "canceled"
        raise requests.Timeout("response lost")
    client.cancel_order_by_id = cancel_then_timeout
    result = broker.cancel_order(current.id)
    assert result.status is OrderStatus.CANCELED

def test_replace_timeout_reconciles_by_new_client_order_id():
    broker, client = make_broker()
    original = order()
    client.by_id[original.id] = original
    def replace_then_timeout(order_id, request):
        replacement = order(order_id="replacement", client_id=request.client_order_id,
                            qty=str(request.qty), symbol=original.symbol, side=original.side,
                            order_type=original.type, limit_price=str(request.limit_price))
        client.by_client[replacement.client_order_id] = replacement
        client.by_id[replacement.id] = replacement
        raise requests.Timeout("response lost")
    client.replace_order_by_id = replace_then_timeout
    result = broker.replace_order(original.id, client_order_id="replace-timeout",
                                  quantity=3, limit_price=Decimal("12.25"))
    assert result.order_id == "replacement"

def test_replacement_does_not_truncate_fractional_quantity():
    broker, _ = make_broker()
    with pytest.raises(ValueError, match="whole number"):
        broker.replace_order("order-1", client_order_id="replace-fraction", quantity=Decimal("1.5"))


def test_timeout_reconciliation_rejects_client_id_with_different_intent():
    broker, client = make_broker()
    def accept_different_intent_then_timeout(request):
        client.submissions.append(request)
        client.by_client[request.client_order_id] = order(
            client_id=request.client_order_id, symbol="MSFT", qty="1"
        )
        raise requests.Timeout("response lost")
    client.submit_order = accept_different_intent_then_timeout
    with pytest.raises(IdempotencyConflict):
        broker.submit_market_order("AAPL", 1, OrderSide.BUY, client_order_id="collision-1")


def test_replacement_client_id_conflict_is_rejected():
    broker, client = make_broker()
    original = order()
    client.by_id[original.id] = original
    client.by_client["replace-conflict"] = order(client_id="replace-conflict", symbol="MSFT", qty="4")
    with pytest.raises(IdempotencyConflict):
        broker.replace_order(original.id, client_order_id="replace-conflict", quantity=4)


def test_live_sdk_client_is_rejected_even_if_factory_is_injected():
    from alpaca.trading.client import TradingClient
    with pytest.raises(ValueError, match="paper endpoint"):
        AlpacaBrokerAdapter(BrokerSettings("PK-paper", "secret"),
                            client_factory=lambda key, secret, *, paper: TradingClient(
                                key, secret, paper=False
                            ))


def test_missing_credentials_fail_before_client_construction():
    called = False
    def factory(*args, **kwargs):
        nonlocal called
        called = True
        return FakeClient()
    with pytest.raises(ValueError, match="required"):
        AlpacaBrokerAdapter(BrokerSettings(), client_factory=factory)
    assert called is False


def test_historical_order_boundaries_must_be_aware_and_ordered():
    broker, _ = make_broker()
    with pytest.raises(ValueError, match="timezone-aware"):
        broker.get_historical_orders(datetime(2026, 11, 6), NOW)
    with pytest.raises(ValueError, match="before end"):
        broker.get_historical_orders(NOW, NOW)


def test_broker_reconciliation_prevents_duplicate_after_adapter_restart():
    first, client = make_broker()
    initial = first.submit_market_order("AAPL", 2, OrderSide.BUY, client_order_id="restart-safe")
    second = AlpacaBrokerAdapter(BrokerSettings("PK-paper", "paper-secret"),
        client_factory=lambda key, secret, *, paper: client, retry=direct_retry)
    recovered = second.submit_market_order("AAPL", 2, OrderSide.BUY, client_order_id="restart-safe")
    assert recovered.order_id == initial.order_id
    assert len(client.submissions) == 1
