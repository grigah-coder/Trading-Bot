"""Unit tests for the Phase 7 order lifecycle and order manager."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from trading_bot.exceptions import OrderStateError
from trading_bot.order.lifecycle import (
    Order,
    is_open,
    is_terminal,
    normalize_status,
    valid_transition,
)
from trading_bot.order.manager import OrderManager

# ---------------------------------------------------------------------------
# Fake broker adapter
# ---------------------------------------------------------------------------


class FakeBroker:
    """Stand-in for AlpacaBrokerAdapter with market/limit/bracket submission."""

    def __init__(self) -> None:
        self.submissions: List[Dict[str, Any]] = []
        self.fail_next = False
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.cancelled: List[str] = []

    def submit_market_order(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        self.submissions.append(("market", dict(kwargs)))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("broker down")
        oid = kwargs.get("client_order_id") or f"broker-{len(self.submissions)}"
        order = {
            "order_id": oid,
            "client_order_id": kwargs.get("client_order_id"),
            "symbol": kwargs["symbol"],
            "side": kwargs["side"],
            "qty": kwargs["qty"],
            "order_type": "market",
            "status": "new",
            "filled_qty": 0,
            "filled_avg_price": 0,
            "time_in_force": kwargs.get("time_in_force", "day"),
        }
        self.orders[oid] = order
        return order

    def submit_limit_order(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        self.submissions.append(("limit", dict(kwargs)))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("broker down")
        oid = kwargs.get("client_order_id") or f"broker-{len(self.submissions)}"
        order = {
            "order_id": oid,
            "client_order_id": kwargs.get("client_order_id"),
            "symbol": kwargs["symbol"],
            "side": kwargs["side"],
            "qty": kwargs["qty"],
            "order_type": "limit",
            "status": "new",
            "filled_qty": 0,
            "filled_avg_price": 0,
            "limit_price": kwargs.get("limit_price"),
            "time_in_force": kwargs.get("time_in_force", "day"),
        }
        self.orders[oid] = order
        return order

    def submit_bracket_order(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        self.submissions.append(("bracket", dict(kwargs)))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("broker down")
        oid = kwargs.get("client_order_id") or f"broker-{len(self.submissions)}"
        order = {
            "order_id": oid,
            "client_order_id": kwargs.get("client_order_id"),
            "symbol": kwargs["symbol"],
            "side": kwargs["side"],
            "qty": kwargs["qty"],
            "order_type": "market",
            "order_class": "bracket",
            "status": "new",
            "filled_qty": 0,
            "filled_avg_price": 0,
            "stop_price": kwargs.get("stop_price"),
            "limit_price": kwargs.get("take_profit_price"),
            "time_in_force": kwargs.get("time_in_force", "day"),
            "legs": [
                {"order_id": "sl-1", "order_type": "stop",
                 "stop_price": kwargs.get("stop_price")},
                {"order_id": "tp-1", "order_type": "limit",
                 "limit_price": kwargs.get("take_profit_price")},
            ],
        }
        self.orders[oid] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Record and acknowledge a cancellation request."""
        self.cancelled.append(order_id)
        order = self.orders.get(order_id)
        if order is not None:
            order["status"] = "canceled"
        return True


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestOrderStatusHelpers:
    def test_open_states(self):
        assert is_open("new")
        assert is_open("partially_filled")
        assert not is_open("filled")
        assert not is_open("canceled")

    def test_terminal_states(self):
        assert is_terminal("filled")
        assert is_terminal("canceled")
        assert is_terminal("cancelled")
        assert is_terminal("rejected")
        assert not is_terminal("new")

    def test_normalize_status(self):
        assert normalize_status("FILLED") == "filled"
        assert normalize_status("Canceled") == "canceled"
        assert normalize_status("") == ""

    def test_valid_transition(self):
        assert valid_transition("new", "filled")
        assert valid_transition("new", "partially_filled")
        assert valid_transition("new", "canceled")
        assert valid_transition("filled", "filled")  # self-loop ok
        assert not valid_transition("filled", "new")  # terminal can't un-fill

    def test_unknown_status_lenient(self):
        assert valid_transition("unknown", "filled") is True


class TestOrder:
    def test_defaults(self):
        o = Order(
            order_id="o1", client_order_id="c1", symbol="AAPL",
            side="buy", quantity=10.0,
        )
        assert o.status == "pending_new"
        assert o.is_open
        assert not o.is_terminal
        assert not o.is_filled
        assert o.remaining_qty == 10.0
        assert o.fill_pct == 0.0

    def test_fill_partial(self):
        o = Order(order_id="o1", client_order_id="c1", symbol="AAPL",
                  side="buy", quantity=10.0)
        o.fill(4.0, 100.0)
        assert o.filled_qty == 4.0
        assert o.avg_fill_price == 100.0
        assert o.status == "partially_filled"
        assert o.fill_pct == 0.4

    def test_fill_full_becomes_filled(self):
        o = Order(order_id="o1", client_order_id="c1", symbol="AAPL",
                  side="buy", quantity=10.0)
        o.fill(6.0, 100.0)
        o.fill(4.0, 101.0)
        assert o.status == "filled"
        assert o.avg_fill_price == pytest.approx(100.4)
        assert o.remaining_qty == 0.0
        assert o.fill_pct == 1.0

    def test_cancel(self):
        o = Order(order_id="o1", client_order_id="c1", symbol="AAPL",
                  side="buy", quantity=10.0)
        o.cancel()
        assert o.is_cancelled
        assert o.is_terminal

    def test_to_dict_roundtrip(self):
        o = Order(order_id="o1", client_order_id="c1", symbol="aapl",
                  side="BUY", quantity=10.0, order_type="limit",
                  stop_price=95.0, limit_price=105.0)
        d = o.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["side"] == "buy"
        assert d["order_type"] == "limit"
        assert d["stop_price"] == 95.0
        assert d["limit_price"] == 105.0
        assert d["order_id"] == "o1"

    def test_from_broker_dict(self):
        raw = {
            "order_id": "br-1", "client_order_id": "cid-1",
            "symbol": "msft", "side": "sell", "qty": 5.0,
            "order_type": "market", "status": "FILLED",
            "filled_qty": 5.0, "filled_avg_price": 150.0,
        }
        o = Order.from_broker_dict(raw)
        assert o.order_id == "br-1"
        assert o.symbol == "MSFT"
        assert o.side == "sell"
        assert o.status == "filled"
        assert o.is_filled
        assert o.is_terminal


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------


class TestOrderManagerSubmit:
    def test_submit_market_via_broker(self):
        broker = FakeBroker()
        mgr = OrderManager(broker)
        result = mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 10.0,
            "order_type": "market", "signal_id": "sig-1",
        })
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["side"] == "buy"
        assert result["status"] == "new"
        assert len(broker.submissions) == 1
        assert broker.submissions[0][0] == "market"
        assert broker.submissions[0][1]["qty"] == 10.0
        assert broker.submissions[0][1]["client_order_id"] == result["client_order_id"]

    def test_submit_limit_passes_limit_price(self):
        broker = FakeBroker()
        mgr = OrderManager(broker)
        result = mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 2.0,
            "order_type": "limit", "limit_price": 150.0,
            "signal_id": "sig-2",
        })
        assert result is not None
        assert len(broker.submissions) == 1
        assert broker.submissions[0][0] == "limit"
        assert broker.submissions[0][1]["limit_price"] == 150.0

    def test_submit_bracket_when_stop_and_limit_present(self):
        """Orders carrying both protective prices route to bracket submission."""
        broker = FakeBroker()
        mgr = OrderManager(broker)
        result = mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 3.0,
            "order_type": "market", "stop_price": 98.0,
            "limit_price": 104.0, "signal_id": "sig-3",
        })
        assert result is not None
        assert len(broker.submissions) == 1
        kind, kwargs = broker.submissions[0]
        assert kind == "bracket"
        assert kwargs["qty"] == 3.0
        assert kwargs["stop_price"] == 98.0
        assert kwargs["take_profit_price"] == 104.0
        assert result["bracket_legs"] == {
            "stop_loss": "sl-1", "take_profit": "tp-1",
        }
        assert result["stop_price"] == 98.0
        assert result["limit_price"] == 104.0

    def test_cancel_bracket_legs_cancels_tracked_legs(self):
        """Exits cancel the tracked stop/TP legs before selling the position."""
        broker = FakeBroker()
        mgr = OrderManager(broker)
        mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 3.0,
            "order_type": "market", "stop_price": 98.0,
            "limit_price": 104.0, "signal_id": "sig-4",
        })
        assert broker.cancelled == []
        cancelled = mgr.cancel_bracket_legs("AAPL")
        assert cancelled == ["sl-1", "tp-1"]
        assert broker.cancelled == ["sl-1", "tp-1"]
        assert mgr.orders_for_symbol("AAPL")[0]["bracket_legs"] == {}
        # Legs already gone are not cancelled twice.
        assert mgr.cancel_bracket_legs("AAPL") == []

    def test_cancel_bracket_legs_ignores_other_symbols(self):
        """Cancelling one symbol must not touch another symbol's legs."""
        broker = FakeBroker()
        mgr = OrderManager(broker)
        mgr.submit({
            "symbol": "MSFT", "side": "buy", "quantity": 1.0,
            "order_type": "market", "stop_price": 98.0,
            "limit_price": 104.0, "signal_id": "sig-5",
        })
        assert mgr.cancel_bracket_legs("AAPL") == []
        assert broker.cancelled == []

    def test_submit_no_broker_records_locally(self):
        mgr = OrderManager(None)
        result = mgr.submit({"symbol": "AAPL", "side": "buy", "quantity": 3.0})
        assert result is not None
        assert result["status"] == "pending_new"
        assert mgr.get_order(result["order_id"]) is not None

    def test_submit_rejects_missing_symbol(self):
        mgr = OrderManager(FakeBroker())
        with pytest.raises(OrderStateError):
            mgr.submit({"side": "buy", "quantity": 1.0})

    def test_submit_rejects_bad_side(self):
        mgr = OrderManager(FakeBroker())
        with pytest.raises(OrderStateError):
            mgr.submit({"symbol": "AAPL", "side": "hold", "quantity": 1.0})

    def test_submit_rejects_nonpositive_qty(self):
        mgr = OrderManager(FakeBroker())
        with pytest.raises(OrderStateError):
            mgr.submit({"symbol": "AAPL", "side": "buy", "quantity": 0.0})

    def test_broker_failure_returns_none_and_blocks_briefly(self):
        """A failed submission is reported as failed, not as a pending order."""
        broker = FakeBroker()
        broker.fail_next = True
        mgr = OrderManager(broker, submission_failure_window=2)
        result = mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 1.0,
            "client_order_id": "cid-fail",
        })
        assert result is None
        # The dedup index survives the bounded reconcile window, so an
        # immediate duplicate signal does not create a second order.
        assert mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 1.0,
            "client_order_id": "cid-fail",
        }) is None
        assert len(broker.submissions) == 1

    def test_failed_submission_indices_expire_after_window(self):
        """A failed dedup index is cleaned up so a retry can be submitted."""
        broker = FakeBroker()
        broker.fail_next = True
        mgr = OrderManager(broker, submission_failure_window=1)
        assert mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 1.0,
            "client_order_id": "cid-retry",
        }) is None

        mgr.reconcile([], [])  # window reached -> indices dropped
        result = mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 1.0,
            "client_order_id": "cid-retry",
        })
        assert result is not None
        assert len(broker.submissions) == 2

    def test_broker_call_does_not_hold_the_manager_lock(self):
        """The broker call runs unlocked so other threads can query state."""
        import threading

        acquired = threading.Event()

        class SlowBroker(FakeBroker):
            def submit_market_order(self, **kwargs):
                def query():
                    manager.get_active_orders()
                    acquired.set()

                worker = threading.Thread(target=query)
                worker.start()
                worker.join(timeout=2.0)
                return super().submit_market_order(**kwargs)

        broker = SlowBroker()
        manager = OrderManager(broker)
        result = manager.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 1.0,
        })
        assert result is not None
        assert acquired.is_set()

    def test_dedup_by_client_order_id(self):
        broker = FakeBroker()
        mgr = OrderManager(broker)
        first = mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 1.0,
            "client_order_id": "cid-1",
        })
        second = mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 1.0,
            "client_order_id": "cid-1",
        })
        assert first["order_id"] == second["order_id"]
        assert len(broker.submissions) == 1  # not double-submitted

    def test_dedup_by_signal_id(self):
        broker = FakeBroker()
        mgr = OrderManager(broker)
        first = mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 1.0,
            "signal_id": "sig-1",
        })
        second = mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 1.0,
            "signal_id": "sig-1",
        })
        assert first["order_id"] == second["order_id"]
        assert len(broker.submissions) == 1


class TestOrderManagerFillAndReconcile:
    def test_on_fill_partial(self):
        broker = FakeBroker()
        mgr = OrderManager(broker)
        result = mgr.submit({"symbol": "AAPL", "side": "buy", "quantity": 10.0,
                             "signal_id": "sig-1"})
        oid = result["order_id"]
        mgr.on_fill({"order_id": oid, "filled_qty": 4.0, "fill_price": 100.0})
        o = mgr.get_order(oid)
        assert o["filled_qty"] == 4.0
        assert o["status"] == "partially_filled"

    def test_on_fill_full_marks_filled(self):
        broker = FakeBroker()
        mgr = OrderManager(broker)
        result = mgr.submit({"symbol": "AAPL", "side": "buy", "quantity": 10.0,
                             "signal_id": "sig-1"})
        oid = result["order_id"]
        mgr.on_fill({"order_id": oid, "filled_qty": 10.0, "fill_price": 100.0})
        o = mgr.get_order(oid)
        assert o["status"] == "filled"
        assert o["is_filled"] is True

    def test_on_fill_bracket_leg_forgets_that_leg(self):
        """A filled exit leg is removed from the parent's live legs."""
        broker = FakeBroker()
        mgr = OrderManager(broker)
        mgr.submit({
            "symbol": "AAPL", "side": "buy", "quantity": 3.0,
            "order_type": "market", "stop_price": 98.0,
            "limit_price": 104.0, "signal_id": "sig-6",
        })
        mgr.on_fill({"order_id": "tp-1", "filled_qty": 3.0,
                     "fill_price": 104.0})
        assert mgr.orders_for_symbol("AAPL")[0]["bracket_legs"] == {
            "stop_loss": "sl-1",
        }

    def test_on_fill_unknown_order_is_noop(self):
        broker = FakeBroker()
        mgr = OrderManager(broker)
        mgr.on_fill({"order_id": "does-not-exist", "filled_qty": 1.0})
        assert mgr.get_recent() == []

    def test_reconcile_marks_terminal(self):
        broker = FakeBroker()
        mgr = OrderManager(broker)
        result = mgr.submit({"symbol": "AAPL", "side": "buy", "quantity": 10.0,
                             "signal_id": "sig-1"})
        oid = result["order_id"]
        mgr.reconcile([], [
            {"order_id": oid, "client_order_id": "x", "symbol": "AAPL",
             "side": "buy", "qty": 10.0, "order_type": "market",
             "status": "filled", "filled_qty": 10.0, "filled_avg_price": 100.0},
        ])
        # Terminal order removed from active tracking.
        assert oid not in {o2["order_id"] for o2 in mgr.get_active_orders()}
        assert oid not in {o2["order_id"] for o2 in mgr.get_recent()}

    def test_get_active_orders_only_open(self):
        broker = FakeBroker()
        mgr = OrderManager(broker)
        mgr.submit({"symbol": "AAPL", "side": "buy", "quantity": 1.0, "signal_id": "a"})
        mgr.submit({"symbol": "MSFT", "side": "sell", "quantity": 1.0, "signal_id": "b"})
        active = mgr.get_active_orders()
        assert len(active) == 2

    def test_orders_for_symbol(self):
        broker = FakeBroker()
        mgr = OrderManager(broker)
        mgr.submit({"symbol": "AAPL", "side": "buy", "quantity": 1.0, "signal_id": "a"})
        mgr.submit({"symbol": "MSFT", "side": "sell", "quantity": 1.0, "signal_id": "b"})
        aapl = mgr.orders_for_symbol("AAPL")
        assert len(aapl) == 1
        assert aapl[0]["symbol"] == "AAPL"
