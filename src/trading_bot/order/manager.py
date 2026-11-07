"""Order manager.

Implements the :class:`trading_bot.engine.OrderManager` protocol: submission,
fill handling and reconciliation. It wraps a broker adapter (the Alpaca paper
adapter) and keeps an in-memory view of every order the bot has submitted,
deduplicating submissions by ``client_order_id`` so a retry never double-fills.

Responsibilities
----------------
* Translate an engine order dict into a broker ``submit_market_order`` /
  ``submit_limit_order`` call.
* Track each order's lifecycle (pending -> open -> filled/cancelled/rejected).
* Apply fills (full or partial) via :meth:`on_fill`.
* Reconcile the in-memory view against broker-reported positions and orders
  via :meth:`reconcile`, marking terminal orders and surfacing any fills the
  engine missed.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_bot.broker.adapter import TERMINAL_STATES
from trading_bot.exceptions import OrderStateError
from trading_bot.logging_config import configure_logging
from trading_bot.order.lifecycle import Order, normalize_status, valid_transition

logger = configure_logging(__name__).getChild("order_manager")


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class OrderManager:
    """Order submission, fill tracking and reconciliation.

    Parameters
    ----------
    broker_adapter:
        Broker adapter used to actually place orders. May be ``None`` in
        which case submissions are recorded locally but never sent to the
        broker (useful for offline tests).

    """

    def __init__(self, broker_adapter: Optional[Any] = None) -> None:
        self.broker_adapter = broker_adapter
        self._lock = threading.RLock()
        # order_id -> Order
        self._orders: Dict[str, Order] = {}
        # client_order_id -> order_id (dedup index)
        self._by_client_order_id: Dict[str, str] = {}
        # signal_id -> order_id (dedup index)
        self._by_signal_id: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # OrderManager protocol
    # ------------------------------------------------------------------
    def submit(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Submit an order to the broker and track it locally.

        Returns the normalized order dict, or ``None`` when the order could
        not be placed and no existing order was found.
        """
        order = dict(order or {})
        symbol = str(order.get("symbol", "")).upper()
        side = str(order.get("side", "buy")).strip().lower()
        qty = float(order.get("quantity") or order.get("qty") or 0)
        order_type = str(order.get("order_type", "market")).strip().lower()
        client_order_id = str(order.get("client_order_id") or "")
        signal_id = str(order.get("signal_id") or "")
        time_in_force = str(order.get("time_in_force", "day")).strip().lower()
        limit_price = order.get("limit_price") or order.get("price")
        stop_price = order.get("stop_price")

        if not symbol:
            raise OrderStateError("order missing symbol")
        if side not in {"buy", "sell"}:
            raise OrderStateError(f"invalid order side {side!r}")
        if qty <= 0:
            raise OrderStateError(f"order quantity must be positive, got {qty!r}")

        # Fallback ids so every tracked order has a client_order_id.
        fallback_client_id = client_order_id or f"tb-{self._next_id():x}"
        fallback_signal_id = signal_id or f"sig-{self._next_id():x}"

        with self._lock:
            # Idempotency: never submit the same client_order_id twice.
            if client_order_id and client_order_id in self._by_client_order_id:
                existing_id = self._by_client_order_id[client_order_id]
                existing = self._orders[existing_id]
                logger.info(
                    "Order %s already submitted via client_order_id %s; not resubmitting",
                    existing_id, client_order_id,
                )
                return existing.to_dict()

            # Idempotency: same signal should not spawn two orders.
            if signal_id and signal_id in self._by_signal_id:
                existing_id = self._by_signal_id[signal_id]
                existing = self._orders[existing_id]
                logger.info(
                    "Signal %s already has order %s; not resubmitting",
                    signal_id, existing_id,
                )
                return existing.to_dict()

            # Create a pending order first so we have an id even if the
            # broker response is lost (uncertain-timeout safety).
            pending = Order(
                order_id=order.get("order_id") or f"ord-{self._next_id():x}",
                client_order_id=client_order_id or fallback_client_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type=order_type,
                limit_price=_as_optional_float(limit_price),
                stop_price=_as_optional_float(stop_price),
                time_in_force=time_in_force,
                signal_id=signal_id or fallback_signal_id,
                status="pending_new",
            )
            # register indices before the broker call so a retry cannot race
            self._register_locked(pending)

            submitted: Optional[Dict[str, Any]] = None
            if self.broker_adapter is not None:
                try:
                    if order_type == "limit":
                        submitted = self.broker_adapter.submit_limit_order(
                            symbol=symbol,
                            qty=qty,
                            side=side,
                            limit_price=float(limit_price) if limit_price is not None else 0.0,
                            client_order_id=pending.client_order_id,
                            time_in_force=time_in_force,
                        )
                    else:
                        submitted = self.broker_adapter.submit_market_order(
                            symbol=symbol,
                            qty=qty,
                            side=side,
                            client_order_id=pending.client_order_id,
                            time_in_force=time_in_force,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Order submission failed for %s: %s", symbol, exc)
                    # Leave the order in pending_new; reconcile can sort it out.
                    return pending.to_dict()

            if submitted is not None:
                # Broker acknowledged: adopt its ids/status.
                broker_id = str(
                    submitted.get("order_id")
                    or submitted.get("id")
                    or pending.order_id
                )
                broker_client_id = str(submitted.get("client_order_id") or pending.client_order_id)
                if broker_id != pending.order_id:
                    self._rekey_locked(pending.order_id, broker_id)
                pending.order_id = broker_id
                pending.client_order_id = broker_client_id
                pending.status = normalize_status(
                    submitted.get("status", "new")
                )
                pending.filled_qty = float(submitted.get("filled_qty") or 0)
                pending.avg_fill_price = float(submitted.get("filled_avg_price") or 0)
                pending.updated_at = submitted.get(
                    "updated_at", datetime.now(timezone.utc).isoformat()
                )
                self._reindex_locked(pending)
            else:
                # No acknowledgement: keep pending_new (uncertain state).
                pending.status = "pending_new"

            return pending.to_dict()

    def on_fill(self, fill: Dict[str, Any]) -> None:
        """Apply a fill (full or partial) to a tracked order.

        ``fill`` may carry ``order_id``, ``client_order_id``, ``filled_qty``
        and ``fill_price`` / ``price``.
        """
        if not fill:
            return
        order_id = str(fill.get("order_id") or "")
        client_order_id = str(fill.get("client_order_id") or "")
        fill_qty = float(fill.get("filled_qty") or fill.get("qty") or 0)
        fill_price = fill.get("fill_price") or fill.get("price")

        with self._lock:
            order = self._orders.get(order_id)
            if order is None and client_order_id:
                order = self._orders.get(self._by_client_order_id.get(client_order_id, ""))
            if order is None:
                logger.warning("Fill received for unknown order %s", order_id or client_order_id)
                return
            if fill_qty > 0 and fill_price is not None:
                order.fill(fill_qty, float(fill_price))
            elif order.is_terminal:
                # A fill report with no qty for an already-terminal order is a dup.
                return
            else:
                # Fill acknowledged but qty unknown: mark filled if terminal per broker.
                broker_status = normalize_status(fill.get("status", ""))
                if broker_status in TERMINAL_STATES and broker_status == "filled":
                    order.transition("filled")

    def reconcile(
        self, positions: List[Dict[str, Any]], orders: List[Dict[str, Any]]
    ) -> None:
        """Sync tracked orders with broker-reported positions and orders.

        Terminal broker orders are marked terminal locally; any newly-observed
        fill is applied. Orders that are terminal in both views are dropped
        from active tracking (kept in ``recent`` via ``get_recent``).
        """
        with self._lock:
            broker_by_id: Dict[str, Dict[str, Any]] = {}
            broker_by_client: Dict[str, Dict[str, Any]] = {}
            for raw in orders or ():
                oid = str(raw.get("order_id") or raw.get("id") or "")
                if oid:
                    broker_by_id[oid] = raw
                cid = str(raw.get("client_order_id") or "")
                if cid:
                    broker_by_client[cid] = raw

            newly_terminal: List[str] = []
            for order in list(self._orders.values()):
                broker_raw = broker_by_id.get(order.order_id) or broker_by_client.get(
                    order.client_order_id
                )
                if broker_raw is None:
                    continue
                broker_status = normalize_status(broker_raw.get("status", ""))
                if broker_status and broker_status != order.status:
                    if valid_transition(order.status, broker_status):
                        order.transition(broker_status)
                    else:
                        logger.debug(
                            "Order %s broker status %s not a valid transition from %s; keeping local",
                            order.order_id, broker_status, order.status,
                        )
                # Apply any fill delta the broker reports.
                broker_filled = float(broker_raw.get("filled_qty") or 0)
                if broker_filled > order.filled_qty and broker_raw.get("filled_avg_price") not in (None, ""):
                    order.fill(
                        broker_filled - order.filled_qty,
                        float(broker_raw.get("filled_avg_price")),
                    )
                if order.is_terminal:
                    newly_terminal.append(order.order_id)

            for oid in newly_terminal:
                order = self._orders.pop(oid, None)
                if order:
                    if order.client_order_id:
                        self._by_client_order_id.pop(order.client_order_id, None)
                    if order.signal_id:
                        self._by_signal_id.pop(order.signal_id, None)
                    logger.info(
                        "Order %s reached terminal state %s; removed from active tracking",
                        oid, order.status,
                    )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            order = self._orders.get(order_id)
            return order.to_dict() if order else None

    def get_active_orders(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [o.to_dict() for o in self._orders.values() if o.is_open]

    def get_recent(self) -> List[Dict[str, Any]]:
        """All tracked orders including terminal ones still in memory."""
        with self._lock:
            return [o.to_dict() for o in self._orders.values()]

    def orders_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        sym = symbol.upper()
        with self._lock:
            return [
                o.to_dict()
                for o in self._orders.values()
                if o.symbol == sym
            ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _next_id(self) -> int:
        # Monotonic counter for locally-generated ids.
        if not hasattr(self, "_id_counter"):
            self._id_counter = 0
        self._id_counter += 1
        return self._id_counter

    def _register_locked(self, order: Order) -> None:
        self._orders[order.order_id] = order
        if order.client_order_id:
            self._by_client_order_id[order.client_order_id] = order.order_id
        if order.signal_id:
            self._by_signal_id[order.signal_id] = order.order_id

    def _reindex_locked(self, order: Order) -> None:
        # Rebuild indices after id/client_order_id may have changed.
        self._by_client_order_id = {
            cid: oid for cid, oid in self._by_client_order_id.items() if oid != order.order_id
        }
        self._by_signal_id = {
            sid: oid for sid, oid in self._by_signal_id.items() if oid != order.order_id
        }
        self._register_locked(order)

    def _rekey_locked(self, old_id: str, new_id: str) -> None:
        order = self._orders.pop(old_id, None)
        if order is None:
            return
        order.order_id = new_id
        self._orders[new_id] = order
