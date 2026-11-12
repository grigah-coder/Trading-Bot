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
from trading_bot.logging_config import get_logger
from trading_bot.order.lifecycle import (
    STOP_LOSS,
    TAKE_PROFIT,
    Order,
    normalize_status,
    valid_transition,
)

logger = get_logger(__name__)


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

    def __init__(
        self,
        broker_adapter: Optional[Any] = None,
        *,
        submission_failure_window: int = 2,
    ) -> None:
        self.broker_adapter = broker_adapter
        self._lock = threading.RLock()
        # order_id -> Order
        self._orders: Dict[str, Order] = {}
        # client_order_id -> order_id (dedup index)
        self._by_client_order_id: Dict[str, str] = {}
        # signal_id -> order_id (dedup index)
        self._by_signal_id: Dict[str, str] = {}
        # order_id -> reconcile sequence number when submission failed.
        # Entries are dropped after ``submission_failure_window`` reconciles
        # so a transient failure cannot block a symbol forever.
        self._failed_submissions: Dict[str, int] = {}
        self._submission_failure_window = max(0, int(submission_failure_window))
        self._reconcile_seq = 0

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
        limit_price = _as_optional_float(
            order.get("limit_price") or order.get("price")
        )
        stop_price = _as_optional_float(order.get("stop_price"))
        use_bracket = stop_price is not None and limit_price is not None

        if not symbol:
            raise OrderStateError("order missing symbol")
        if side not in {"buy", "sell"}:
            raise OrderStateError(f"invalid order side {side!r}")
        if qty <= 0:
            raise OrderStateError(f"order quantity must be positive, got {qty!r}")

        with self._lock:
            # Idempotency: never submit the same client_order_id twice.
            if client_order_id and client_order_id in self._by_client_order_id:
                existing_id = self._by_client_order_id[client_order_id]
                existing = self._orders[existing_id]
                if existing_id in self._failed_submissions:
                    # Unresolved failed submission: report failure rather
                    # than a phantom pending order.
                    return None
                logger.info(
                    "Order %s already submitted via client_order_id %s; not resubmitting",
                    existing_id, client_order_id,
                )
                return existing.to_dict()

            # Idempotency: same signal should not spawn two orders.
            if signal_id and signal_id in self._by_signal_id:
                existing_id = self._by_signal_id[signal_id]
                existing = self._orders[existing_id]
                if existing_id in self._failed_submissions:
                    return None
                logger.info(
                    "Signal %s already has order %s; not resubmitting",
                    signal_id, existing_id,
                )
                return existing.to_dict()

            # Fallback ids are generated under the lock so the counter is safe.
            fallback_client_id = client_order_id or f"tb-{self._next_id():x}"
            fallback_signal_id = signal_id or f"sig-{self._next_id():x}"

            # Create a pending order first so we have an id even if the
            # broker response is lost (uncertain-timeout safety).
            pending = Order(
                order_id=order.get("order_id") or f"ord-{self._next_id():x}",
                client_order_id=client_order_id or fallback_client_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                signal_id=signal_id or fallback_signal_id,
                status="pending_new",
            )
            self._register_locked(pending)

            if self.broker_adapter is None:
                # Offline mode: record locally, never send anything.
                return pending.to_dict()

        # --- broker call, deliberately outside the lock ------------------
        try:
            submitted = self._submit_to_broker(
                pending, use_bracket=use_bracket,
            )
        except Exception as exc:  # noqa: BLE001 - typed by the adapter
            self._record_submission_failure(pending.order_id, exc)
            return None

        with self._lock:
            if submitted is None:
                self._record_submission_failure_locked(pending.order_id)
                return None

            # Broker acknowledged: adopt its ids/status and clear any failure
            # marker (an uncertain timeout may have been resolved here).
            self._failed_submissions.pop(pending.order_id, None)
            broker_id = str(
                submitted.get("order_id")
                or submitted.get("id")
                or pending.order_id
            )
            broker_client_id = str(
                submitted.get("client_order_id") or pending.client_order_id
            )
            if broker_id != pending.order_id:
                self._rekey_locked(pending.order_id, broker_id)
            pending.order_id = broker_id
            pending.client_order_id = broker_client_id
            pending.status = normalize_status(
                submitted.get("status", "new")
            )
            pending.filled_qty = float(submitted.get("filled_qty") or 0)
            pending.avg_fill_price = float(submitted.get("filled_avg_price") or 0)
            pending.bracket_legs = self._bracket_legs(submitted)
            pending.updated_at = submitted.get(
                "updated_at", datetime.now(timezone.utc).isoformat()
            )
            self._reindex_locked(pending)
            return pending.to_dict()

    def _submit_to_broker(
        self, pending: Order, *, use_bracket: bool,
    ) -> Optional[Dict[str, Any]]:
        """Send ``pending`` to the broker. Must be called without the lock."""
        if use_bracket:
            return self.broker_adapter.submit_bracket_order(
                symbol=pending.symbol,
                qty=pending.quantity,
                side=pending.side,
                stop_price=float(pending.stop_price),
                take_profit_price=float(pending.limit_price),
                client_order_id=pending.client_order_id,
                time_in_force=pending.time_in_force,
            )
        if pending.order_type == "limit":
            return self.broker_adapter.submit_limit_order(
                symbol=pending.symbol,
                qty=pending.quantity,
                side=pending.side,
                limit_price=(
                    float(pending.limit_price)
                    if pending.limit_price is not None
                    else 0.0
                ),
                client_order_id=pending.client_order_id,
                time_in_force=pending.time_in_force,
            )
        return self.broker_adapter.submit_market_order(
            symbol=pending.symbol,
            qty=pending.quantity,
            side=pending.side,
            client_order_id=pending.client_order_id,
            time_in_force=pending.time_in_force,
        )

    def _record_submission_failure(
        self, order_id: str, exc: BaseException,
    ) -> None:
        """Track a failed submission without holding the caller's lock."""
        with self._lock:
            self._record_submission_failure_locked(order_id)
        logger.warning(
            "Order submission failed for %s (%s): %s",
            order_id,
            type(exc).__name__,
            exc,
        )

    def _record_submission_failure_locked(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if order is None:
            return
        # Keep the order and its dedup indices for a bounded number of
        # reconciles so a late broker order can still be adopted. The engine
        # receives ``None`` and counts the attempt as failed.
        self._failed_submissions[order_id] = self._reconcile_seq

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
                # A bracket exit leg filled: it is not tracked as its own
                # order, only as a leg id on the parent. Forget the leg so it
                # is never cancelled or re-used.
                for tracked in self._orders.values():
                    for kind, leg_id in list(tracked.bracket_legs.items()):
                        if leg_id == order_id:
                            tracked.bracket_legs.pop(kind, None)
                            logger.info(
                                "Bracket leg %s (%s) filled for %s",
                                leg_id, kind, tracked.symbol,
                            )
                            return
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
            self._reconcile_seq += 1
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
                # A submission whose response was lost is adopted here: take
                # the broker's order id and bracket legs, and clear the
                # failure marker so the dedup index is kept normally.
                broker_id = str(
                    broker_raw.get("order_id") or broker_raw.get("id") or ""
                )
                if broker_id and broker_id != order.order_id and not order.is_terminal:
                    old_id = order.order_id
                    self._rekey_locked(old_id, broker_id)
                    self._failed_submissions.pop(old_id, None)
                    self._failed_submissions.pop(order.order_id, None)
                    self._reindex_locked(order)
                legs = self._bracket_legs(broker_raw)
                if legs:
                    order.bracket_legs = legs
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

            self._prune_failed_submissions_locked()

    def cancel_bracket_legs(self, symbol: str) -> List[str]:
        """Cancel every tracked protective leg for ``symbol``.

        Returns the ids the broker reported as cancelled (or already gone).
        Cancellation is best-effort per leg: one failed leg does not stop the
        remaining legs from being cancelled.
        """
        sym = str(symbol or "").strip().upper()
        if not sym:
            return []
        with self._lock:
            leg_ids = {
                leg_id
                for order in self._orders.values()
                if order.symbol == sym
                for leg_id in order.bracket_legs.values()
                if leg_id
            }
        cancelled: List[str] = []
        for leg_id in sorted(leg_ids):
            try:
                if self.broker_adapter is None or self.broker_adapter.cancel_order(leg_id):
                    cancelled.append(leg_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to cancel bracket leg %s for %s: %s",
                    leg_id, sym, exc,
                )
        with self._lock:
            for order in self._orders.values():
                if order.symbol == sym:
                    order.bracket_legs = {
                        kind: leg_id
                        for kind, leg_id in order.bracket_legs.items()
                        if leg_id not in cancelled
                    }
        return cancelled

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
    @staticmethod
    def _bracket_legs(submitted: Dict[str, Any]) -> Dict[str, str]:
        """Map broker bracket leg ids to ``stop_loss`` / ``take_profit``."""
        legs: Dict[str, str] = {}
        for leg in submitted.get("legs") or ():
            leg_id = str(leg.get("order_id") or leg.get("id") or "")
            if not leg_id:
                continue
            order_type = str(leg.get("order_type", "")).lower()
            if "stop" in order_type:
                legs.setdefault(STOP_LOSS, leg_id)
            elif "limit" in order_type:
                legs.setdefault(TAKE_PROFIT, leg_id)
            elif leg.get("stop_price") not in (None, "", 0.0):
                legs.setdefault(STOP_LOSS, leg_id)
            elif leg.get("limit_price") not in (None, "", 0.0):
                legs.setdefault(TAKE_PROFIT, leg_id)
        return legs

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

    def _prune_failed_submissions_locked(self) -> None:
        """Drop dedup indices for failed submissions after a bounded window."""
        if not self._failed_submissions:
            return
        window = self._submission_failure_window
        expired = [
            order_id
            for order_id, seq in list(self._failed_submissions.items())
            if self._reconcile_seq - seq >= window
        ]
        for order_id in expired:
            self._failed_submissions.pop(order_id, None)
            order = self._orders.get(order_id)
            if order is None:
                continue
            # A late broker order was adopted above when it matched, so only
            # orders still stuck in pending_new are discarded.
            if order.status != "pending_new":
                continue
            self._orders.pop(order_id, None)
            if order.client_order_id:
                self._by_client_order_id.pop(order.client_order_id, None)
            if order.signal_id:
                self._by_signal_id.pop(order.signal_id, None)
            logger.info(
                "Dropped failed submission %s after %d reconcile(s)",
                order_id, window,
            )
