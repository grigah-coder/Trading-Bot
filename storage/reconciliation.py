"""Startup recovery and broker reconciliation.

The broker is authoritative.  The database is a durable record of what the bot
last observed, not a source of truth about the live account.  Startup therefore
loads persisted state, reads the broker, writes the broker's view back to the
database, and blocks new trades whenever the two cannot be reconciled safely.

Nothing in this module deletes historical rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, Sequence

from core.trading_gate import TradingGate
from execution.broker import (
    AccountInfo,
    Broker,
    BrokerOrder,
    OrderNotFound,
    Position,
)
from storage.database import TradingDatabase
from storage.models import (
    AccountSnapshotRow,
    ApplicationEventRow,
    OrderRow,
    PositionRow,
    RiskEventRow,
    StrategyVersionRow,
    context_json,
)
from storage.repositories import (
    OPEN_ORDER_STATUSES,
    TERMINAL_ORDER_STATUSES,
)
from storage.validation import (
    new_id,
    utc_now_timestamp,
    utc_timestamp,
    validate_symbol,
    validate_text,
)

_QTY_TOLERANCE = Decimal("0.00000001")


class BrokerReadOnly(Protocol):
    """The subset of the broker contract needed during reconciliation."""

    def get_account(self) -> AccountInfo: ...

    def get_open_positions(self) -> Sequence[Position]: ...

    def get_open_orders(self) -> Sequence[BrokerOrder]: ...

    def get_order(self, order_id: str) -> BrokerOrder: ...


@dataclass(frozen=True, slots=True)
class PersistedState:
    """Everything loaded from SQLite before touching the broker."""

    open_orders: tuple[OrderRow, ...] = ()
    positions: tuple[PositionRow, ...] = ()
    latest_account: AccountSnapshotRow | None = None
    active_strategy: StrategyVersionRow | None = None


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    kind: str
    severity: str
    detail: str
    symbol: str | None = None
    order_id: str | None = None
    client_order_id: str | None = None

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    completed: bool
    trading_allowed: bool
    persisted_state: PersistedState
    issues: tuple[ReconciliationIssue, ...] = ()
    adopted_orders: tuple[str, ...] = ()
    adopted_positions: tuple[str, ...] = ()
    account: AccountInfo | None = None
    error: str | None = None

    @property
    def errors(self) -> tuple[ReconciliationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.is_error)

    @property
    def warnings(self) -> tuple[ReconciliationIssue, ...]:
        return tuple(issue for issue in self.issues if not issue.is_error)


@dataclass(slots=True)
class StartupReconciliator:
    """Reconcile persisted state against the live broker account."""

    database: TradingDatabase
    broker: BrokerReadOnly
    trading_gate: TradingGate = field(default_factory=TradingGate)
    clock: Any = None
    last_result: ReconciliationResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.clock is None:
            self.clock = utc_now_timestamp

    # ------------------------------------------------------------------
    # Persisted state
    # ------------------------------------------------------------------
    def load_persisted_state(self) -> PersistedState:
        repositories = self.database.repositories
        return PersistedState(
            open_orders=tuple(repositories.orders.list_open()),
            positions=tuple(repositories.positions.latest_by_symbol()),
            latest_account=repositories.account_snapshots.latest(),
            active_strategy=repositories.strategy_versions.latest(),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def reconcile(self) -> ReconciliationResult:
        observed_at = utc_timestamp(self.clock(), "clock")
        try:
            state = self.load_persisted_state()
        except Exception as exc:  # corrupt or unavailable database
            return self._fail(
                "storage_unavailable",
                f"persisted state could not be loaded: {exc}",
                PersistedState(),
                observed_at,
            )

        self._record_event(
            "INFO",
            "reconciliation",
            "startup reconciliation started",
            {
                "open_orders": len(state.open_orders),
                "positions": len(state.positions),
                "has_account_snapshot": state.latest_account is not None,
                "has_strategy_version": state.active_strategy is not None,
            },
        )

        try:
            account = self.broker.get_account()
            broker_positions = tuple(self.broker.get_open_positions())
            broker_orders = tuple(self.broker.get_open_orders())
        except Exception as exc:
            return self._fail(
                "broker_unavailable",
                f"broker state could not be retrieved: {exc}",
                state,
                observed_at,
            )

        try:
            order_issues, adopted_orders = self._reconcile_orders(
                state.open_orders, broker_orders, observed_at
            )
            position_issues, adopted_positions = self._reconcile_positions(
                state.positions, broker_positions, observed_at
            )
        except Exception as exc:
            return self._fail(
                "reconciliation_error",
                f"reconciliation could not complete: {exc}",
                state,
                observed_at,
                account=account,
            )

        issues = tuple(order_issues + position_issues)
        failures = tuple(issue for issue in issues if issue.is_error)
        trading_allowed = not failures
        self._record_account_snapshot(
            account, observed_at, trading_blocked=account.trading_blocked or not trading_allowed
        )

        if trading_allowed:
            self.trading_gate.unblock("reconciliation")
            self._record_event(
                "INFO",
                "reconciliation",
                "startup reconciliation completed",
                {
                    "adopted_orders": len(adopted_orders),
                    "adopted_positions": len(adopted_positions),
                    "warnings": len(issues),
                },
            )
        else:
            summary = self._summarize(failures)
            self.trading_gate.block("reconciliation", summary)
            self._record_event(
                "ERROR",
                "reconciliation",
                "startup reconciliation found inconsistencies; trading is blocked",
                {"issues": [self._issue_context(issue) for issue in issues]},
            )
            self._record_risk_event(
                "reconciliation",
                "reconciliation_failed",
                summary,
                observed_at,
                context={
                    "issues": [self._issue_context(issue) for issue in issues],
                    "adopted_orders": list(adopted_orders),
                    "adopted_positions": list(adopted_positions),
                },
            )

        result = ReconciliationResult(
            completed=True,
            trading_allowed=trading_allowed,
            persisted_state=state,
            issues=issues,
            adopted_orders=tuple(adopted_orders),
            adopted_positions=tuple(adopted_positions),
            account=account,
        )
        self.last_result = result
        return result

    def require_trading_allowed(self) -> None:
        """Raise :class:`~core.trading_gate.TradingBlocked` when unsafe."""
        self.trading_gate.require_open()

    # ------------------------------------------------------------------
    # Order reconciliation
    # ------------------------------------------------------------------
    def _reconcile_orders(
        self,
        local_orders: Sequence[OrderRow],
        broker_orders: Sequence[BrokerOrder],
        observed_at: str,
    ) -> tuple[list[ReconciliationIssue], list[str]]:
        repositories = self.database.repositories
        issues: list[ReconciliationIssue] = []
        adopted: list[str] = []
        local_by_id = {order.order_id: order for order in local_orders}
        broker_by_id = {order.order_id: order for order in broker_orders}

        for broker_order in broker_orders:
            row = self._order_row(broker_order, observed_at)
            existing = repositories.orders.get_by_client_order_id(row.client_order_id)
            if existing is not None and existing.order_id != row.order_id:
                issues.append(
                    ReconciliationIssue(
                        kind="client_order_id_conflict",
                        severity="error",
                        detail=(
                            "broker order reuses a client_order_id already stored "
                            "for a different order_id"
                        ),
                        order_id=row.order_id,
                        client_order_id=row.client_order_id,
                    )
                )
            repositories.orders.upsert(row)
            if row.order_id not in local_by_id:
                adopted.append(row.order_id)
                issues.append(
                    ReconciliationIssue(
                        kind="open_order_adopted",
                        severity="warning",
                        detail="broker reported an open order missing from the database",
                        order_id=row.order_id,
                        client_order_id=row.client_order_id,
                    )
                )

        for local in local_orders:
            if local.order_id in broker_by_id:
                continue
            try:
                current = self.broker.get_order(local.order_id)
            except OrderNotFound:
                issues.append(
                    ReconciliationIssue(
                        kind="order_missing_at_broker",
                        severity="error",
                        detail="database order is not known to the broker",
                        order_id=local.order_id,
                        client_order_id=local.client_order_id,
                    )
                )
                repositories.orders.update_status(
                    local.order_id,
                    "unknown",
                    raw_status="not_found_at_broker",
                    updated_at=observed_at,
                )
                continue
            row = self._order_row(current, observed_at)
            repositories.orders.upsert(row)
            if row.status in OPEN_ORDER_STATUSES:
                issues.append(
                    ReconciliationIssue(
                        kind="open_order_not_reported",
                        severity="warning",
                        detail=(
                            "order still reports an open status but was absent from "
                            "the broker open-order list"
                        ),
                        order_id=row.order_id,
                        client_order_id=row.client_order_id,
                    )
                )
            elif row.status not in TERMINAL_ORDER_STATUSES:
                issues.append(
                    ReconciliationIssue(
                        kind="unknown_order_status",
                        severity="warning",
                        detail=f"broker returned status {row.status!r}",
                        order_id=row.order_id,
                        client_order_id=row.client_order_id,
                    )
                )
        return issues, adopted

    # ------------------------------------------------------------------
    # Position reconciliation
    # ------------------------------------------------------------------
    def _reconcile_positions(
        self,
        local_positions: Sequence[PositionRow],
        broker_positions: Sequence[Position],
        observed_at: str,
    ) -> tuple[list[ReconciliationIssue], list[str]]:
        repositories = self.database.repositories
        issues: list[ReconciliationIssue] = []
        adopted: list[str] = []
        local_by_symbol = {position.symbol: position for position in local_positions}
        broker_by_symbol: dict[str, Position] = {}

        for position in broker_positions:
            symbol = validate_symbol(position.symbol)
            broker_by_symbol[symbol] = position
            local = local_by_symbol.get(symbol)
            if local is None:
                adopted.append(symbol)
                issues.append(
                    ReconciliationIssue(
                        kind="position_adopted",
                        severity="warning",
                        detail="broker position was not present in persisted state",
                        symbol=symbol,
                    )
                )
            else:
                if not _quantities_equal(local.quantity, position.quantity):
                    issues.append(
                        ReconciliationIssue(
                            kind="position_quantity_mismatch",
                            severity="error",
                            detail=(
                                f"database quantity {local.quantity} does not match "
                                f"broker quantity {position.quantity}"
                            ),
                            symbol=symbol,
                        )
                    )
                if local.side.lower() != str(position.side).lower():
                    issues.append(
                        ReconciliationIssue(
                            kind="position_side_mismatch",
                            severity="error",
                            detail=(
                                f"database side {local.side!r} does not match "
                                f"broker side {position.side!r}"
                            ),
                            symbol=symbol,
                        )
                    )
            # The broker snapshot always wins and is written as a new
            # immutable observation.
            repositories.positions.record(self._position_row(position, observed_at))

        for symbol, local in local_by_symbol.items():
            if symbol not in broker_by_symbol and abs(local.quantity) > _QTY_TOLERANCE:
                issues.append(
                    ReconciliationIssue(
                        kind="position_missing_at_broker",
                        severity="error",
                        detail="database shows an open position but the broker is flat",
                        symbol=symbol,
                    )
                )
        return issues, adopted

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _record_account_snapshot(
        self, account: AccountInfo, observed_at: str, *, trading_blocked: bool
    ) -> None:
        self.database.repositories.account_snapshots.insert(
            AccountSnapshotRow(
                snapshot_id=new_id("acct"),
                observed_at=observed_at,
                account_id=account.account_id,
                status=account.status,
                currency=account.currency,
                cash=account.cash,
                buying_power=account.buying_power,
                equity=account.equity,
                portfolio_value=account.portfolio_value,
                day_trade_count=account.day_trade_count,
                trading_blocked=trading_blocked,
            )
        )

    def _record_event(
        self,
        level: str,
        phase: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.database.repositories.application_events.insert(
            ApplicationEventRow(
                event_id=new_id("app"),
                timestamp=utc_now_timestamp(),
                level=level,
                phase=phase,
                message=message,
                context=context_json(context),
            )
        )

    def _record_risk_event(
        self,
        kind: str,
        code: str,
        message: str,
        observed_at: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.database.repositories.risk_events.insert(
            RiskEventRow(
                event_id=new_id("risk"),
                timestamp=observed_at,
                kind=kind,
                code=code,
                message=message,
                symbol=None,
                client_order_id=None,
                context=context_json(context),
            )
        )

    def _fail(
        self,
        code: str,
        message: str,
        state: PersistedState,
        observed_at: str,
        *,
        account: AccountInfo | None = None,
        issues: tuple[ReconciliationIssue, ...] = (),
    ) -> ReconciliationResult:
        self.trading_gate.block("reconciliation", message)
        try:
            self._record_event(
                "ERROR",
                "reconciliation",
                message,
                {"code": code, "issues": [self._issue_context(i) for i in issues]},
            )
            self._record_risk_event(
                "reconciliation",
                code,
                message,
                observed_at,
                context={"code": code},
            )
            if account is not None:
                self._record_account_snapshot(account, observed_at, trading_blocked=True)
        except Exception:
            # Persistence of the failure record must not mask the original
            # safety decision: the gate is already blocked.
            pass
        result = ReconciliationResult(
            completed=False,
            trading_allowed=False,
            persisted_state=state,
            issues=issues,
            account=account,
            error=message,
        )
        self.last_result = result
        return result

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _order_row(order: BrokerOrder, observed_at: str) -> OrderRow:
        status = str(getattr(order.status, "value", order.status))
        side = "unknown" if order.side is None else str(getattr(order.side, "value", order.side))
        client_order_id = str(order.client_order_id or order.order_id)
        return OrderRow(
            order_id=validate_text(str(order.order_id), "order_id"),
            client_order_id=validate_text(client_order_id, "client_order_id"),
            symbol=validate_symbol(str(order.symbol)),
            side=side,
            order_type=validate_text(str(order.order_type), "order_type").lower(),
            time_in_force=validate_text(str(order.time_in_force), "time_in_force").lower(),
            status=status,
            quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            limit_price=order.limit_price,
            filled_average_price=order.filled_average_price,
            submitted_at=utc_timestamp(order.submitted_at or observed_at, "submitted_at"),
            updated_at=utc_timestamp(order.updated_at or observed_at, "updated_at"),
            source="broker",
            raw_status=order.raw_status or status,
        )

    @staticmethod
    def _position_row(position: Position, observed_at: str) -> PositionRow:
        return PositionRow(
            position_id=new_id("pos"),
            symbol=validate_symbol(position.symbol),
            quantity=position.quantity,
            side=str(position.side or "long").lower(),
            average_entry_price=position.average_entry_price,
            market_value=position.market_value,
            cost_basis=position.cost_basis,
            unrealized_pl=position.unrealized_pl,
            unrealized_pl_fraction=position.unrealized_pl_fraction,
            observed_at=observed_at,
            source="broker",
        )

    @staticmethod
    def _issue_context(issue: ReconciliationIssue) -> dict[str, Any]:
        return {
            "kind": issue.kind,
            "severity": issue.severity,
            "detail": issue.detail,
            "symbol": issue.symbol,
            "order_id": issue.order_id,
            "client_order_id": issue.client_order_id,
        }

    @staticmethod
    def _summarize(issues: Sequence[ReconciliationIssue]) -> str:
        kinds = sorted({issue.kind for issue in issues})
        return f"reconciliation failed: {', '.join(kinds)}"


def _quantities_equal(left: Decimal, right: Decimal) -> bool:
    return abs(left - right) <= _QTY_TOLERANCE
