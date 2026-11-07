"""Mandatory execution path that independently obtains risk and approves orders."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol

from execution.broker import (
    Broker,
    BrokerConnectionError,
    BrokerOperationUnknown,
    BrokerOrder,
    BrokerRejectedOrder,
    BrokerRequestTimeout,
    OrderSide,
    OrderStatus,
    OrderSubmissionUnknown,
    TimeInForce,
)
from risk.engine import RiskEngine
from risk.exceptions import CircuitBreakerActive
from risk.models import PortfolioRiskContext, TradeProposal


class RiskContextSource(Protocol):
    """Infrastructure source for broker, position, instrument, and market state."""

    def load(self, proposal: TradeProposal) -> PortfolioRiskContext: ...

    def health_check(self) -> None: ...

    def shutdown(self) -> None: ...


class RiskGatedOrderExecutor:
    """Core engine executor; it always re-assesses risk immediately before submit."""

    risk_gate_enforced = True

    def __init__(
        self,
        risk_engine: RiskEngine,
        broker: Broker,
        context_source: RiskContextSource,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._risk = risk_engine
        self._broker = broker
        self._context_source = context_source
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def execute(self, proposal: TradeProposal) -> BrokerOrder:
        try:
            context = self._context_source.load(proposal)
        except Exception as exc:
            self._risk.activate_breaker("broker_unavailable")
            raise CircuitBreakerActive("Independent risk context could not be loaded") from exc
        approval = self._risk.approve(proposal, context, now=self._clock())
        try:
            side = OrderSide(approval.side.value)
            time_in_force = TimeInForce(approval.time_in_force.lower())
            if approval.order_type.value == "limit":
                if approval.limit_price is None:
                    raise CircuitBreakerActive("Risk-approved limit order has no limit price")
                result = self._broker.submit_limit_order(
                    approval.symbol, approval.quantity, side, approval.limit_price,
                    client_order_id=approval.client_order_id, time_in_force=time_in_force,
                )
            else:
                result = self._broker.submit_market_order(
                    approval.symbol, approval.quantity, side,
                    client_order_id=approval.client_order_id, time_in_force=time_in_force,
                )
        except BrokerRejectedOrder:
            raise
        except (OrderSubmissionUnknown, BrokerOperationUnknown) as exc:
            self._risk.activate_breaker("unexpected_execution_state")
            raise CircuitBreakerActive("Broker execution outcome is unresolved") from exc
        except (BrokerConnectionError, BrokerRequestTimeout) as exc:
            self._risk.activate_breaker("broker_unavailable")
            raise CircuitBreakerActive("Broker became unavailable during order submission") from exc
        if result.status is OrderStatus.UNKNOWN:
            self._risk.activate_breaker("unexpected_execution_state")
            raise CircuitBreakerActive(
                "Broker returned an unknown order status",
                context={"order_id": result.order_id, "raw_status": result.raw_status},
            )
        if result.status is OrderStatus.REJECTED:
            raise BrokerRejectedOrder(
                "Broker rejected the approved order",
                context={"order_id": result.order_id, "raw_status": result.raw_status},
            )
        return result

    def health_check(self) -> None:
        self._broker.get_account()
        self._context_source.health_check()

    def shutdown(self) -> None:
        failures: list[Exception] = []
        for component in (self._context_source, self._broker):
            close = getattr(component, "shutdown", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise RuntimeError("Risk-gated execution shutdown failed") from failures[0]
