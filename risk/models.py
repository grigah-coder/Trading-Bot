"""Typed inputs and outputs for independent trade-risk decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradeOrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True, slots=True)
class TradeProposal:
    symbol: str
    side: TradeSide | str
    stop_price: Decimal
    client_order_id: str
    requested_quantity: Decimal | None = None
    order_type: TradeOrderType | str = TradeOrderType.MARKET
    limit_price: Decimal | None = None
    time_in_force: str = "day"


@dataclass(frozen=True, slots=True)
class AccountRiskData:
    equity: Decimal
    day_start_equity: Decimal
    peak_equity: Decimal
    buying_power: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioRiskPosition:
    symbol: str
    market_value: Decimal
    sector: str | None = None
    sector_classification_reliable: bool = False


@dataclass(frozen=True, slots=True)
class MarketConditions:
    bid: Decimal
    ask: Decimal
    average_daily_dollar_volume: Decimal
    daily_volatility_fraction: Decimal
    observed_at: datetime
    regular_session_open: bool


@dataclass(frozen=True, slots=True)
class InstrumentConstraints:
    quantity_increment: Decimal
    minimum_quantity: Decimal
    maximum_quantity: Decimal | None
    minimum_notional: Decimal
    sector: str | None
    sector_classification_reliable: bool


@dataclass(frozen=True, slots=True)
class EstimatedTransactionCosts:
    per_unit: Decimal
    fixed: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioRiskContext:
    account: AccountRiskData | None
    positions: tuple[PortfolioRiskPosition, ...]
    positions_reconciled: bool
    execution_state_consistent: bool
    broker_available: bool
    market: MarketConditions
    instrument: InstrumentConstraints
    costs: EstimatedTransactionCosts
    trading_session_id: str
    open_orders: tuple[PortfolioRiskPosition, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RiskApproval:
    symbol: str
    side: TradeSide
    quantity: Decimal
    entry_price: Decimal
    stop_price: Decimal
    order_notional: Decimal
    estimated_risk: Decimal
    estimated_transaction_cost: Decimal
    client_order_id: str
    order_type: TradeOrderType
    limit_price: Decimal | None
    time_in_force: str


@dataclass(frozen=True, slots=True)
class RiskRecoveryStatus:
    broker_available: bool
    positions_reconciled: bool
    market_data_fresh: bool
    execution_state_consistent: bool
    current_drawdown_fraction: Decimal | None = None
    risk_calculations_valid: bool = True

    @property
    def healthy(self) -> bool:
        return all((self.broker_available, self.positions_reconciled,
                    self.market_data_fresh, self.execution_state_consistent,
                    self.risk_calculations_valid))


class CircuitBreakerReason(str, Enum):
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    BROKER_UNAVAILABLE = "broker_unavailable"
    POSITION_RECONCILIATION_FAILED = "position_reconciliation_failed"
    MARKET_DATA_STALE = "market_data_stale"
    RISK_CALCULATION_FAILURE = "risk_calculation_failure"
    UNEXPECTED_EXECUTION_STATE = "unexpected_execution_state"
