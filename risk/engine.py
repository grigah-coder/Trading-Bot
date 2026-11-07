"""Independent pre-trade risk assessment and circuit-breaker control."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any

from config.settings import RiskSettings
from risk.exceptions import CircuitBreakerActive, RiskRejected
from risk.models import (
    AccountRiskData,
    CircuitBreakerReason,
    MarketConditions,
    PortfolioRiskContext,
    PortfolioRiskPosition,
    RiskApproval,
    RiskRecoveryStatus,
    TradeOrderType,
    TradeProposal,
    TradeSide,
)


class RiskEngine:
    """Approves a sized order only when broker, portfolio, and market inputs pass."""

    _TRANSIENT_BREAKERS = frozenset({
        CircuitBreakerReason.BROKER_UNAVAILABLE,
        CircuitBreakerReason.POSITION_RECONCILIATION_FAILED,
        CircuitBreakerReason.MARKET_DATA_STALE,
        CircuitBreakerReason.RISK_CALCULATION_FAILURE,
        CircuitBreakerReason.UNEXPECTED_EXECUTION_STATE,
    })

    def __init__(self, settings: RiskSettings) -> None:
        if not isinstance(settings, RiskSettings):
            raise ValueError("RiskEngine requires validated RiskSettings")
        self.settings = settings
        self._breakers: dict[CircuitBreakerReason, str | None] = {}

    @property
    def breaker_active(self) -> bool:
        return bool(self._breakers)

    @property
    def breaker_reasons(self) -> frozenset[CircuitBreakerReason]:
        return frozenset(self._breakers)

    def activate_breaker(self, reason: CircuitBreakerReason | str, *, session_id: str | None = None) -> None:
        normalized = reason if isinstance(reason, CircuitBreakerReason) else CircuitBreakerReason(reason)
        self._breakers[normalized] = session_id

    def recover(
        self,
        status: RiskRecoveryStatus,
        *,
        session_id: str,
        acknowledge_drawdown: bool = False,
    ) -> bool:
        """Clear transient breakers only after all health checks are clean.

        Daily-loss breakers reset only on a new session. Drawdown breakers need
        operator acknowledgement and verified recovery below the configured cap.
        """
        if not status.healthy:
            return False
        cleared = False
        for reason, tripped_session in tuple(self._breakers.items()):
            if reason in self._TRANSIENT_BREAKERS:
                del self._breakers[reason]
                cleared = True
            elif reason is CircuitBreakerReason.DAILY_LOSS_LIMIT and session_id != tripped_session:
                del self._breakers[reason]
                cleared = True
            elif (
                reason is CircuitBreakerReason.MAXIMUM_DRAWDOWN
                and acknowledge_drawdown
                and status.current_drawdown_fraction is not None
                and status.current_drawdown_fraction < _decimal(self.settings.max_drawdown_fraction)
            ):
                del self._breakers[reason]
                cleared = True
        return cleared

    def approve(
        self,
        proposal: TradeProposal,
        context: PortfolioRiskContext,
        *,
        now: datetime | None = None,
    ) -> RiskApproval:
        if self.breaker_active:
            raise self._breaker_error()
        try:
            return self._assess(proposal, context, now or datetime.now(timezone.utc))
        except (RiskRejected, CircuitBreakerActive):
            raise
        except Exception as exc:
            self.activate_breaker(CircuitBreakerReason.RISK_CALCULATION_FAILURE,
                                  session_id=getattr(context, "trading_session_id", None))
            raise CircuitBreakerActive("Risk calculation failed; trading is halted") from exc

    def _assess(
        self, proposal: TradeProposal, context: PortfolioRiskContext, now: datetime
    ) -> RiskApproval:
        if not context.broker_available or context.account is None:
            self._trip(CircuitBreakerReason.BROKER_UNAVAILABLE, context, "Broker account data is unavailable")
        if not context.positions_reconciled:
            self._trip(CircuitBreakerReason.POSITION_RECONCILIATION_FAILED, context,
                       "Broker positions are not reconciled")
        if not context.execution_state_consistent:
            self._trip(CircuitBreakerReason.UNEXPECTED_EXECUTION_STATE, context,
                       "Execution state is inconsistent with broker state")

        account = context.account
        assert account is not None
        try:
            equity = _valid_positive(account.equity, "account equity")
            day_start = _valid_positive(account.day_start_equity, "day-start equity")
            peak = _valid_positive(account.peak_equity, "peak equity")
            buying_power = _valid_nonnegative(account.buying_power, "buying power")
        except Exception as exc:
            self.activate_breaker(CircuitBreakerReason.BROKER_UNAVAILABLE,
                                  session_id=context.trading_session_id)
            raise CircuitBreakerActive("Broker account risk fields are unavailable or invalid") from exc
        daily_loss = (day_start - equity) / day_start
        if daily_loss >= _decimal(self.settings.daily_loss_limit_fraction):
            self._trip(CircuitBreakerReason.DAILY_LOSS_LIMIT, context,
                       "Daily loss threshold has been reached")
        drawdown = (peak - equity) / peak
        if drawdown >= _decimal(self.settings.max_drawdown_fraction):
            self._trip(CircuitBreakerReason.MAXIMUM_DRAWDOWN, context,
                       "Maximum drawdown threshold has been reached")

        market = context.market
        now_utc = _aware_utc(now, "decision time")
        observed = _aware_utc(market.observed_at, "market timestamp")
        age = now_utc - observed
        max_age = _decimal(self.settings.market_data_max_age_seconds)
        if age.total_seconds() < 0 or age.total_seconds() > float(max_age):
            self._trip(CircuitBreakerReason.MARKET_DATA_STALE, context, "Market data is stale")

        symbol = _normalize_symbol(proposal.symbol)
        side = _enum(proposal.side, TradeSide, "side")
        order_type = _enum(proposal.order_type, TradeOrderType, "order_type")
        client_order_id = proposal.client_order_id.strip() if isinstance(proposal.client_order_id, str) else ""
        if not client_order_id:
            self._reject("client_order_id is required", "invalid_order")
        if order_type is TradeOrderType.LIMIT:
            if proposal.limit_price is None:
                self._reject("limit order requires a limit price", "invalid_price")
            entry = _valid_positive(proposal.limit_price, "limit price")
        else:
            if proposal.limit_price is not None:
                self._reject("market order cannot include a limit price", "invalid_order")
            entry = _valid_positive(context.market.ask if side is TradeSide.BUY else context.market.bid,
                                    "market entry price")
        stop = _valid_positive(proposal.stop_price, "stop-loss price")
        if (side is TradeSide.BUY and stop >= entry) or (side is TradeSide.SELL and stop <= entry):
            self._reject("stop-loss must be below a buy entry and above a sell entry", "invalid_stop_loss")

        bid = _valid_positive(market.bid, "bid")
        ask = _valid_positive(market.ask, "ask")
        if ask < bid:
            self._reject("market ask is below bid", "invalid_price")
        if _valid_nonnegative(market.average_daily_dollar_volume, "average dollar volume") < _decimal(self.settings.minimum_average_daily_dollar_volume):
            self._reject("average dollar volume is below the liquidity minimum", "minimum_liquidity")
        midpoint = (bid + ask) / Decimal("2")
        spread_fraction = (ask - bid) / midpoint
        if spread_fraction > _decimal(self.settings.maximum_spread_fraction):
            self._reject("bid-ask spread exceeds the configured maximum", "maximum_spread")
        volatility = _valid_nonnegative(market.daily_volatility_fraction, "daily volatility")
        if volatility > _decimal(self.settings.maximum_daily_volatility_fraction):
            self._reject("daily volatility exceeds the configured maximum", "volatility_restriction")
        if self.settings.regular_session_only and not market.regular_session_open:
            self._reject("trading is restricted to the regular market session", "trading_session")

        instrument = context.instrument
        increment = _valid_positive(instrument.quantity_increment, "quantity increment")
        minimum_quantity = _valid_positive(instrument.minimum_quantity, "minimum quantity")
        maximum_quantity = (
            _valid_positive(instrument.maximum_quantity, "maximum quantity")
            if instrument.maximum_quantity is not None else None
        )
        minimum_notional = _valid_nonnegative(instrument.minimum_notional, "minimum notional")
        costs_per_unit = _valid_nonnegative(context.costs.per_unit, "per-unit transaction cost")
        fixed_cost = _valid_nonnegative(context.costs.fixed, "fixed transaction cost")
        risk_budget = equity * _decimal(self.settings.max_risk_per_trade_fraction)
        risk_per_unit = abs(entry - stop) + costs_per_unit
        if fixed_cost >= risk_budget or risk_per_unit <= 0:
            self._reject("transaction costs or stop distance leave no risk budget", "risk_budget")
        sized_quantity = _floor_to_increment((risk_budget - fixed_cost) / risk_per_unit, increment)
        limiting_reason = "maximum risk per trade"
        quantity_limit = "maximum_risk_per_trade"

        def apply_cap(cap: Decimal, reason: str, code: str) -> None:
            nonlocal sized_quantity, limiting_reason, quantity_limit
            capped = _floor_to_increment(cap, increment)
            if capped < sized_quantity:
                sized_quantity = capped
                limiting_reason = reason
                quantity_limit = code

        if maximum_quantity is not None:
            apply_cap(maximum_quantity, "instrument maximum quantity", "instrument_constraint")
        order_cap = equity * _decimal(self.settings.max_order_notional_fraction) / entry
        apply_cap(order_cap, "maximum order notional", "maximum_order_notional")

        all_exposures = (*context.positions, *context.open_orders)
        current_symbol_exposure = sum(
            (_valid_nonnegative(abs(position.market_value), "position exposure")
             for position in all_exposures if position.symbol.strip().upper() == symbol),
            Decimal("0"),
        )
        position_cap = (equity * _decimal(self.settings.max_position_fraction) - current_symbol_exposure) / entry
        apply_cap(max(Decimal("0"), position_cap), "maximum position exposure", "maximum_position_exposure")
        total_exposure = sum(
            (_valid_nonnegative(abs(position.market_value), "portfolio exposure")
             for position in all_exposures),
            Decimal("0"),
        )
        portfolio_cap = (equity * _decimal(self.settings.max_portfolio_exposure_fraction) - total_exposure) / entry
        apply_cap(max(Decimal("0"), portfolio_cap), "maximum portfolio exposure", "maximum_portfolio_exposure")
        apply_cap(buying_power / entry, "available buying power", "insufficient_buying_power")

        current_symbols = {position.symbol.strip().upper() for position in all_exposures
                           if _valid_nonnegative(abs(position.market_value), "position exposure") > 0}
        if symbol not in current_symbols and len(current_symbols) >= self.settings.max_open_positions:
            self._reject("maximum open positions reached", "maximum_open_positions")

        if (instrument.sector_classification_reliable and instrument.sector
                and instrument.sector.strip()):
            sector = instrument.sector.strip().casefold()
            current_sector_exposure = sum((
                abs(_valid_nonnegative(abs(position.market_value), "sector exposure")) for position in all_exposures
                if position.sector_classification_reliable and position.sector
                and position.sector.strip().casefold() == sector
            ), Decimal("0"))
            sector_cap = (equity * _decimal(self.settings.max_sector_exposure_fraction)
                          - current_sector_exposure) / entry
            apply_cap(max(Decimal("0"), sector_cap), "maximum sector exposure", "maximum_sector_exposure")

        requested = proposal.requested_quantity
        if requested is not None:
            requested = _valid_positive(requested, "requested quantity")
            if requested % increment != 0:
                self._reject("requested quantity violates instrument increment", "instrument_constraint")
            if requested > sized_quantity:
                self._reject(f"requested size exceeds {limiting_reason}", quantity_limit)
            quantity = requested
        else:
            quantity = sized_quantity
        if quantity < minimum_quantity or quantity * entry < minimum_notional:
            self._reject("risk-sized quantity is below instrument minimums", "instrument_minimum")

        notional = quantity * entry
        transaction_cost = quantity * costs_per_unit + fixed_cost
        estimated_risk = quantity * risk_per_unit + fixed_cost
        if estimated_risk > risk_budget:
            self._reject("estimated loss exceeds maximum risk per trade", "maximum_risk_per_trade")
        return RiskApproval(
            symbol=symbol, side=side, quantity=quantity, entry_price=entry,
            stop_price=stop, order_notional=notional, estimated_risk=estimated_risk,
            estimated_transaction_cost=transaction_cost, client_order_id=client_order_id,
            order_type=order_type, limit_price=proposal.limit_price,
            time_in_force=str(getattr(proposal.time_in_force, "value", proposal.time_in_force)),
        )

    def _trip(self, reason: CircuitBreakerReason, context: PortfolioRiskContext, message: str) -> None:
        self.activate_breaker(reason, session_id=context.trading_session_id)
        raise self._breaker_error(message)

    def _breaker_error(self, message: str = "Trading is halted by a risk circuit breaker") -> CircuitBreakerActive:
        return CircuitBreakerActive(message, context={"breakers": tuple(sorted(reason.value for reason in self._breakers))})

    @staticmethod
    def _reject(message: str, reason: str) -> None:
        raise RiskRejected(message, context={"reason": reason})


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RiskRejected("Risk input must be numeric and finite",
                           context={"reason": "invalid_risk_input"}) from exc
    if not result.is_finite():
        raise RiskRejected("Risk input must be numeric and finite",
                           context={"reason": "invalid_risk_input"})
    return result


def _valid_positive(value: Any, name: str) -> Decimal:
    result = _decimal(value)
    if result <= 0:
        raise RiskRejected(f"{name} must be positive", context={"field": name})
    return result


def _valid_nonnegative(value: Any, name: str) -> Decimal:
    result = _decimal(value)
    if result < 0:
        raise RiskRejected(f"{name} cannot be negative", context={"field": name})
    return result


def _floor_to_increment(value: Decimal, increment: Decimal) -> Decimal:
    return (value / increment).to_integral_value(rounding=ROUND_DOWN) * increment


def _enum(value: Any, enum_type: type, name: str):
    try:
        return value if isinstance(value, enum_type) else enum_type(str(value).lower())
    except ValueError as exc:
        raise RiskRejected(f"Unsupported {name}", context={"field": name}) from exc


def _normalize_symbol(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RiskRejected("Instrument symbol is required", context={"reason": "invalid_instrument"})
    return value.strip().upper()


def _aware_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RiskRejected(f"{name} must be timezone-aware", context={"field": name})
    return value.astimezone(timezone.utc)
