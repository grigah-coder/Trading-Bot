"""Unit tests for the Phase 5 independent risk engine."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from trading_bot.config.settings import Settings
from trading_bot.exceptions import (
    RiskDataUnavailableError,
)
from trading_bot.risk import rules
from trading_bot.risk.engine import RiskEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeState:
    """Minimal stand-in for AppState capturing circuit breaker calls."""

    def __init__(self) -> None:
        self.tripped = False
        self.reset_called = False

    def trip_circuit_breaker(self, when=None) -> None:  # noqa: ARG002
        self.tripped = True

    def reset_circuit_breaker(self) -> None:
        self.reset_called = True


def _settings(**overrides: Any) -> Settings:
    # Keep unit tests hermetic: no shared on-disk risk state unless a test
    # explicitly opts in by passing risk_state_file.
    overrides.setdefault("risk_state_file", "")
    return Settings(**overrides)


def _account(equity: float = 10000.0, cash: float = 10000.0,
             blocked: bool = False, status: str = "active") -> Dict[str, Any]:
    return {
        "account_id": "acct-1",
        "status": status,
        "currency": "USD",
        "cash": cash,
        "equity": equity,
        "portfolio_value": equity,
        "buying_power": equity,
        "long_market_value": 0.0,
        "short_market_value": 0.0,
        "account_blocked": blocked,
        "trading_blocked": blocked,
        "transfers_blocked": False,
        "day_trade_count": 0,
        "last_equity": equity,
        "day_pl": 0.0,
        "day_pl_pct": 0.0,
    }


def _position(symbol: str = "AAPL", qty: float = 0.0) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "asset_id": "",
        "exchange": "",
        "asset_class": "us_equity",
        "qty": qty,
        "avg_entry_price": 0.0,
        "current_price": 0.0,
        "market_value": 0.0,
        "cost_basis": 0.0,
        "unrealized_pl": 0.0,
        "unrealized_plpc": 0.0,
        "unrealized_intraday_pl": 0.0,
        "unrealized_intraday_plpc": 0.0,
        "current_date": "2024-01-01",
        "lastday_price": 0.0,
        "change_today": 0.0,
        "side": "long" if qty > 0 else "flat",
    }


def _signal(symbol: str = "AAPL", side: str = "buy", price: float = 100.0,
            qty: float = 10.0, stop_pct: float = 0.02,
            take_pct: float = 0.04) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "price": price,
        "stop_loss_pct": stop_pct,
        "take_profit_pct": take_pct,
        "signal_id": "sig-1",
    }


# ---------------------------------------------------------------------------
# Rules tests
# ---------------------------------------------------------------------------

class TestAccountTradable:
    def test_active_account_passes(self):
        ok, reason, _ = rules.check_account_tradable(_account())
        assert ok is True
        assert reason == "account tradable"

    def test_blocked_account_rejected(self):
        ok, reason, _ = rules.check_account_tradable(_account(blocked=True))
        assert ok is False
        assert "blocked" in reason

    def test_missing_account_rejected(self):
        ok, reason, _ = rules.check_account_tradable({})
        assert ok is False
        assert "unavailable" in reason

    def test_non_tradable_status_rejected(self):
        ok, reason, _ = rules.check_account_tradable(_account(status="suspended"))
        assert ok is False
        assert "not tradable" in reason


class TestConcurrentPositions:
    def test_under_limit_passes(self):
        positions = [_position("AAPL", 10)]
        ok, reason, _ = rules.check_concurrent_positions(
            positions, _settings(max_concurrent_positions=2),
        )
        assert ok is True

    def test_at_limit_rejected(self):
        positions = [_position("AAPL", 10), _position("MSFT", 5)]
        ok, reason, _ = rules.check_concurrent_positions(
            positions, _settings(max_concurrent_positions=2),
        )
        assert ok is False
        assert "max concurrent" in reason


class TestSymbolNotAlreadyHeld:
    def test_not_held_passes(self):
        ok, _, _ = rules.check_symbol_not_already_held(
            _signal(symbol="AAPL"), [_position("MSFT", 5)],
        )
        assert ok is True

    def test_already_held_rejected(self):
        ok, reason, _ = rules.check_symbol_not_already_held(
            _signal(symbol="AAPL"), [_position("AAPL", 10)],
        )
        assert ok is False
        assert "already holding" in reason


class TestPositionSize:
    def test_sizes_by_pct_of_equity(self):
        # equity 10000 * 0.03 = 300 notional / price 100 = 3 shares
        qty, reason = rules.compute_position_size(
            _signal(price=100.0), _account(equity=10000.0),
            _settings(position_size_pct=0.03),
        )
        assert qty == pytest.approx(3.0)

    def test_capped_by_notional(self):
        settings = _settings(position_size_pct=0.5, max_position_notional=50.0)
        qty, reason = rules.compute_position_size(
            _signal(price=100.0), _account(equity=10000.0), settings,
        )
        assert qty == pytest.approx(0.5)  # 50 / 100

    def test_capped_by_max_shares(self):
        settings = _settings(max_shares_per_trade=2)
        qty, reason = rules.compute_position_size(
            _signal(price=100.0), _account(equity=10000.0), settings,
        )
        assert qty == pytest.approx(2.0)

    def test_no_price_returns_none(self):
        qty, reason = rules.compute_position_size(
            {"symbol": "AAPL", "side": "buy", "quantity": 10}, _account(),
            _settings(),
        )
        assert qty is None
        assert "price" in reason

    def test_no_equity_returns_none(self):
        qty, reason = rules.compute_position_size(
            _signal(price=100.0), _account(equity=0.0, cash=0.0),
            _settings(),
        )
        assert qty is None


class TestPortfolioRisk:
    def test_within_limit_passes(self):
        ok, reason, _ = rules.check_portfolio_risk(
            _signal(price=100.0, qty=10, stop_pct=0.02),
            _account(equity=10000.0), 10.0, _settings(max_portfolio_risk_pct=0.01),
        )
        # risk = 100*0.02*10 = 20; max = 10000*0.01 = 100 -> pass
        assert ok is True

    def test_exceeds_limit_rejected(self):
        ok, reason, _ = rules.check_portfolio_risk(
            _signal(price=100.0, qty=1000, stop_pct=0.02),
            _account(equity=10000.0), 1000.0,
            _settings(max_portfolio_risk_pct=0.01),
        )
        # risk = 100*0.02*1000 = 2000; max = 100 -> reject
        assert ok is False
        assert "exceeds" in reason


class TestStopLossBand:
    def test_within_band_passes(self):
        ok, _, _ = rules.check_stop_loss_band(
            _signal(stop_pct=0.02), _settings(stop_loss_min_pct=0.01,
                                              stop_loss_max_pct=0.05),
        )
        assert ok is True

    def test_below_min_rejected(self):
        ok, reason, _ = rules.check_stop_loss_band(
            _signal(stop_pct=0.005), _settings(stop_loss_min_pct=0.01,
                                               stop_loss_max_pct=0.05),
        )
        assert ok is False
        assert "below minimum" in reason

    def test_above_max_rejected(self):
        ok, reason, _ = rules.check_stop_loss_band(
            _signal(stop_pct=0.2), _settings(stop_loss_min_pct=0.01,
                                             stop_loss_max_pct=0.05),
        )
        assert ok is False
        assert "above maximum" in reason


class TestRewardRisk:
    def test_acceptable_ratio_passes(self):
        # stop 2%, take 4% -> reward/risk = 0.04/0.02 = 2.0 >= 1.0
        ok, _, _ = rules.check_reward_risk(
            _signal(stop_pct=0.02, take_pct=0.04),
            _settings(min_reward_risk_ratio=1.0),
        )
        assert ok is True

    def test_below_minimum_rejected(self):
        # stop 2%, take 1% -> ratio 0.5 < 2.0 -> reject
        ok, reason, _ = rules.check_reward_risk(
            _signal(stop_pct=0.02, take_pct=0.01),
            _settings(min_reward_risk_ratio=2.0),
        )
        assert ok is False
        assert "below minimum" in reason


class TestDailyLossLimit:
    def test_disabled_passes(self):
        ok, _, _ = rules.check_daily_loss_limit(
            9999.0, _settings(daily_loss_limit_enabled=False),
        )
        assert ok is True

    def test_under_limit_passes(self):
        ok, _, _ = rules.check_daily_loss_limit(
            100.0, _settings(daily_loss_limit_usd=250.0),
        )
        assert ok is True

    def test_at_limit_rejected(self):
        ok, reason, _ = rules.check_daily_loss_limit(
            250.0, _settings(daily_loss_limit_usd=250.0),
        )
        assert ok is False
        assert "reached" in reason

    def test_pct_limit_applies(self):
        # equity 10000 * 0.0025 = 25; loss 30 > 25 -> reject
        ok, _, _ = rules.check_daily_loss_limit(
            30.0, _settings(daily_loss_limit_usd=999999.0,
                            daily_loss_limit_pct=0.0025), equity=10000.0,
        )
        assert ok is False


# ---------------------------------------------------------------------------
# RiskEngine tests
# ---------------------------------------------------------------------------

class TestRiskEngineHappyPath:
    def test_approves_valid_signal(self):
        engine = RiskEngine(_settings())
        decision = engine.assess(
            _signal(price=100.0, qty=10, stop_pct=0.012, take_pct=0.04),
            _account(equity=10000.0),
            [],
        )
        assert decision["approved"] is True
        assert decision["order"] is not None
        assert decision["qty"] == pytest.approx(3.0)  # pct-capped
        assert decision["stop_price"] == pytest.approx(98.8)
        assert decision["limit_price"] == pytest.approx(104.0)
        assert all(r["passed"] for r in decision["rules"])

    def test_rejects_blocked_account(self):
        engine = RiskEngine(_settings())
        decision = engine.assess(
            _signal(), _account(blocked=True), [],
        )
        assert decision["approved"] is False
        assert decision["rejection_reason"] is not None

    def test_rejects_position_limit(self):
        settings = _settings(max_concurrent_positions=1)
        engine = RiskEngine(settings)
        decision = engine.assess(
            _signal(symbol="MSFT"), _account(),
            [_position("AAPL", 10)],
        )
        assert decision["approved"] is False

    def test_rejects_already_held(self):
        engine = RiskEngine(_settings())
        decision = engine.assess(
            _signal(symbol="AAPL"), _account(),
            [_position("AAPL", 10)],
        )
        assert decision["approved"] is False
        assert "already holding" in decision["rejection_reason"]

    def test_rejects_no_price(self):
        engine = RiskEngine(_settings())
        decision = engine.assess(
            {"symbol": "AAPL", "side": "buy", "quantity": 10},
            _account(), [],
        )
        assert decision["approved"] is False

    def test_raises_on_missing_account(self):
        engine = RiskEngine(_settings())
        with pytest.raises(RiskDataUnavailableError):
            engine.assess(_signal(), {}, [])


class TestCircuitBreaker:
    def test_is_halted_after_trip(self):
        state = _FakeState()
        engine = RiskEngine(_settings(), state=state)
        engine.trip_circuit_breaker("manual halt")
        assert engine.is_halted() is True
        assert state.tripped is True

    def test_assess_rejects_when_halted(self):
        state = _FakeState()
        engine = RiskEngine(_settings(), state=state)
        engine.trip_circuit_breaker("halted")
        decision = engine.assess(_signal(), _account(), [])
        assert decision["approved"] is False
        assert decision["rejection_reason"] == "halted"

    def test_reset_clears_halt(self):
        state = _FakeState()
        engine = RiskEngine(_settings(), state=state)
        engine.trip_circuit_breaker("halted")
        engine.reset_circuit_breaker()
        assert engine.is_halted() is False
        assert state.reset_called is True

    def test_daily_loss_trips_breaker(self):
        state = _FakeState()
        engine = RiskEngine(
            _settings(daily_loss_limit_usd=100.0), state=state,
        )
        engine.record_loss(150.0)
        assert engine.is_halted() is True
        assert state.tripped is True

    def test_reset_daily_clears_loss_tracker(self):
        engine = RiskEngine(_settings())
        engine.record_loss(50.0)
        engine.reset_daily()
        # fresh engine state; record_loss added 50 but reset cleared it
        assert engine._daily_loss_usd == 0.0


class TestRiskStatePersistence:
    """Daily loss and breaker state survive restarts via risk_state_file."""

    def test_loss_is_persisted_and_reloaded(self, tmp_path):
        """Daily loss survives a restart on the same trading day."""
        path = str(tmp_path / "risk_state.json")
        first = RiskEngine(_settings(risk_state_file=path))
        first.record_loss(50.0)
        assert first._daily_loss_usd == 50.0

        second = RiskEngine(_settings(risk_state_file=path))
        assert second._daily_loss_usd == 50.0
        assert second.is_halted() is False

    def test_halt_is_persisted_and_reloaded(self, tmp_path):
        """A daily-loss halt is restored from disk after a restart."""
        path = str(tmp_path / "risk_state.json")
        first = RiskEngine(
            _settings(risk_state_file=path, daily_loss_limit_usd=100.0),
        )
        first.record_loss(150.0)
        assert first.is_halted() is True

        second = RiskEngine(
            _settings(risk_state_file=path, daily_loss_limit_usd=100.0),
        )
        assert second.is_halted() is True
        assert "daily loss" in second._halt_reason

    def test_new_trading_day_resets_daily_loss_and_halt(self, tmp_path):
        """A stale date clears the loss budget and its halt."""
        import json

        path = tmp_path / "risk_state.json"
        path.write_text(json.dumps({
            "version": 1,
            "date": "2000-01-01",
            "daily_loss_usd": 200.0,
            "halted": True,
            "reason": "daily loss limit reached ($200.00)",
        }))

        engine = RiskEngine(_settings(risk_state_file=str(path)))
        assert engine._daily_loss_usd == 0.0
        assert engine.is_halted() is False

    def test_same_day_state_is_preserved(self, tmp_path):
        """Same-day persisted loss and manual halt are preserved."""
        import json

        path = tmp_path / "risk_state.json"
        engine = RiskEngine(_settings(risk_state_file=str(path)))
        today = engine._market_today()
        path.write_text(json.dumps({
            "version": 1,
            "date": today,
            "daily_loss_usd": 120.0,
            "halted": True,
            "reason": "manual halt",
        }))

        reloaded = RiskEngine(_settings(risk_state_file=str(path)))
        assert reloaded._daily_loss_usd == 120.0
        assert reloaded.is_halted() is True
