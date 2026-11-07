"""Unit tests for the broker adapter (Phase 4).

No live Alpaca credentials or network are used; the TradingClient is mocked.
Paper-only enforcement and idempotent/uncertain-timeout handling are covered.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from alpaca.common.exceptions import APIError  # pyright: ignore[reportMissingImports]
from alpaca.trading.enums import (  # pyright: ignore[reportMissingImports]
    OrderSide,
    OrderType,
)

from trading_bot.broker.adapter import (
    OPEN_STATES,
    TERMINAL_STATES,
    AlpacaBrokerAdapter,
)
from trading_bot.config.settings import Settings
from trading_bot.exceptions import (
    BrokerUnavailableError,
    OrderRejectedError,
)


def _api_error(message: str, status_code: int) -> APIError:
    """Build an APIError with a given HTTP status code."""
    from types import SimpleNamespace as _SN
    return APIError(message, http_error=_SN(response=_SN(status_code=status_code)))


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _make_settings(**overrides):
    data = dict(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        alpaca_paper=True,
        alpaca_data_feed="iex",
        trading_mode="paper",
        live_trading_enabled=False,
        retry_attempts=1,
        retry_base_delay=0.01,
        retry_max_delay=1.0,
        retry_backoff_factor=2.0,
        retry_jitter=False,
    )
    data.update(overrides)
    return Settings(**data)


def _account():
    return SimpleNamespace(
        account_id="acc-1",
        status="ACTIVE",
        currency="USD",
        cash=100000.0,
        equity=100000.0,
        portfolio_value=100000.0,
        buying_power=100000.0,
        long_market_value=0.0,
        short_market_value=0.0,
        account_blocked=False,
        trading_blocked=False,
        transfers_blocked=False,
        daytrade_count=0,
        last_equity=100000.0,
    )


def _position(symbol="AAPL", qty=10, avg_entry=100.0, current=105.0):
    return SimpleNamespace(
        symbol=symbol,
        asset_id="asset-1",
        exchange="NASDAQ",
        asset_class="us_equity",
        qty=qty,
        avg_entry_price=avg_entry,
        current_price=current,
        market_value=qty * current,
        cost_basis=qty * avg_entry,
        unrealized_pl=(current - avg_entry) * qty,
        unrealized_plpc=(current - avg_entry) / avg_entry,
        unrealized_intraday_pl=0.0,
        unrealized_intraday_plpc=0.0,
        current_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
        lastday_price=100.0,
        change_today=0.0,
    )


def _order(order_id="order-1", client_order_id="cid-1", symbol="AAPL",
           side="buy", qty=10, filled_qty=10, filled_avg_price=100.0,
           order_type="market", time_in_force="day", status="filled",
           limit_price=None, stop_price=None, submitted_at=None,
           filled_at=None):
    return SimpleNamespace(
        id=order_id,
        client_order_id=client_order_id,
        symbol=symbol,
        asset_id="asset-1",
        side=side,
        qty=qty,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        order_type=order_type,
        time_in_force=time_in_force,
        status=status,
        limit_price=limit_price,
        stop_price=stop_price,
        submitted_at=submitted_at or datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc),
        filled_at=filled_at or datetime(2024, 1, 2, 9, 31, tzinfo=timezone.utc),
        expired_at=None,
        canceled_at=None,
        failed_at=None,
        replaced_at=None,
        replaced_by=None,
        replaces=None,
    )


def _adapter(settings=None):
    settings = settings or _make_settings()
    with patch("trading_bot.broker.adapter.TradingClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        adapter = AlpacaBrokerAdapter(settings)
        adapter._client = mock_client
    return adapter, mock_client


# ---------------------------------------------------------------------------
# Paper-only enforcement
# ---------------------------------------------------------------------------
def test_adapter_rejects_non_paper():
    settings = Settings.model_construct(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        alpaca_paper=False,
        alpaca_data_feed="iex",
        trading_mode="paper",
        live_trading_enabled=False,
    )
    with pytest.raises(ValueError, match="paper trading only"):
        AlpacaBrokerAdapter(settings)


def test_adapter_rejects_live_trading_enabled():
    settings = Settings.model_construct(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        alpaca_paper=True,
        alpaca_data_feed="iex",
        trading_mode="live",
        live_trading_enabled=True,
    )
    with pytest.raises(ValueError, match="paper trading only"):
        AlpacaBrokerAdapter(settings)


def test_adapter_rejects_missing_credentials():
    settings = _make_settings(alpaca_api_key="", alpaca_api_secret="")
    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        AlpacaBrokerAdapter(settings)


def test_adapter_accepts_paper():
    adapter, _ = _adapter()
    assert adapter.settings.is_paper() is True


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------
def test_get_account():
    adapter, client = _adapter()
    client.get_account.return_value = _account()
    result = adapter.get_account()
    assert result is not None
    assert result["equity"] == 100000.0
    assert result["cash"] == 100000.0
    assert result["buying_power"] == 100000.0
    assert result["status"] == "active"
    assert result["currency"] == "USD"


def test_get_account_unauthorized_raises():
    adapter, client = _adapter()
    err = _api_error("unauthorized", 401)
    client.get_account.side_effect = err
    with pytest.raises(OrderRejectedError, match="unauthorized/forbidden"):
        adapter.get_account()


def test_get_account_invalid_request_raises():
    adapter, client = _adapter()
    err = _api_error("bad request", 422)
    client.get_account.side_effect = err
    with pytest.raises(OrderRejectedError, match="invalid request"):
        adapter.get_account()


def test_get_account_unavailable_raises():
    adapter, client = _adapter()
    err = _api_error("server error", 503)
    client.get_account.side_effect = err
    with pytest.raises(BrokerUnavailableError):
        adapter.get_account()


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------
def test_get_positions():
    adapter, client = _adapter()
    client.get_all_positions.return_value = [_position("AAPL"), _position("MSFT", qty=5)]
    result = adapter.get_positions()
    assert len(result) == 2
    assert result[0]["symbol"] == "AAPL"
    assert result[0]["qty"] == 10
    assert result[0]["side"] == "long"
    assert result[1]["symbol"] == "MSFT"
    assert result[1]["qty"] == 5


def test_get_position_found():
    adapter, client = _adapter()
    client.get_open_position.return_value = _position("AAPL")
    result = adapter.get_position("AAPL")
    assert result is not None
    assert result["symbol"] == "AAPL"


def test_get_position_not_found_returns_none():
    adapter, client = _adapter()
    err = _api_error("not found", 404)
    client.get_open_position.side_effect = err
    assert adapter.get_position("ZZZZ") is None


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------
def test_get_orders():
    adapter, client = _adapter()
    client.get_orders.return_value = [_order("o1", status="filled"), _order("o2", status="open")]
    result = adapter.get_orders(status="open")
    assert len(result) == 2
    assert result[0]["order_id"] == "o1"
    assert result[0]["is_filled"] is True
    assert result[1]["is_open"] is True


def test_get_order_found():
    adapter, client = _adapter()
    client.get_order_by_id.return_value = _order("o1")
    result = adapter.get_order("o1")
    assert result is not None
    assert result["order_id"] == "o1"


def test_get_order_not_found_returns_none():
    adapter, client = _adapter()
    err = _api_error("not found", 404)
    client.get_order_by_id.side_effect = err
    assert adapter.get_order("missing") is None


# ---------------------------------------------------------------------------
# Order submission
# ---------------------------------------------------------------------------
def test_submit_market_order():
    adapter, client = _adapter()
    client.submit_order.return_value = _order("o1", client_order_id="cid-1")
    result = adapter.submit_market_order("AAPL", 10, "buy", client_order_id="cid-1")
    assert result is not None
    assert result["order_id"] == "o1"
    assert result["symbol"] == "AAPL"
    assert result["side"] == "buy"
    assert result["is_filled"] is True
    # verify request was built correctly
    req = client.submit_order.call_args[0][0]
    assert req.symbol == "AAPL"
    assert req.qty == 10
    assert req.side == OrderSide.BUY
    assert req.type == OrderType.MARKET
    assert req.client_order_id == "cid-1"


def test_submit_limit_order():
    adapter, client = _adapter()
    client.submit_order.return_value = _order(
        "o1", client_order_id="cid-1", side="sell", order_type="limit", filled_avg_price=100.5
    )
    result = adapter.submit_limit_order("AAPL", 10, "sell", 100.5, client_order_id="cid-1")
    assert result is not None
    assert result["order_type"] == "limit"
    assert result["side"] == "sell"
    req = client.submit_order.call_args[0][0]
    assert req.limit_price == 100.5


def test_submit_order_idempotent_when_exists():
    """A previous attempt may have succeeded; don't double-submit."""
    adapter, client = _adapter()
    existing = _order("o-existing", client_order_id="cid-1", status="filled")
    client.get_orders.return_value = [existing]
    client.submit_order.return_value = _order("o-new")
    result = adapter.submit_market_order("AAPL", 10, "buy", client_order_id="cid-1")
    assert result["order_id"] == "o-existing"
    # submit_order must NOT have been called
    client.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# Uncertain timeout safety (no blind retry)
# ---------------------------------------------------------------------------
def test_submit_order_uncertain_timeout_checks_existing():
    """On timeout, check for an existing order before failing (no blind retry)."""
    adapter, client = _adapter()
    err = _api_error("timeout", 504)
    client.submit_order.side_effect = err
    # An order with this client_order_id already exists -> treat as accepted
    existing = _order("o-existing", client_order_id="cid-1", status="filled")
    client.get_orders.return_value = [existing]
    result = adapter.submit_market_order("AAPL", 10, "buy", client_order_id="cid-1")
    assert result is not None
    assert result["order_id"] == "o-existing"


def test_submit_order_uncertain_timeout_no_existing_raises():
    """On timeout with no existing order, raise BrokerUnavailableError (don't retry blindly)."""
    adapter, client = _adapter()
    err = _api_error("timeout", 504)
    client.submit_order.side_effect = err
    client.get_orders.return_value = []
    with pytest.raises(BrokerUnavailableError):
        adapter.submit_market_order("AAPL", 10, "buy", client_order_id="cid-1")


# ---------------------------------------------------------------------------
# Cancellation / replacement
# ---------------------------------------------------------------------------
def test_cancel_order():
    adapter, client = _adapter()
    assert adapter.cancel_order("o1") is True
    client.cancel_order_by_id.assert_called_once_with("o1")


def test_cancel_order_already_closed_returns_true():
    adapter, client = _adapter()
    err = _api_error("not found", 404)
    client.cancel_order_by_id.side_effect = err
    assert adapter.cancel_order("o1") is True


def test_cancel_all_orders():
    adapter, client = _adapter()
    adapter.cancel_all_orders()
    client.cancel_orders.assert_called_once()


def test_replace_order():
    adapter, client = _adapter()
    client.replace_order_by_id.return_value = _order("o-new", order_type="limit", limit_price=101.0)
    result = adapter.replace_order("o1", qty=20, limit_price=101.0)
    assert result is not None
    assert result["order_id"] == "o-new"
    req = client.replace_order_by_id.call_args[0][1]
    assert req.qty == 20
    assert req.limit_price == 101.0


# ---------------------------------------------------------------------------
# Close position
# ---------------------------------------------------------------------------
def test_close_position():
    adapter, client = _adapter()
    client.close_position.return_value = _order("o1", side="sell", status="filled")
    result = adapter.close_position("AAPL")
    assert result is not None
    assert result["side"] == "sell"
    client.close_position.assert_called_once_with("AAPL")


def test_close_all_positions():
    adapter, client = _adapter()
    client.close_all_positions.return_value = [_order("o1", side="sell")]
    result = adapter.close_all_positions()
    assert len(result) == 1
    client.close_all_positions.assert_called_once_with(cancel_orders=True)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def test_is_connected_true():
    adapter, client = _adapter()
    client.get_account.return_value = _account()
    assert adapter.is_connected() is True


def test_is_connected_false():
    adapter, client = _adapter()
    client.get_account.side_effect = Exception("network down")
    assert adapter.is_connected() is False


# ---------------------------------------------------------------------------
# Order state normalization
# ---------------------------------------------------------------------------
def test_normalize_order_partial_fill():
    adapter, _ = _adapter()
    order = _order("o1", filled_qty=5, filled_avg_price=100.0, status="partially_filled")
    result = adapter._normalize_order(order)
    assert result["filled_qty"] == 5
    assert result["filled_avg_price"] == 100.0
    assert result["is_open"] is True
    assert result["is_terminal"] is False
    assert result["is_filled"] is False


def test_normalize_order_cancelled():
    adapter, _ = _adapter()
    order = _order("o1", status="cancelled")
    result = adapter._normalize_order(order)
    assert result["is_cancelled"] is True
    assert result["is_terminal"] is True


def test_normalize_order_rejected():
    adapter, _ = _adapter()
    order = _order("o1", status="rejected")
    result = adapter._normalize_order(order)
    assert result["is_rejected"] is True
    assert result["is_terminal"] is True


def test_open_states_and_terminal_states():
    assert "filled" in TERMINAL_STATES
    assert "partially_filled" in OPEN_STATES
    assert "open" in OPEN_STATES
    assert "rejected" in TERMINAL_STATES
    assert OPEN_STATES.isdisjoint(TERMINAL_STATES)
