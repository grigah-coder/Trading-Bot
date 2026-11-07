# Broker Integration

`broker.py` contains the provider-neutral broker contract, value models, order
statuses, and structured broker errors. `alpaca_broker.py` implements that
contract with Alpaca's supported `alpaca-py` Trading API.

The adapter always creates `TradingClient(..., paper=True)`. Credentials are
loaded through `Settings.from_env()` from `ALPACA_API_KEY` and
`ALPACA_API_SECRET`; `ALPACA_PAPER=false` is rejected before client creation.
The SDK's HTTP session receives a configured request timeout.

Every order submission and replacement requires a caller-supplied
`client_order_id`. Before submission, the adapter queries Alpaca for that ID.
If a submit times out or otherwise has an ambiguous transport/server failure,
it queries by that ID and returns the broker order if found. If the lookup
cannot establish the outcome, it raises `OrderSubmissionUnknown`; callers must
reconcile with Alpaca before taking further action. The adapter never blindly
resends an ambiguously submitted order. The broker remains authoritative for
order status and fills.

Tests use mocked Alpaca client responses. No real paper API request or order
was sent as part of the automated checks.


New orders in the application engine must use `RiskGatedOrderExecutor`, which
loads fresh account, portfolio, instrument, cost, and market context and sizes
the order through the independent risk engine before submitting it. `TradingEngine`
refuses an order executor without the risk-gate contract.
