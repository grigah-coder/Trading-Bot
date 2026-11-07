# Trading Bot

A modular Python foundation for an Alpaca paper-trading bot. It includes
validated configuration, lifecycle state, a dependency-injected engine,
failure recovery, and an XNYS-aware scheduler. Strategy evaluation, risk checks,
and order submission are interfaces supplied to the engine. A paper-only Alpaca
broker adapter is available under `execution/`; strategy logic and automatic
engine-to-broker wiring are not part of this foundation.

## Requirements

- Python 3.11 or newer
- Alpaca paper-trading credentials supplied through the environment when using
  the paper broker adapter

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set values in `.env`, then export them into the process environment using your
shell or a local environment manager. The application reads environment
variables and does not load `.env` files itself.

Select the research strategy with `STRATEGY_NAME=trend_momentum` and set its
validated parameters with `STRATEGY_PARAM_<NAME>` variables, as shown in
`.env.example`. Application assembly should create `TrendMomentumConfig` from
`settings.strategy.parameters`, then inject the strategy and
`SignalValidator` into the core engine.

`ALPACA_PAPER` defaults to `true`. Keep it enabled for the intended paper
environment. Never commit `.env` or paste credentials into source files.

## Project layout

| Directory | Responsibility |
| --- | --- |
| `config/` | Typed, environment-based settings |
| `core/` | Shared infrastructure such as retry behavior |
| `data/` | Data access and in-memory caching |
| `strategies/` | Strategy interface and configurable trend/momentum research example |
| `indicators/` | Pure moving average, momentum, and ATR calculations |
| `signals/` | Typed signal records and expiration/session/duplicate validation |
| `risk/` | Independent portfolio and pre-trade controls |
| `execution/` | Broker-neutral contract and Alpaca paper adapter |
| `portfolio/` | Portfolio state and protective order tracking |
| `storage/` | Persistent trade and portfolio records |
| `backtesting/` | Reserved for future simulation tools |
| `monitoring/` | Reserved for future health and metrics reporting |
| `tests/` | Automated checks |
| `scripts/` | Developer and maintenance commands |

The engine and application lifecycle live in `core/`. The scheduler uses the
XNYS exchange calendar for holidays, daylight-saving changes, regular sessions,
and early closes. Components are injected when the application starts, and
startup health checks run before entering `RUNNING`. The engine rejects an
order executor unless it declares and implements the independent risk gate.

The initial strategy is a research example only and makes no profitability
claim. Strategy parameters are configurable through `STRATEGY_PARAM_*` values.
FinBERT is not used in strategy decisions pending independent evidence of
incremental predictive value. No strategy in this repository submits orders;
production integration must pass every proposed order through the independent
risk-gated execution path.

Configuration is split into broker, trading, strategy, risk, database, and
logging settings. Invalid values fail during startup. `ALPACA_PAPER` must remain
true in this phase.

## Development

```bash
python -m pytest
```

Runtime dependencies are `exchange-calendars`, `alpaca-py`, and `requests`;
pytest is the only development dependency. The Alpaca adapter always constructs
`TradingClient` with paper mode enabled and rejects `ALPACA_PAPER=false`.
