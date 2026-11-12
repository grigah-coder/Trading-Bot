# Trading Bot

A reliable, maintainable, risk-controlled algorithmic trading system built for
**Alpaca paper trading only**.

Default configuration: `TRADING_MODE=paper`, `LIVE_TRADING_ENABLED=false`.
Live trading is never activated by this repository.

## Design principles

* **Risk first** — an independent risk engine approves or rejects every trade.
  Risk controls can never be bypassed by strategy code.
* **Paper trading only** — the broker adapter targets Alpaca's paper environment.
  Real-money trades are never executed.
* **Clean architecture** — separated concerns: market data, broker, risk,
  strategy, order management, persistence, backtesting and monitoring.
* **No look-ahead bias** — data validation, freshness checks and timezone
  normalization prevent leakage; backtesting uses realistic execution,
  slippage, spreads and transaction costs.
* **Observable** — structured logging, health checks and non-blocking alerts; no
  API credentials are ever written to logs.

## Project structure

```text
Trading-Bot/
├── .env.example              # Copy to `.env` and fill in your own values
├── .gitignore
├── LICENSE                   # MIT
├── README.md
├── pyproject.toml            # Packaging, lint and coverage configuration
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Dev / test dependencies
├── data/                     # Runtime SQLite database (gitignored)
├── logs/                     # Rotating logs and state files (gitignored)
├── src/trading_bot/
│   ├── __init__.py
│   ├── __main__.py           # Entry point: live loop and --backtest mode
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── cache.py              # Thread-safe TTL cache
│   ├── logging_config.py     # Structured rotating logger
│   ├── retry.py              # Retry / backoff with rate-limit handling
│   ├── trade_ledger.py       # Persistent FIFO trade ledger
│   ├── engine.py             # Central trading engine
│   ├── lifecycle.py          # Start-up / graceful shutdown / signals
│   ├── scheduler.py          # Trading-loop cadence
│   ├── state.py              # Shared application state
│   ├── backtest/             # Backtesting engine and metrics
│   ├── broker/               # Alpaca broker adapter
│   ├── config/               # Validated settings (paper-only)
│   ├── data/                 # Market data, calendar and validation
│   ├── db/                   # SQLite models, migrations and repository
│   ├── monitoring/           # Health checks and alert notifiers
│   ├── order/                # Order lifecycle and order manager
│   ├── risk/                 # Independent risk engine and rules
│   └── strategy/             # Strategy base, indicators and signals
└── tests/
    ├── unit/                 # Unit tests
    └── integration/          # End-to-end pipeline tests
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/grigah-coder/Trading-Bot.git
   cd Trading-Bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit `.env` and set ALPACA_API_KEY / ALPACA_API_SECRET (paper credentials).
   # Never commit `.env`.
   ```

## Running

All commands assume the virtual environment is active:

```bash
source venv/bin/activate
```

Run the paper-trading engine (requires paper credentials in `.env`):

```bash
python -m trading_bot
```

Run a one-off historical backtest through the same entry point:

```bash
python -m trading_bot --backtest
```

The engine runs until interrupted. `SIGINT`/`SIGTERM` trigger a graceful
shutdown: the scheduler stops, open work is reconciled and monitoring is
closed.

## Testing and linting

```bash
# Tests
pytest -q

# Tests with coverage
pytest -q --cov=trading_bot

# Lint
ruff check src tests
```

## Operating mode

The system defaults to paper trading. Set `ALPACA_PAPER=true` and
`LIVE_TRADING_ENABLED=false` in `.env`. Live trading is disabled by the settings
validator and must never be enabled.

All percentage settings are fractions: `0.012` means 1.2%.

## License

MIT — see [LICENSE](LICENSE).
