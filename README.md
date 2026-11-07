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
* **Observable** — structured logging, health checks and alerts; no API
  credentials are ever written to logs.

## Project structure

```text
Trading-Bot/
├── .env.example              # Copy to `.env` and fill in your own values
├── .gitignore
├── pyproject.toml            # Packaging and lint configuration
├── requirements.txt          # Runtime dependencies (minimal)
├── requirements-dev.txt      # Dev / test dependencies
├── README.md
├── Dockerfile                # (optional) containerized paper-trading run
├── config/                   # Configuration schemas
├── data/                     # Historical / runtime market data (gitignored)
├── logs/                     # Rotating logs (gitignored)
├── scripts/                  # Helper scripts
├── src/trading_bot/
│   ├── __init__.py
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── cache.py              # Thread-safe TTL cache
│   ├── logging_config.py     # Structured rotating logger
│   ├── order_state.py        # Thread-safe bracket-leg tracker
│   ├── retry.py              # Retry / backoff with rate-limit handling
│   ├── trade_ledger.py       # Persistent FIFO trade ledger
│   ├── engine.py             # Central trading engine (Phase 2)
│   ├── lifecycle.py          # Application lifecycle (Phase 2)
│   ├── scheduler.py          # Scheduler (Phase 2)
│   ├── state.py              # State management (Phase 2)
│   ├── data/                 # Market data engine (Phase 3)
│   ├── broker/               # Alpaca broker adapter (Phase 4)
│   ├── risk/                 # Independent risk engine (Phase 5)
│   ├── strategy/             # Strategy & signal engine (Phase 6)
│   ├── order/                # Order lifecycle (Phase 7)
│   ├── db/                   # Persistence (Phase 8)
│   ├── backtest/             # Backtesting engine (Phase 9)
│   └── monitoring/           # Health checks & alerts (Phase 11)
└── tests/
    ├── unit/                 # Unit tests
    └── integration/          # Integration tests
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

5. Run the test suite:
   ```bash
   pytest -q
   ```

## Operating mode

The system defaults to paper trading. Set `ALPACA_PAPER=true` and
`LIVE_TRADING_ENABLED=false` in `.env`. Live trading is disabled and must never
be enabled.

## Development

The project is delivered in 12 phases. Phase 1 establishes the repository
structure, configuration and cleanup. Subsequent phases implement the trading
engine, market data, broker integration, risk engine, strategy, order
management, persistence, backtesting, testing, monitoring and the final
paper-trading release.

Run lint:
```bash
ruff check src tests
```

Run tests:
```bash
pytest -q
```
