# Independent Risk Engine

`RiskEngine` calculates risk-sized order quantities and rejects trades that
exceed account, instrument, portfolio, or market constraints. It consumes
broker-backed account and position state, market quotes with timestamps, and
instrument metadata supplied by an infrastructure `RiskContextSource`.
Sector limits apply only when the instrument provider marks sector data as
reliable. Missing broker account or position reconciliation data fails closed.

## Documented starter limits

These explicit defaults apply when operators do not override the corresponding
`RISK_*` environment variable. They are conservative project starter values,
not a claim that they suit every account or strategy:

- Per-trade estimated loss, including supplied transaction-cost estimates: 0.5%
  of account equity.
- Daily loss: 0.25% of day-start equity; maximum drawdown: 5% from the supplied
  portfolio peak equity.
- Per-symbol exposure: 3% of equity; total portfolio exposure: 25%; single
  order notional: 3%; maximum open symbols: 2.
- Reliable sector exposure: 15% of equity. Unreliable or missing sector labels
  are not guessed or treated as authoritative.
- Average daily dollar volume at least $1,000,000, maximum quoted spread 0.5%,
  maximum daily volatility 8%, and market observations no older than 60 seconds.
- Only the regular market session is allowed by default.

All values are strongly validated at settings load. Use `.env.example` to
review or override them. Every live order path should inject
`RiskGatedOrderExecutor` as the core engine's `OrderExecutor`; it re-loads risk
context and re-approves immediately before calling the broker. Do not inject a
raw broker adapter as the engine's order executor.

Daily-loss breakers remain active until the next trading session. Drawdown
breakers require healthy recovery checks and explicit operator acknowledgement.
Transient data, broker, reconciliation, and execution-state breakers clear only
after all four recovery checks pass.
