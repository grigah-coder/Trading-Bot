# Strategy framework

Strategies implement `Strategy.evaluate(StrategySnapshot) -> Signal | None`.
Each snapshot contains one symbol's validated candles, timeframe, and aware
decision timestamp. Historical data should come through
`HistoricalDataService.get_history` so bars unavailable at decision time are
excluded; strategies also check candle quality, gaps, age, and bar completion.

`TrendMomentumStrategy` is a configurable, long-only research example. It enters
when fast EMA exceeds slow EMA, the latest close exceeds slow EMA, momentum is
positive, and ATR as a fraction of price is inside the configured volatility
regime. The stop and target are based on ATR multiples. It exits an existing
strategy-tracked position when fast EMA is at or below slow EMA or momentum is
non-positive. It does not emit a short signal. Parameters can be supplied via
`STRATEGY_PARAM_<NAME>` environment values; `TrendMomentumConfig.from_mapping`
converts and validates them.

Signal strength is left unset because this rule has not been calibrated. This
strategy makes no profitability claim and is not validated for live trading.
FinBERT is excluded from strategy decisions until independent research shows
incremental predictive value. Any evaluation must reserve data not used for
parameter tuning as the final performance set.

The strategy's in-memory position intent is research scaffolding, not broker
position truth. Production wiring must reconcile actual positions and must
convert signals into risk-approved order intents through the independent risk
gate. This module does not submit orders.
