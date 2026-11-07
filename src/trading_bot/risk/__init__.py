"""Independent risk engine package.

Provides risk-based position sizing, maximum risk per trade, daily loss and
drawdown limits, exposure and position caps, liquidity/spread/volatility
restrictions, stop-loss validation and circuit breakers.

The risk engine independently approves or rejects every trade. Implemented in
Phase 5 (Independent Risk Engine). Intentionally empty of trading logic in
Phase 1.
"""

__all__ = ["engine", "rules"]
