"""Backtesting and quantitative validation package.

Provides the backtesting engine with realistic execution simulation,
transaction costs, slippage, spread assumptions, stop/take-profit simulation,
portfolio risk controls and performance metrics (Sharpe, Sortino, drawdown,
win rate, expectancy, ...).

Implemented in Phase 9 (Backtesting and Quantitative Validation).
Intentionally empty of trading logic in Phase 1.
"""

__all__ = ["engine", "metrics"]
