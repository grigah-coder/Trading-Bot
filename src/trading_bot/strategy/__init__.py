"""Strategy and signal engine package.

Provides strategy interfaces, indicator calculations, entry/exit conditions,
signal generation/validation/expiration and duplicate-signal prevention.

Implemented in Phase 6 (Strategy and Signal Engine). FinBERT is kept out of the
live trading path until its incremental predictive value is statistically
demonstrated. Intentionally empty of trading logic in Phase 1.
"""

__all__ = ["base", "indicators", "signals"]
