"""Central trading engine.

Owns the application lifecycle: component initialization, the scheduler,
state management, configuration management, structured logging and exception
handling.

Implemented in Phase 2 (Core Architecture). No trading logic here.
"""

from __future__ import annotations

__all__ = ["TradingEngine"]

# Placeholder deferred to Phase 2. Intentionally contains no logic so that
# Phase 1 establishes structure without implementing trading behavior.
class TradingEngine:  # pragma: no cover - Phase 2 implementation
    """Application entry point that wires data, broker, risk, strategy and order
    components together and runs the trading loop."""
