"""Order management and position lifecycle package.

Provides the order state machine, submission, acknowledgment, partial-fill
handling, cancellation, replacement, stop-loss/take-profit management, bracket
orders, position tracking and reconciliation.

Implemented in Phase 7 (Order Management and Position Lifecycle).
Intentionally empty of trading logic in Phase 1.
"""

__all__ = ["lifecycle", "manager"]
