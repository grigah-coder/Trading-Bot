"""Scheduler.

Drives the trading loop cadence, market-session-aware scheduling and timed
tasks (e.g. end-of-day summary).

Implemented in Phase 2 (Core Architecture).
"""

from __future__ import annotations

__all__ = ["Scheduler"]

class Scheduler:  # pragma: no cover - Phase 2 implementation
    """Schedules trading iterations and timed tasks."""
