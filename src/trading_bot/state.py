"""State management.

In-memory application state: risk state, position state, order state and
session state, with safe persistence boundaries.

Implemented in Phase 2 (Core Architecture) and extended in Phases 5 and 7.
"""

from __future__ import annotations

__all__ = ["AppState"]

class AppState:  # pragma: no cover - Phase 2 implementation
    """Holds mutable application state behind a controlled interface."""
