"""Provider-neutral, immutable strategy signal types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    EXIT = "exit"


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    direction: Direction
    entry_reference: float
    stop_loss_reference: float | None
    take_profit_reference: float | None
    timestamp: datetime
    strategy_id: str
    strength: float | None
    reason: str

    @property
    def identity(self) -> tuple[str, str, str, datetime]:
        return (self.symbol.strip().upper(), self.strategy_id.strip(), self.direction.value, self.timestamp)
