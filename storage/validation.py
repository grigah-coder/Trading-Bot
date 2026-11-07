"""Validation and normalisation shared by the storage repositories.

Every value that enters the database passes through one of these helpers.  The
goal is not to be a general-purpose validation library, but to guarantee that
persisted trading records have unambiguous identifiers, canonical symbols,
finite decimal values, and UTC timestamps.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from core.exceptions import TradingBotError


class StorageError(TradingBotError):
    """Base error for persistence failures."""

    code = "storage_error"


class DataValidationError(StorageError):
    """Raised when a record cannot be safely persisted."""

    code = "storage_invalid_data"


class DuplicateRecordError(StorageError):
    """Raised when a unique record would be silently overwritten."""

    code = "storage_duplicate_record"


class ReconciliationError(StorageError):
    """Raised when persisted state cannot be reconciled with the broker."""

    code = "storage_reconciliation_failed"


_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-/]{0,31}$")
_ALLOWED_SIDES = frozenset({"buy", "sell", "unknown"})


def validate_symbol(value: Any) -> str:
    """Return an upper-case, broker-compatible symbol."""
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError("symbol must be a non-empty string")
    symbol = value.strip().upper()
    if not _SYMBOL_PATTERN.match(symbol):
        raise DataValidationError(
            "symbol contains unsupported characters",
            context={"symbol": symbol},
        )
    return symbol


def validate_text(value: Any, name: str, *, max_length: int | None = None) -> str:
    """Return a stripped, non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{name} must be a non-empty string")
    text = value.strip()
    if max_length is not None and len(text) > max_length:
        raise DataValidationError(
            f"{name} cannot exceed {max_length} characters",
            context={"length": len(text)},
        )
    return text


def validate_decimal(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
    allow_none: bool = False,
) -> Decimal | None:
    """Convert ``value`` to a finite ``Decimal`` and enforce its range."""
    if value is None:
        if allow_none:
            return None
        raise DataValidationError(f"{name} is required")
    if isinstance(value, Decimal):
        parsed = value
    else:
        try:
            parsed = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise DataValidationError(f"{name} must be numeric") from exc
    if not parsed.is_finite():
        raise DataValidationError(f"{name} must be finite")
    if positive and parsed <= 0:
        raise DataValidationError(f"{name} must be greater than zero")
    if non_negative and parsed < 0:
        raise DataValidationError(f"{name} cannot be negative")
    return parsed


def validate_integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataValidationError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise DataValidationError(f"{name} must be at least {minimum}")
    return value


def validate_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DataValidationError(f"{name} must be a boolean")
    return value


def normalize_side(value: Any) -> str:
    """Normalise ``OrderSide.BUY``, ``"BUY"`` or ``"buy"`` to ``"buy"``."""
    text = str(getattr(value, "value", value) or "").strip().lower()
    if text not in _ALLOWED_SIDES:
        raise DataValidationError(
            "side must be buy, sell or unknown",
            context={"side": text},
        )
    return text


def normalize_status(value: Any) -> str:
    """Normalise an order status while accepting broker extensions."""
    text = str(getattr(value, "value", value) or "").strip().lower()
    if not text:
        raise DataValidationError("status must be a non-empty string")
    if len(text) > 48:
        raise DataValidationError("status cannot exceed 48 characters")
    return text


def utc_timestamp(value: Any, name: str = "timestamp") -> str:
    """Return an ISO-8601 timestamp in UTC.

    Naive timestamps are rejected: silently guessing a timezone is exactly the
    kind of ambiguity that makes trading records impossible to audit later.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise DataValidationError(f"{name} must be a non-empty timestamp")
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DataValidationError(f"{name} is not a valid ISO-8601 timestamp") from exc
    else:
        raise DataValidationError(f"{name} must be a datetime or ISO-8601 string")

    if moment.tzinfo is None or moment.utcoffset() is None:
        raise DataValidationError(f"{name} must include a timezone offset")
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def optional_utc_timestamp(value: Any, name: str = "timestamp") -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return utc_timestamp(value, name)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_timestamp() -> str:
    return utc_now().isoformat(timespec="microseconds")


def new_id(prefix: str | None = None) -> str:
    """Return a globally unique identifier, optionally namespaced."""
    identifier = uuid.uuid4().hex
    return f"{prefix}_{identifier}" if prefix else identifier
