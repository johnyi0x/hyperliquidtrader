"""UTC timestamps for collector cycle hours."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

HOUR_S = 3600


def as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    raise TypeError(f"cannot parse timestamp: {value!r}")


def to_iso(value: Any) -> str:
    return as_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_unix(value: Any) -> int:
    return int(as_utc(value).timestamp())


def floor_hour(value: Any) -> datetime:
    dt = as_utc(value)
    return dt.replace(minute=0, second=0, microsecond=0)
