"""
Per-(coin, interval) closed-candle cache.

Hyperliquid candleSnapshot costs ~20 IP weight per call regardless of coin, so
the live loop must not refetch 1h/30m/15m/... on every 1m wake.

Closed bars never change. A 1h series is therefore identical to a fresh fetch
until the next 1h bar closes — the same last-closed-bar the tuner uses.
When a new bar of that interval has closed, we pull a short tail and merge by
open time `t` (full refetch on gaps or first fill).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .candles import INTERVAL_MS, interval_ms

# First fill matches live's previous request size (min_bars + 5).
_TAIL_BARS = 8


def snapshot_weight_budget(
    n_pairs: int,
    n_intervals: int,
    *,
    reserve: int,
    weight_per_call: int = 20,
    overhead: int = 20,
) -> dict[str, int]:
    """
    Estimate how many pairs fit in the 1200/min IP budget.

    Hour/half-hour marks refresh every TF at once (worst case).
    Typical minutes only refresh 1m (+ 3m/5m when those close).
    """
    cap = max(1, 1200 - int(reserve) - int(overhead))
    hour_calls = max(1, int(n_pairs) * max(1, int(n_intervals)))
    hour_weight = hour_calls * int(weight_per_call)
    hour_max = cap // (max(1, int(n_intervals)) * int(weight_per_call))
    steady_max = cap // int(weight_per_call)
    return {
        "cap": cap,
        "hour_calls": hour_calls,
        "hour_weight": hour_weight,
        "hour_max_pairs": max(1, int(hour_max)),
        "steady_max_pairs": max(1, int(steady_max)),
    }


def _closed_open_ms(interval: str, now_ms: int) -> int:
    """Open time of the latest fully closed bar (same bucket rule as live)."""
    step = interval_ms(interval)
    bucket = now_ms // step
    return (bucket - 1) * step


def _drop_open_bar(rows: list[dict], interval: str, now_ms: int) -> list[dict]:
    step = interval_ms(interval)
    closed: list[dict] = []
    for candle in rows:
        try:
            t = int(candle["t"])
        except (KeyError, TypeError, ValueError):
            continue
        close_ms = int(candle.get("T", t + step - 1))
        if close_ms < now_ms:
            closed.append(candle)
    return closed


def _dedupe_sort(rows: list[dict]) -> list[dict]:
    by_t: dict[int, dict] = {}
    for c in rows:
        try:
            by_t[int(c["t"])] = c
        except (KeyError, TypeError, ValueError):
            continue
    return [by_t[k] for k in sorted(by_t)]


@dataclass
class _Slot:
    rows: list[dict] = field(default_factory=list)
    last_open_ms: int = -1
    bucket: int = -1


class CandleBook:
    def __init__(self, info: Any, logger: logging.Logger | None = None) -> None:
        self.info = info
        self.log = logger or logging.getLogger("hl-multi")
        self._slots: dict[tuple[str, str], _Slot] = {}

    def _fetch_raw(self, coin: str, interval: str, bars: int) -> list[dict]:
        step = interval_ms(interval)
        end = int(time.time() * 1000)
        start = end - step * (max(1, int(bars)) + 5)
        raw = self.info.candles_snapshot(coin, interval, start, end)
        if not raw:
            return []
        return _dedupe_sort(list(raw))

    def get(self, coin: str, interval: str, min_bars: int = 40) -> list[dict]:
        """Closed candles for `coin`@`interval`, cached until that TF closes a new bar."""
        if interval not in INTERVAL_MS:
            raise ValueError(f"Unsupported interval {interval!r}")
        need = max(40, int(min_bars))
        now_ms = int(time.time() * 1000)
        step = interval_ms(interval)
        bucket = now_ms // step
        key = (str(coin), interval)
        slot = self._slots.get(key)
        want_open = _closed_open_ms(interval, now_ms)

        if (
            slot is not None
            and slot.bucket == bucket
            and slot.last_open_ms == want_open
            and len(slot.rows) >= need
        ):
            return slot.rows

        if slot is None or not slot.rows or len(slot.rows) < need:
            rows = _drop_open_bar(self._fetch_raw(coin, interval, need + 5), interval, now_ms)
        else:
            # New closed bar of this TF: short tail + merge. Gap → full refetch.
            tail = _drop_open_bar(
                self._fetch_raw(coin, interval, _TAIL_BARS), interval, now_ms
            )
            merged = _dedupe_sort(slot.rows + tail)
            if not merged:
                rows = merged
            else:
                last_t = int(merged[-1]["t"])
                gap = last_t > slot.last_open_ms + step + 1
                if gap or last_t < want_open:
                    rows = _drop_open_bar(
                        self._fetch_raw(coin, interval, need + 5), interval, now_ms
                    )
                else:
                    rows = merged
                    if len(rows) > need + 50:
                        rows = rows[-(need + 50) :]

        if rows and int(rows[-1]["t"]) < want_open:
            # Tail was too short (clock/API lag) — full window.
            rows = _drop_open_bar(
                self._fetch_raw(coin, interval, need + 5), interval, now_ms
            )

        last_open = int(rows[-1]["t"]) if rows else -1
        self._slots[key] = _Slot(rows=rows, last_open_ms=last_open, bucket=bucket)
        return rows
