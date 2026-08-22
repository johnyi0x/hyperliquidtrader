"""EMA and chop detection on closed candles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmaSnapshot:
    close: float
    ema: float
    candle_t: int
    closes: tuple[float, ...]
    highs: tuple[float, ...]
    lows: tuple[float, ...]


def compute_ema_series(closes: list[float], period: int) -> list[float]:
    if len(closes) < period:
        return []
    k = 2.0 / (period + 1)
    ema_vals: list[float] = []
    sma = sum(closes[:period]) / period
    ema_vals.append(sma)
    for price in closes[period:]:
        prev = ema_vals[-1]
        ema_vals.append(price * k + prev * (1 - k))
    return ema_vals


def build_snapshot(
    candles: list[dict],
    period: int,
) -> EmaSnapshot | None:
    if len(candles) < period + 2:
        return None
    closes = [float(c["c"]) for c in candles]
    highs = [float(c["h"]) for c in candles]
    lows = [float(c["l"]) for c in candles]
    ema_tail = compute_ema_series(closes, period)
    if not ema_tail:
        return None
    return EmaSnapshot(
        close=closes[-1],
        ema=ema_tail[-1],
        candle_t=int(candles[-1]["t"]),
        closes=tuple(closes),
        highs=tuple(highs),
        lows=tuple(lows),
    )


def count_ema_crosses(closes: tuple[float, ...], period: int, lookback: int) -> int:
    if len(closes) < period + lookback:
        lookback = max(1, len(closes) - period)
    start = len(closes) - lookback
    if start < period:
        return 0
    crosses = 0
    prev_above: bool | None = None
    for i in range(start, len(closes)):
        window = list(closes[: i + 1])
        ema_s = compute_ema_series(window, period)
        if not ema_s:
            continue
        above = closes[i] > ema_s[-1]
        if prev_above is not None and above != prev_above:
            crosses += 1
        prev_above = above
    return crosses


def bar_range_pct(
    highs: tuple[float, ...],
    lows: tuple[float, ...],
    closes: tuple[float, ...],
    lookback: int,
) -> float:
    if not closes:
        return 0.0
    n = min(lookback, len(closes))
    h = max(highs[-n:])
    l = min(lows[-n:])
    mid = closes[-1]
    if mid <= 0:
        return 0.0
    return (h - l) / mid * 100.0


def is_chop_market(
    snap: EmaSnapshot,
    *,
    period: int,
    lookback: int,
    max_crosses: int,
    range_min_pct: float,
    range_max_pct: float,
) -> bool:
    """Whipsaw: wide range but price crosses EMA too often."""
    crosses = count_ema_crosses(snap.closes, period, lookback)
    rng = bar_range_pct(snap.highs, snap.lows, snap.closes, lookback)
    if rng < range_min_pct:
        return False
    if rng > range_max_pct:
        return True
    return crosses >= max_crosses
