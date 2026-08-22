"""
TradingView / Hyperliquid chart RSI:
  - Length 14: ta.rsi(close, 14)  (Wilder RMA)
  - Smoothing SMA length 14: ta.sma(rsi, 14) on that series (second chart line)
Live bot signals use raw Wilder RSI; smoothed line is still computed for chart comparison.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RsiValues:
    raw: float  # ta.rsi — main RSI line
    smoothed: float  # ta.sma(rsi, 14) — smoothing line on HL chart


def wilder_rsi_series(closes: list[float], period: int) -> list[float]:
    """Equivalent to Pine ta.rsi(source, period)."""
    if len(closes) < period + 1:
        return []

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out: list[float] = []

    if avg_loss == 0:
        out.append(100.0)
    else:
        out.append(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out.append(100.0)
        else:
            rs = avg_gain / avg_loss
            out.append(100.0 - 100.0 / (1.0 + rs))

    return out


def _ta_sma_series(values: list[float], length: int) -> list[float]:
    """Equivalent to Pine ta.sma(values, length) at each bar."""
    if length <= 1:
        return values[:]
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - length + 1)
        window = values[start : i + 1]
        out.append(sum(window) / len(window))
    return out


def compute_rsi(
    closes: list[float],
    period: int = 14,
    smooth_period: int = 14,
) -> RsiValues | None:
    series = wilder_rsi_series(closes, period)
    if not series:
        return None
    smooth_series = _ta_sma_series(series, smooth_period)
    return RsiValues(raw=series[-1], smoothed=smooth_series[-1])


def parse_closes(candles: list[dict]) -> list[float]:
    return [float(c["c"]) for c in candles]


# Backward-compatible alias for backtests
_wilder_rsi_series = wilder_rsi_series


def compute_rsi_series(
    closes: list[float],
    period: int = 14,
    smooth_period: int = 14,
) -> tuple[list[float], list[float]]:
    """Full-series raw + smoothed RSI aligned to each close (nan until valid)."""
    import math

    n = len(closes)
    raw: list[float] = [math.nan] * n
    smooth: list[float] = [math.nan] * n
    wilder = wilder_rsi_series(closes, period)
    if not wilder:
        return raw, smooth
    for j, val in enumerate(wilder):
        raw[period + j] = val
    smooth_tail = _ta_sma_series(wilder, smooth_period)
    for j, val in enumerate(smooth_tail):
        smooth[period + j] = val
    return raw, smooth
