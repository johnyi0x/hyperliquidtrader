"""Numpy indicators for multi-strategy backtests (no lookahead)."""

from __future__ import annotations

import numpy as np


def wilder_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1 or period < 2:
        return out
    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_g = gains[:period].mean()
    avg_l = losses[:period].mean()
    out[period] = 100.0 if avg_l == 0 else 100.0 - (100.0 / (1.0 + avg_g / avg_l))
    for i in range(period, len(delta)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 if avg_l == 0 else 100.0 - (100.0 / (1.0 + avg_g / avg_l))
    return out


def ema(closes: np.ndarray, period: int) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period or period < 2:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = closes[:period].mean()
    for i in range(period, n):
        out[i] = alpha * closes[i] + (1.0 - alpha) * out[i - 1]
    return out


def sma(closes: np.ndarray, period: int) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if period < 1 or n < period:
        return out
    csum = np.cumsum(closes, dtype=np.float64)
    out[period - 1] = csum[period - 1] / period
    out[period:] = (csum[period:] - csum[:-period]) / period
    return out


def rolling_std(closes: np.ndarray, period: int) -> np.ndarray:
    """Population std (ddof=0) over each window — same as np.std(window) per bar."""
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if period < 2 or n < period:
        return out
    c = np.asarray(closes, dtype=np.float64)
    csum = np.cumsum(c)
    csum2 = np.cumsum(c * c)
    # window sums for indices period-1 .. n-1
    s = np.empty(n - period + 1, dtype=np.float64)
    s2 = np.empty(n - period + 1, dtype=np.float64)
    s[0] = csum[period - 1]
    s2[0] = csum2[period - 1]
    if n > period:
        s[1:] = csum[period:] - csum[:-period]
        s2[1:] = csum2[period:] - csum2[:-period]
    mean = s / period
    var = s2 / period - mean * mean
    # tiny negatives from fp noise
    np.maximum(var, 0.0, out=var)
    out[period - 1 :] = np.sqrt(var)
    return out


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    prev = np.roll(closes, 1)
    prev[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev), np.abs(lows - prev)))
    out = sma(tr, period)
    return out


def pct_change(closes: np.ndarray, bars: int) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if bars <= 0 or n <= bars:
        return out
    prev = closes[:-bars]
    cur = closes[bars:]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[bars:] = (cur - prev) / np.where(prev == 0, np.nan, prev) * 100.0
    return out


def zscore(closes: np.ndarray, period: int) -> np.ndarray:
    m = sma(closes, period)
    s = rolling_std(closes, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (closes - m) / np.where(s == 0, np.nan, s)


def build_features(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> dict[str, np.ndarray]:
    vol_ma = sma(volumes, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = volumes / np.where(vol_ma == 0, np.nan, vol_ma)
    return {
        "close": closes.astype(np.float64),
        "high": highs.astype(np.float64),
        "low": lows.astype(np.float64),
        "volume": volumes.astype(np.float64),
        "rsi14": wilder_rsi(closes, 14),
        "rsi7": wilder_rsi(closes, 7),
        "ema20": ema(closes, 20),
        "ema50": ema(closes, 50),
        "ema100": ema(closes, 100),
        "sma20": sma(closes, 20),
        "sma50": sma(closes, 50),
        "atr14": atr(highs, lows, closes, 14),
        "z20": zscore(closes, 20),
        "z50": zscore(closes, 50),
        "ret5": pct_change(closes, 5),
        "ret12": pct_change(closes, 12),
        "ret24": pct_change(closes, 24),
        "vol_ratio": vol_ratio.astype(np.float64),
    }
