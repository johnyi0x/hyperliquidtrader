"""
Multi-timeframe consensus — true joint use of all Hyperliquid intervals.

Design (not independent per-TF bots):
  1. On every interval, compute a directional bias from closed bars only
     (EMA trend + RSI confirmation → −1 / 0 / +1).
  2. Align every higher-TF bias onto the execution-TF timeline with no lookahead
     (last HTF bar whose close time ≤ current LTF bar close time).
  3. Weight slower TFs more heavily; build a score in [−1, +1] and an agree-count.
  4. Allow long/short only when score AND agree-count clear thresholds.
  5. AND that permission with an LTF entry trigger (existing strategy masks).

Backtest and live share the same helpers so paper/live stay parity-locked.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .candles import INTERVAL_MS
from .indicators import build_features, ema
from .registry import STRATEGY_BY_ID, build_entry_mask


def close_ms(candle: dict[str, Any], interval: str) -> int:
    """Closed-bar end time in ms (HL `T` preferred)."""
    if candle.get("T") is not None:
        return int(candle["T"])
    step = INTERVAL_MS.get(interval, 60_000)
    return int(candle["t"]) + step - 1


def closes_ms(candles: Sequence[dict[str, Any]], interval: str) -> np.ndarray:
    return np.array([close_ms(c, interval) for c in candles], dtype=np.int64)


def arrays_from_candles(
    candles: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    closes = np.array([float(c["c"]) for c in candles], dtype=np.float64)
    highs = np.array([float(c["h"]) for c in candles], dtype=np.float64)
    lows = np.array([float(c["l"]) for c in candles], dtype=np.float64)
    vols = np.array([float(c.get("v", 0) or 0) for c in candles], dtype=np.float64)
    return closes, highs, lows, vols


def align_to_ltf(
    ltf_close_ms: np.ndarray,
    htf_close_ms: np.ndarray,
    htf_values: np.ndarray,
    *,
    fill: float = 0.0,
) -> np.ndarray:
    """
    Forward-fill HTF values onto LTF bars without lookahead.
    For each LTF bar i, use the last HTF bar with close_ms <= LTF close_ms[i].
    """
    n = len(ltf_close_ms)
    out = np.full(n, fill, dtype=np.float64)
    if len(htf_close_ms) == 0:
        return out
    # searchsorted: index of first HTF close strictly > ltf close → last usable is idx-1
    idx = np.searchsorted(htf_close_ms, ltf_close_ms, side="right") - 1
    valid = idx >= 0
    out[valid] = htf_values[idx[valid]]
    return out


def interval_bias(
    feats: dict[str, np.ndarray],
    *,
    ema_period: int = 50,
    rsi_long: float = 52.0,
    rsi_short: float = 48.0,
) -> np.ndarray:
    """
    Per-bar bias on one interval: −1 / 0 / +1.
    Trend (close vs EMA) and RSI momentum must agree; else neutral.
    """
    close = feats["close"]
    n = len(close)
    key = {20: "ema20", 50: "ema50", 100: "ema100"}.get(int(ema_period))
    series = feats[key] if key and key in feats else ema(close, int(ema_period))
    rsi = feats["rsi14"]

    trend = np.zeros(n, dtype=np.float64)
    trend[np.isfinite(series) & (close > series)] = 1.0
    trend[np.isfinite(series) & (close < series)] = -1.0

    mom = np.zeros(n, dtype=np.float64)
    mom[np.isfinite(rsi) & (rsi >= rsi_long)] = 1.0
    mom[np.isfinite(rsi) & (rsi <= rsi_short)] = -1.0

    bias = np.where(trend == mom, trend, 0.0)
    bias = np.where(np.isfinite(series) & np.isfinite(rsi), bias, 0.0)
    return bias.astype(np.float64)


def interval_weight(interval: str, power: float = 0.5) -> float:
    """Slower TFs weigh more: weight ∝ (interval_ms)^power."""
    ms = float(INTERVAL_MS.get(interval, 60_000))
    return max(1e-6, ms**float(power))


def build_consensus_masks(
    ltf_close_ms: np.ndarray,
    interval_biases: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    min_agree: int,
    min_score: float,
    weight_power: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    interval_biases: {interval: (close_ms, bias_series)}
    Returns (long_ok, short_ok, score, agree_signed) aligned to LTF length.
    agree_signed = (#long votes − #short votes) among non-neutral.
    """
    n = len(ltf_close_ms)
    long_ok = np.zeros(n, dtype=np.bool_)
    short_ok = np.zeros(n, dtype=np.bool_)
    score = np.zeros(n, dtype=np.float64)
    agree_signed = np.zeros(n, dtype=np.float64)
    if n == 0 or not interval_biases:
        return long_ok, short_ok, score, agree_signed

    w_sum = 0.0
    weighted = np.zeros(n, dtype=np.float64)
    long_votes = np.zeros(n, dtype=np.float64)
    short_votes = np.zeros(n, dtype=np.float64)

    for iv, (cms, bias) in interval_biases.items():
        aligned = align_to_ltf(ltf_close_ms, cms, bias, fill=0.0)
        w = interval_weight(iv, weight_power)
        w_sum += w
        weighted += w * aligned
        long_votes += (aligned > 0).astype(np.float64)
        short_votes += (aligned < 0).astype(np.float64)

    if w_sum <= 0:
        return long_ok, short_ok, score, agree_signed

    score = weighted / w_sum
    agree_signed = long_votes - short_votes
    thr = float(min_score)
    need = max(1, int(min_agree))
    long_ok = (score >= thr) & (long_votes >= need)
    short_ok = (score <= -thr) & (short_votes >= need)
    return long_ok, short_ok, score, agree_signed


def prepare_interval_biases(
    candles_by_interval: dict[str, list[dict[str, Any]]],
    *,
    ema_period: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build {interval: (close_ms, bias)} for every interval with enough bars."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for iv, candles in candles_by_interval.items():
        if len(candles) < 60:
            continue
        closes, highs, lows, vols = arrays_from_candles(candles)
        feats = build_features(closes, highs, lows, vols)
        bias = interval_bias(feats, ema_period=ema_period)
        out[iv] = (closes_ms(candles, iv), bias)
    return out


def combined_entry_mask(
    *,
    exec_interval: str,
    exec_candles: list[dict[str, Any]],
    candles_by_interval: dict[str, list[dict[str, Any]]],
    sid: int,
    side: int,
    p0: float,
    p1: float,
    p2: float,
    aux: float,
    ema_period: int,
    min_agree: int,
    min_score: float,
    weight_power: float,
    biases: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    LTF entry mask AND multi-TF consensus permission.
    Returns (combined_mask, closes, score) or None if insufficient data.

    Pass precomputed `biases` (from prepare_interval_biases) to avoid rebuilding
    the same EMA bias map on every trigger combo.
    """
    if len(exec_candles) < 80:
        return None
    # Ensure exec candles are in the map
    by_iv = dict(candles_by_interval)
    by_iv[exec_interval] = exec_candles

    closes, highs, lows, vols = arrays_from_candles(exec_candles)
    feats = build_features(closes, highs, lows, vols)
    trigger = build_entry_mask(sid, feats, p0, p1, p2, aux)

    if biases is None:
        biases = prepare_interval_biases(by_iv, ema_period=ema_period)
    if len(biases) < 2:
        return None

    ltf_cms = closes_ms(exec_candles, exec_interval)
    long_ok, short_ok, score, _ = build_consensus_masks(
        ltf_cms,
        biases,
        min_agree=min_agree,
        min_score=min_score,
        weight_power=weight_power,
    )
    if int(side) > 0:
        combined = trigger & long_ok
    else:
        combined = trigger & short_ok
    return combined, closes, score


def mtf_entry_signal_now(
    setup: Any,
    candles_by_interval: dict[str, list[dict[str, Any]]],
    *,
    biases: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> int:
    """
    Live/paper last-bar signal for an MTF setup.
    Returns backtest side +1 / −1 / 0 (same mask as tune).

    REVERSE_STRATEGY must NOT be applied here — flip only at order time.

    Pass precomputed `biases` from prepare_interval_biases to skip rebuilding
    multi-TF EMA bias maps when only the LTF trigger preset changes.
    """
    exec_iv = str(getattr(setup, "interval", "1m"))
    exec_candles = candles_by_interval.get(exec_iv) or []
    result = combined_entry_mask(
        exec_interval=exec_iv,
        exec_candles=exec_candles,
        candles_by_interval=candles_by_interval,
        sid=int(setup.sid),
        side=int(setup.side),
        p0=float(setup.p0),
        p1=float(setup.p1),
        p2=float(setup.p2),
        aux=float(setup.aux),
        ema_period=int(getattr(setup, "mtf_ema", 50) or 50),
        min_agree=int(getattr(setup, "mtf_min_agree", 3) or 3),
        min_score=float(getattr(setup, "mtf_min_score", 0.35) or 0.35),
        weight_power=float(getattr(setup, "mtf_weight_power", 0.5) or 0.5),
        biases=biases,
    )
    if result is None:
        return 0
    mask, _, _ = result
    if not bool(mask[-1]):
        return 0
    return int(setup.side)


def mtf_consensus_snapshot(
    setup: Any,
    candles_by_interval: dict[str, list[dict[str, Any]]],
) -> str:
    """Human-readable last-bar vote summary for scan logs."""
    exec_iv = str(getattr(setup, "interval", "1m"))
    exec_candles = candles_by_interval.get(exec_iv) or []
    if len(exec_candles) < 60:
        return "n/a"
    by_iv = dict(candles_by_interval)
    by_iv[exec_iv] = exec_candles
    ema_period = int(getattr(setup, "mtf_ema", 50) or 50)
    biases = prepare_interval_biases(by_iv, ema_period=ema_period)
    if not biases:
        return "n/a"
    ltf_cms = closes_ms(exec_candles, exec_iv)
    parts = []
    for iv in sorted(biases.keys(), key=lambda x: INTERVAL_MS.get(x, 0)):
        cms, bias = biases[iv]
        aligned = align_to_ltf(ltf_cms[-1:], cms, bias, fill=0.0)
        v = int(aligned[-1]) if len(aligned) else 0
        tag = "L" if v > 0 else ("S" if v < 0 else "·")
        parts.append(f"{iv}:{tag}")
    long_ok, short_ok, score, _ = build_consensus_masks(
        ltf_cms,
        biases,
        min_agree=int(getattr(setup, "mtf_min_agree", 3) or 3),
        min_score=float(getattr(setup, "mtf_min_score", 0.35) or 0.35),
        weight_power=float(getattr(setup, "mtf_weight_power", 0.5) or 0.5),
    )
    gate = "LONG_OK" if long_ok[-1] else ("SHORT_OK" if short_ok[-1] else "BLOCK")
    return f"{','.join(parts)} score={score[-1]:+.2f} {gate}"


def default_min_agree(n_intervals: int) -> int:
    """Require a majority-ish of intervals (at least 2)."""
    return max(2, (n_intervals + 1) // 2)


def mtf_param_grid(
    n_intervals: int,
    *,
    profile: str = "full",
) -> list[dict[str, float | int]]:
    """Compact consensus meta-param grid for refine stage."""
    base = default_min_agree(n_intervals)
    fast = str(profile or "full").strip().lower() == "fast"
    if fast:
        # Keep majority-ish agree + one looser; single score threshold.
        agrees = sorted({max(2, base - 1), base})
        emas = (50,)
        scores = (0.30,)
    else:
        agrees = sorted({max(2, base - 1), base, min(n_intervals, base + 1)})
        emas = (20, 50)
        scores = (0.30, 0.45)
    out: list[dict[str, float | int]] = []
    for ema_p in emas:
        for agree in agrees:
            for score in scores:
                out.append(
                    {
                        "mtf_ema": int(ema_p),
                        "mtf_min_agree": int(agree),
                        "mtf_min_score": float(score),
                        "mtf_weight_power": 0.5,
                    }
                )
    return out


def strategy_label(sid: int, side: int) -> str:
    name = STRATEGY_BY_ID.get(sid)
    base = name.name if name else f"sid{sid}"
    return f"mtf_{base}"
