"""Compat shims — prefer pmf.price_engine (shared live/backtest math)."""

from __future__ import annotations

import numpy as np

from .price_engine import PriceEngine, load_candle_closes, load_research_into_engine  # noqa: F401


def compute_price_features(
    ts: np.ndarray,
    marks: np.ndarray,
    *,
    candle_closes=None,
    index_coin=None,
) -> dict[str, np.ndarray]:
    """Legacy batch arrays for older tests; live/backtest use PriceEngine."""
    n = int(len(ts))
    nc = int(marks.shape[1]) if marks.ndim == 2 else 0
    eng = PriceEngine()
    if index_coin:
        for j, coin in enumerate(index_coin):
            for i in range(n):
                px = float(marks[i, j]) if j < marks.shape[1] else 0.0
                if px > 0:
                    eng.ingest_mark(coin, float(ts[i]), px)
            if candle_closes and coin in candle_closes:
                for t_close, c in candle_closes[coin]:
                    # synthesize flat bar
                    eng.ingest_bar(
                        coin,
                        "15m",
                        [int((t_close - 900) * 1000), c, c, c, c, 0.0],
                    )
    out = {
        "ret_15m": np.zeros((n, nc), dtype=np.float64),
        "ret_30m": np.zeros((n, nc), dtype=np.float64),
        "ret_60m": np.zeros((n, nc), dtype=np.float64),
        "ema_bias": np.zeros((n, nc), dtype=np.float64),
        "atr_pct": np.zeros((n, nc), dtype=np.float64),
    }
    if not index_coin:
        return out
    for i in range(n):
        t = float(ts[i])
        for j, coin in enumerate(index_coin):
            out["ret_15m"][i, j] = eng.ret(coin, 900.0, t)
            out["ret_30m"][i, j] = eng.ret(coin, 1800.0, t)
            out["ret_60m"][i, j] = eng.ret(coin, 3600.0, t)
            out["ema_bias"][i, j] = eng.ema_bias(coin, t)
            out["atr_pct"][i, j] = eng.atr_pct(coin, t)
    return out


def ret_for_lookback(feats, i, coin_idx, lookback_s):
    if lookback_s <= 1200:
        key = "ret_15m"
    elif lookback_s <= 2700:
        key = "ret_30m"
    else:
        key = "ret_60m"
    arr = feats.get(key) if isinstance(feats, dict) else None
    if arr is None or i < 0 or coin_idx < 0:
        return 0.0
    if i >= arr.shape[0] or coin_idx >= arr.shape[1]:
        return 0.0
    v = float(arr[i, coin_idx])
    return v if np.isfinite(v) else 0.0
