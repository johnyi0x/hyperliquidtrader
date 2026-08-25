"""Shared crowd+price gates (backtest + live) — uses PriceEngine callbacks."""

from __future__ import annotations

from typing import Any, Callable


def _dump_thr(cfg: Any) -> float:
    thr = float(getattr(cfg, "DUMP_RET_PCT", -0.02) or -0.02)
    if thr > 0:
        thr = -abs(thr)
    return thr


def _lookback_s(cfg: Any) -> float:
    return float(getattr(cfg, "DUMP_LOOKBACK_S", 1800.0) or 1800.0)


def price_gate_ok(
    *,
    gate: str | None,
    coin: str,
    side: str,
    managed: set[str],
    cfg: Any,
    ret: Callable[[str, float], float],
    ema_bias: Callable[[str], float],
    atr_pct: Callable[[str], float],
    rsi: Callable[[str], float] | None = None,
    range_dump: Callable[[str, float], float] | None = None,
    has_btc: Callable[[], bool] | None = None,
) -> bool:
    """Return False to drop this crowd vote (skip open / force flatten)."""
    if not gate:
        return True
    side_l = str(side).lower()
    look = _lookback_s(cfg)
    dump_thr = _dump_thr(cfg)
    range_thr = float(getattr(cfg, "DUMP_RANGE_PCT", -0.025) or -0.025)
    if range_thr > 0:
        range_thr = -abs(range_thr)

    if gate == "btcdump":
        if has_btc is not None and not has_btc():
            return True
        btc_ret = ret("BTC", look)
        btc_rd = range_dump("BTC", look) if range_dump else 0.0
        return btc_ret > dump_thr and btc_rd > range_thr

    if gate == "dump":
        r = ret(coin, look)
        rd = range_dump(coin, look) if range_dump else 0.0
        if side_l == "long":
            if r <= dump_thr or rd <= range_thr:
                return False
        else:
            # Shorts: exit/block when price pumps hard vs lookback.
            if r >= -dump_thr:
                return False
        return True

    if gate == "trend":
        bias = ema_bias(coin)
        need = float(getattr(cfg, "TREND_BIAS_MIN", 0.0) or 0.0)
        if side_l == "long" and bias < need:
            return False
        if side_l == "short" and bias > -need:
            return False
        return True

    if gate == "vol":
        if coin in managed:
            return True
        atr = atr_pct(coin)
        max_atr = float(getattr(cfg, "MAX_ATR_PCT", 0.04) or 0.04)
        return atr <= max_atr

    if gate == "rsi":
        if rsi is None:
            return True
        val = rsi(coin)
        hi = float(getattr(cfg, "RSI_MAX", 72.0) or 72.0)
        lo = float(getattr(cfg, "RSI_MIN", 28.0) or 28.0)
        if coin not in managed:
            if side_l == "long" and val >= hi:
                return False
            if side_l == "short" and val <= lo:
                return False
        elif side_l == "long" and val >= hi:
            r = ret(coin, min(look, 900.0))
            if r <= dump_thr * 0.5:
                return False
        return True

    return True
