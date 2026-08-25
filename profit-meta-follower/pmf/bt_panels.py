"""Precomputed per-tick indicator panel for fast backtest replay.

Built once after dataset load. PanelPrice wraps PriceEngine and serves O(1)
lookups when asof matches a book tick (or tick index is set); otherwise falls
back to the live PriceEngine so math stays identical.

Panel build walks each coin once (vectorized bar→tick stamp) — same formulas
as PriceEngine.rsi / ema_bias / atr_pct / ret / range_dump.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit

from .price_engine import PriceEngine

PANEL_TFS: tuple[str, ...] = ("1m", "15m", "1h")
PANEL_LOOKBACKS: tuple[float, ...] = (300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0)
_TF_INDEX: dict[str, int] = {tf: i for i, tf in enumerate(PANEL_TFS)}
_LB_INDEX: dict[float, int] = {float(lb): i for i, lb in enumerate(PANEL_LOOKBACKS)}


@dataclass
class IndPanel:
    """Per-tick × per-coin indicator arrays (same PriceEngine math)."""

    ts: np.ndarray  # (n_ticks,) float64
    price: np.ndarray  # (n_ticks, n_coins)
    rsi: np.ndarray  # (n_ticks, n_coins, 3) — 1m / 15m / 1h
    ema_bias: np.ndarray  # (n_ticks, n_coins, 3) — span=20
    atr_pct_15m: np.ndarray  # (n_ticks, n_coins) — period=14
    ret: np.ndarray  # (n_ticks, n_coins, 6)
    range_dump: np.ndarray  # (n_ticks, n_coins, 6) — tf=1m
    coin_index: dict[str, int]
    index_coin: list[str]
    tfs: tuple[str, ...] = PANEL_TFS
    lookbacks: tuple[float, ...] = PANEL_LOOKBACKS


def _sma_rsi_series(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """rsi_at[end] for end=0..n — identical to PriceEngine.rsi windowed SMA path."""
    n = len(closes)
    out = np.full(n + 1, 50.0, dtype=np.float64)
    if n < period + 2:
        return out
    d = np.diff(closes.astype(np.float64, copy=False))
    gains = np.maximum(d, 0.0)
    losses = np.maximum(-d, 0.0)
    cs_g = np.cumsum(gains)
    cs_l = np.cumsum(losses)
    # Vectorized for end = period+2 .. n
    ends = np.arange(period + 2, n + 1, dtype=np.int64)
    # d slice [start:end-1] with start=end-(period+1) → sum via cumsum at end-2 minus start-1
    g = cs_g[ends - 2] - np.where(ends - (period + 1) > 0, cs_g[ends - (period + 1) - 1], 0.0)
    l = cs_l[ends - 2] - np.where(ends - (period + 1) > 0, cs_l[ends - (period + 1) - 1], 0.0)
    avg_g = g / period
    avg_l = l / period
    rs = np.divide(avg_g, avg_l, out=np.zeros_like(avg_g), where=avg_l > 1e-12)
    vals = 100.0 - (100.0 / (1.0 + rs))
    zero_l = avg_l <= 1e-12
    vals = np.where(zero_l, np.where(avg_g > 0, 100.0, 50.0), vals)
    out[ends] = vals
    return out


def _atr_pct_series(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """atr_pct_at[end] for end=0..n — identical to PriceEngine.atr_pct bar path."""
    n = len(closes)
    out = np.zeros(n + 1, dtype=np.float64)
    if n < period + 1:
        return out
    h = highs.astype(np.float64, copy=False)
    l = lows.astype(np.float64, copy=False)
    c = closes.astype(np.float64, copy=False)
    prev = np.empty(n, dtype=np.float64)
    prev[0] = c[0]
    prev[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    cs = np.cumsum(tr)
    ends = np.arange(period + 1, n + 1, dtype=np.int64)
    # For end >= period+1: start = end-(period+1), use = tr[end-period:end], count=period
    lo = ends - period
    s = cs[ends - 1] - np.where(lo > 0, cs[lo - 1], 0.0)
    last = c[ends - 1]
    out[ends] = np.where(last > 0, (s / period) / last, 0.0)
    return out


@njit(cache=True)
def _close_ema_series_njit(closes: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    n = closes.shape[0]
    out = np.empty(n, dtype=np.float64)
    e = 0.0
    for i in range(n):
        c = closes[i]
        e = c if i == 0 else alpha * c + (1.0 - alpha) * e
        out[i] = e
    return out


def _close_ema_series(closes: np.ndarray, span: int = 20) -> np.ndarray:
    if len(closes) == 0:
        return np.empty(0, dtype=np.float64)
    return _close_ema_series_njit(np.asarray(closes, dtype=np.float64), int(span))


def _batch_price_at(
    asofs: np.ndarray,
    mark_t: np.ndarray | None,
    mark_p: np.ndarray | None,
    bar_packs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Vectorized PriceEngine.price_at (mark prefer, else densest candle close)."""
    n = len(asofs)
    out = np.zeros(n, dtype=np.float64)
    if mark_t is not None and len(mark_t):
        idx = np.searchsorted(mark_t, asofs, side="right") - 1
        ok = idx >= 0
        px = np.zeros(n, dtype=np.float64)
        if ok.any():
            px[ok] = mark_p[idx[ok]]  # type: ignore[index]
        good = ok & (px > 0)
        out[good] = px[good]
    need = out <= 0
    if not need.any():
        return out
    for tf in ("1m", "15m", "1h"):
        pack = bar_packs.get(tf)
        if pack is None:
            continue
        t_close, closes, _h, _l, _e = pack
        if len(t_close) == 0:
            continue
        end = np.searchsorted(t_close, asofs, side="right")
        hit = need & (end > 0)
        if not hit.any():
            continue
        c = closes[end[hit] - 1]
        take = c > 0
        ii = np.flatnonzero(hit)
        out[ii[take]] = c[take]
        need = out <= 0
        if not need.any():
            break
    return out


@njit(cache=True)
def _stamp_range_dump_core(
    highs: np.ndarray,
    closes: np.ndarray,
    ends: np.ndarray,
    lo_i_all: np.ndarray,
    out: np.ndarray,
) -> None:
    n = ends.shape[0]
    n_lb = lo_i_all.shape[0]
    for li in range(n_lb):
        for i in range(n):
            end = int(ends[i])
            if end <= 0:
                continue
            a = int(lo_i_all[li, i])
            if a >= end:
                take = max(3, end // 10)
                a = max(0, end - take)
            w_hi = highs[a]
            for k in range(a + 1, end):
                h = highs[k]
                if h > w_hi:
                    w_hi = h
            w_close = closes[end - 1]
            out[i, li] = (w_close / w_hi - 1.0) if w_hi > 0.0 else 0.0


def _stamp_range_dump(
    asofs: np.ndarray,
    t_close: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    lookbacks: tuple[float, ...],
) -> np.ndarray:
    """(n_ticks, n_lb) range_dump — same window rules as PriceEngine.range_dump."""
    n = len(asofs)
    n_lb = len(lookbacks)
    out = np.zeros((n, n_lb), dtype=np.float64)
    ends = np.searchsorted(t_close, asofs, side="right").astype(np.int64)
    lo_i_all = np.empty((n_lb, n), dtype=np.int64)
    for li, lb in enumerate(lookbacks):
        lo_i_all[li] = np.searchsorted(t_close, asofs - float(lb), side="left")
    _stamp_range_dump_core(highs, closes, ends, lo_i_all, out)
    return out


def build_ind_panel(
    ds: Any,
    *,
    progress: bool = False,
    eng: PriceEngine | None = None,
) -> IndPanel | None:
    """Precompute indicators at every book tick for every coin in index_coin."""
    eng = eng or getattr(ds, "price_engine", None)
    if eng is None or ds.n_ticks < 1 or ds.n_coins < 1:
        return None

    n = int(ds.n_ticks)
    nc = int(ds.n_coins)
    coins: list[str] = list(ds.index_coin)
    ts = np.asarray(ds.ts, dtype=np.float64)

    price = np.zeros((n, nc), dtype=np.float64)
    rsi = np.full((n, nc, len(PANEL_TFS)), 50.0, dtype=np.float64)
    ema = np.zeros((n, nc, len(PANEL_TFS)), dtype=np.float64)
    atr = np.zeros((n, nc), dtype=np.float64)
    ret = np.zeros((n, nc, len(PANEL_LOOKBACKS)), dtype=np.float64)
    rd = np.zeros((n, nc, len(PANEL_LOOKBACKS)), dtype=np.float64)

    t0 = time.time()
    if progress:
        print(f"Load: indicator panel ({n} ticks × {nc} coins)...", flush=True)

    span = 20
    need_ema = max(3, span // 2)
    period = 14

    for j, coin in enumerate(coins):
        marks = eng._marks.get(coin) or []
        if marks:
            mark_t = np.asarray([float(t) for t, _p in marks], dtype=np.float64)
            mark_p = np.asarray([float(p) for _t, p in marks], dtype=np.float64)
        else:
            mark_t = None
            mark_p = None

        bar_pack: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        rsi_ser: dict[str, np.ndarray] = {}
        for tf in PANEL_TFS:
            bars = eng._bars.get(coin, {}).get(tf) or []
            if not bars:
                continue
            t_close = np.asarray([b.t_close for b in bars], dtype=np.float64)
            closes = np.asarray([float(b.c) for b in bars], dtype=np.float64)
            highs = np.asarray([float(b.h) for b in bars], dtype=np.float64)
            lows = np.asarray([float(b.l) for b in bars], dtype=np.float64)
            emas = _close_ema_series(closes, span)
            bar_pack[tf] = (t_close, closes, highs, lows, emas)
            rsi_ser[tf] = _sma_rsi_series(closes, period)

        atr_ser = None
        pack15 = bar_pack.get("15m")
        if pack15 is not None:
            _t, closes15, highs15, lows15, _e = pack15
            atr_ser = _atr_pct_series(highs15, lows15, closes15, period)

        price[:, j] = _batch_price_at(ts, mark_t, mark_p, bar_pack)
        px = price[:, j]

        for ti, tf in enumerate(PANEL_TFS):
            pack = bar_pack.get(tf)
            if pack is None:
                continue
            t_close, closes, _h, _l, emas = pack
            ends = np.searchsorted(t_close, ts, side="right")
            rser = rsi_ser[tf]
            rsi[:, j, ti] = rser[ends]

            ok = ends >= need_ema
            if ok.any():
                ema_v = emas[ends[ok] - 1]
                last = np.where(px[ok] > 0, px[ok], closes[ends[ok] - 1])
                bias = np.where((ema_v > 0) & (last > 0), last / ema_v - 1.0, 0.0)
                ema[ok, j, ti] = bias
            for i in np.flatnonzero(~ok):
                ema[int(i), j, ti] = float(eng.ema_bias(coin, float(ts[i]), tf=tf, span=span))

        if atr_ser is not None and pack15 is not None:
            t_close15 = pack15[0]
            ends15 = np.searchsorted(t_close15, ts, side="right")
            ok_atr = ends15 >= period + 1
            atr[ok_atr, j] = atr_ser[ends15[ok_atr]]
            for i in np.flatnonzero(~ok_atr):
                atr[int(i), j] = float(eng.atr_pct(coin, float(ts[i]), tf="15m", period=period))
        else:
            for i in range(n):
                atr[i, j] = float(eng.atr_pct(coin, float(ts[i]), tf="15m", period=period))

        for li, lb in enumerate(PANEL_LOOKBACKS):
            base = _batch_price_at(ts - float(lb), mark_t, mark_p, bar_pack)
            good = (px > 0) & (base > 0)
            ret[good, j, li] = px[good] / base[good] - 1.0

        pack1m = bar_pack.get("1m")
        if pack1m is not None:
            t_close, closes, highs, _lows, _e = pack1m
            rd[:, j, :] = _stamp_range_dump(ts, t_close, highs, closes, PANEL_LOOKBACKS)
        else:
            for i in range(n):
                asof = float(ts[i])
                for li, lb in enumerate(PANEL_LOOKBACKS):
                    rd[i, j, li] = float(eng.range_dump(coin, float(lb), asof, tf="1m"))

        if progress and (j % 5 == 0 or j + 1 >= nc):
            elapsed = time.time() - t0
            frac = (j + 1) / nc
            eta = elapsed * (nc - j - 1) / (j + 1) if j + 1 else 0.0
            print(
                f"\r  panel coin {j + 1}/{nc} {frac:5.1%}  elapsed {int(elapsed)}s  eta {int(eta)}s   ",
                end="",
                flush=True,
            )

    if progress:
        print(f"\nLoad: indicator panel done in {time.time() - t0:.1f}s", flush=True)

    return IndPanel(
        ts=ts.copy(),
        price=price,
        rsi=rsi,
        ema_bias=ema,
        atr_pct_15m=atr,
        ret=ret,
        range_dump=rd,
        coin_index=dict(ds.coin_index),
        index_coin=list(coins),
    )


class PanelPrice:
    """PriceEngine-compatible facade: panel O(1) on ticks, else real engine."""

    __slots__ = ("_eng", "_panel", "_ds", "_tick_i", "_ts_index")

    def __init__(self, eng: PriceEngine, panel: IndPanel, ds: Any | None = None) -> None:
        self._eng = eng
        self._panel = panel
        self._ds = ds
        self._tick_i: int | None = None
        self._ts_index: dict[float, int] = {float(panel.ts[i]): i for i in range(len(panel.ts))}

    def set_tick(self, tick_i: int | None) -> None:
        if tick_i is None:
            self._tick_i = None
            return
        n = len(self._panel.ts)
        self._tick_i = int(tick_i) if 0 <= int(tick_i) < n else None

    def _resolve_tick(self, asof: float | None, tick_i: int | None = None) -> int | None:
        if tick_i is not None:
            i = int(tick_i)
            if 0 <= i < len(self._panel.ts):
                return i
            return None
        if self._tick_i is not None and asof is not None:
            i = self._tick_i
            if abs(float(self._panel.ts[i]) - float(asof)) < 1e-9:
                return i
        if asof is None:
            return self._tick_i
        t = float(asof)
        hit = self._ts_index.get(t)
        if hit is not None:
            return hit
        ts = self._panel.ts
        lo, hi = 0, len(ts)
        while lo < hi:
            mid = (lo + hi) // 2
            if ts[mid] <= t:
                lo = mid + 1
            else:
                hi = mid
        for j in (lo - 1, lo):
            if 0 <= j < len(ts) and abs(float(ts[j]) - t) < 1e-9:
                return j
        return None

    def _coin_j(self, coin: str) -> int | None:
        return self._panel.coin_index.get(str(coin))

    @property
    def _bars(self) -> dict:
        return self._eng._bars

    def _bar_end(self, coin: str, interval: str, asof: float) -> int:
        return self._eng._bar_end(coin, interval, asof)

    def has_coin(self, coin: str) -> bool:
        return self._eng.has_coin(coin)

    def candles_as_hl_dicts(
        self,
        coin: str,
        interval: str,
        asof: float,
        *,
        max_bars: int = 160,
    ) -> list[dict[str, Any]]:
        return self._eng.candles_as_hl_dicts(coin, interval, asof, max_bars=max_bars)

    def market_ctx_at(self, coin: str, asof: float, *, default_day_vol: float = 5_000_000.0) -> Any:
        return self._eng.market_ctx_at(coin, asof, default_day_vol=default_day_vol)

    def price_at(self, coin: str, asof: float, *, tick_i: int | None = None) -> float:
        i = self._resolve_tick(asof, tick_i)
        j = self._coin_j(coin) if i is not None else None
        if i is not None and j is not None:
            return float(self._panel.price[i, j])
        return float(self._eng.price_at(coin, asof))

    def rsi(
        self,
        coin: str,
        asof: float | None = None,
        *,
        tf: str = "15m",
        period: int = 14,
        tick_i: int | None = None,
    ) -> float:
        ti = _TF_INDEX.get(str(tf).strip().lower())
        if period == 14 and ti is not None:
            i = self._resolve_tick(asof, tick_i)
            j = self._coin_j(coin) if i is not None else None
            if i is not None and j is not None:
                return float(self._panel.rsi[i, j, ti])
        return float(self._eng.rsi(coin, asof, tf=tf, period=period))

    def ema_bias(
        self,
        coin: str,
        asof: float | None = None,
        *,
        tf: str = "15m",
        span: int = 20,
        tick_i: int | None = None,
    ) -> float:
        ti = _TF_INDEX.get(str(tf).strip().lower())
        if span == 20 and ti is not None:
            i = self._resolve_tick(asof, tick_i)
            j = self._coin_j(coin) if i is not None else None
            if i is not None and j is not None:
                return float(self._panel.ema_bias[i, j, ti])
        return float(self._eng.ema_bias(coin, asof, tf=tf, span=span))

    def atr_pct(
        self,
        coin: str,
        asof: float | None = None,
        *,
        tf: str = "15m",
        period: int = 14,
        tick_i: int | None = None,
    ) -> float:
        if str(tf).strip().lower() == "15m" and period == 14:
            i = self._resolve_tick(asof, tick_i)
            j = self._coin_j(coin) if i is not None else None
            if i is not None and j is not None:
                return float(self._panel.atr_pct_15m[i, j])
        return float(self._eng.atr_pct(coin, asof, tf=tf, period=period))

    def ret(
        self,
        coin: str,
        lookback_s: float,
        asof: float | None = None,
        *,
        tick_i: int | None = None,
    ) -> float:
        li = _LB_INDEX.get(float(lookback_s))
        if li is not None:
            i = self._resolve_tick(asof, tick_i)
            j = self._coin_j(coin) if i is not None else None
            if i is not None and j is not None:
                return float(self._panel.ret[i, j, li])
        return float(self._eng.ret(coin, lookback_s, asof))

    def range_dump(
        self,
        coin: str,
        lookback_s: float,
        asof: float | None = None,
        *,
        tf: str = "1m",
        tick_i: int | None = None,
    ) -> float:
        li = _LB_INDEX.get(float(lookback_s))
        if li is not None and str(tf).strip().lower() == "1m":
            i = self._resolve_tick(asof, tick_i)
            j = self._coin_j(coin) if i is not None else None
            if i is not None and j is not None:
                return float(self._panel.range_dump[i, j, li])
        return float(self._eng.range_dump(coin, lookback_s, asof, tf=tf))
