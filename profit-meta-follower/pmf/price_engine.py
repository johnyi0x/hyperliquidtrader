"""Unified price feature engine for backtest + live (identical math).

Uses every series research gather stores:
  - marks.jsonl / book mkt: mark, funding, oi, basis, day_vol
  - candles 1m / 15m / 1h: OHLCV closed bars

Both backtest (asof = book tick) and live (asof = now) call the same methods.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

INTERVAL_S: dict[str, float] = {
    "1m": 60.0,
    "15m": 900.0,
    "1h": 3600.0,
}
CANDLE_INTERVALS: tuple[str, ...] = ("1m", "15m", "1h")


@dataclass
class CandleBar:
    t_open_ms: int
    o: float
    h: float
    l: float
    c: float
    v: float
    interval: str

    @property
    def t_close(self) -> float:
        return self.t_open_ms / 1000.0 + float(INTERVAL_S.get(self.interval, 900.0))


def _coin_dir(coin: str) -> str:
    return str(coin).replace(":", "_").replace("/", "_")


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _research_day_dirs(research_dir: Path, *, max_days: int) -> list[Path]:
    if not research_dir.exists():
        return []
    days: list[tuple[datetime, Path]] = []
    for p in research_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            dt = datetime.strptime(p.name, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        days.append((dt, p))
    days.sort(key=lambda x: x[0])
    if max_days > 0 and len(days) > max_days:
        days = days[-max_days:]
    return [p for _dt, p in days]


def _bisect_right_ts(series: list[tuple[float, Any]], asof: float) -> int:
    lo, hi = 0, len(series)
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] <= asof:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _empty_bar_map() -> dict[str, list[CandleBar]]:
    return {iv: [] for iv in CANDLE_INTERVALS}


class PriceEngine:
    """Single source of truth for dump / trend / vol / RSI features."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self.log = logger or logging.getLogger("pmf-price")
        # coin -> [(ts, mark)]
        self._marks: dict[str, list[tuple[float, float]]] = defaultdict(list)
        # coin -> [(ts, funding)], [(ts, basis)], [(ts, oi)], [(ts, day_vol)]
        self._funding: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._basis: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._oi: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._day_vol: dict[str, list[tuple[float, float]]] = defaultdict(list)
        # coin -> interval -> bars sorted by t_close
        self._bars: dict[str, dict[str, list[CandleBar]]] = defaultdict(_empty_bar_map)
        self._last_candle_fetch = 0.0
        self._candle_jobs: list[tuple[str, str]] = []
        self._candle_job_i = 0
        self._seeded: set[tuple[str, str]] = set()
        # Lazily filled close-EMA series (same math as ema_bias loop over all bars).
        self._close_ema: dict[tuple[str, str, int], list[float]] = {}
        # candles_as_hl_dicts cache: (coin, interval, end_bar_t_open, max_bars) → rows
        self._hl_cache: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}

    def __getstate__(self) -> dict[str, Any]:
        # Process-pool workers on Windows pickle the dataset; drop non-picklable logger.
        state = dict(self.__dict__)
        state["log"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.log is None:
            self.log = logging.getLogger("pmf-price")
        if not hasattr(self, "_hl_cache") or self._hl_cache is None:
            self._hl_cache = {}
        if not hasattr(self, "_close_ema") or self._close_ema is None:
            self._close_ema = {}

    def _invalidate_bar_cache(self, coin: str, interval: str) -> None:
        dead = [k for k in self._close_ema if k[0] == coin and k[1] == interval]
        for k in dead:
            self._close_ema.pop(k, None)
        dead_hl = [k for k in self._hl_cache if k[0] == coin and k[1] == interval]
        for k in dead_hl:
            self._hl_cache.pop(k, None)

    # ----- ingest -----

    def ingest_mark(
        self,
        coin: str,
        ts: float,
        mark: float,
        *,
        funding: float = 0.0,
        basis: float = 0.0,
        oi: float = 0.0,
        day_vol: float = 0.0,
        aux: bool = True,
    ) -> None:
        """Record mark (and optionally funding/oi/basis/day_vol).

        aux=False densifies the mark path only — used when aligning book ticks so we
        never clobber richer marks.jsonl aux fields with zeros.
        """
        coin = str(coin)
        if mark <= 0 or ts < 0:
            return
        self._append_px(self._marks[coin], ts, mark)
        if not aux:
            return
        self._append_px(self._funding[coin], ts, float(funding))
        self._append_px(self._basis[coin], ts, float(basis))
        if oi > 0:
            self._append_px(self._oi[coin], ts, float(oi))
        if day_vol > 0:
            self._append_px(self._day_vol[coin], ts, float(day_vol))

    def ingest_bar(self, coin: str, interval: str, bar: CandleBar | list[Any] | dict[str, Any]) -> None:
        coin = str(coin)
        iv = str(interval).strip().lower()
        if iv not in INTERVAL_S:
            return
        parsed = self._parse_bar(iv, bar)
        if parsed is None:
            return
        series = self._bars[coin][iv]
        if series and series[-1].t_open_ms == parsed.t_open_ms:
            series[-1] = parsed
        elif series and series[-1].t_open_ms > parsed.t_open_ms:
            # insert sorted (rare)
            series.append(parsed)
            series.sort(key=lambda b: b.t_open_ms)
        else:
            series.append(parsed)
        self._invalidate_bar_cache(coin, iv)

    @staticmethod
    def _append_px(series: list[tuple[float, float]], ts: float, px: float) -> None:
        if series and abs(series[-1][0] - ts) < 1e-9:
            series[-1] = (ts, px)
        else:
            series.append((ts, px))

    @staticmethod
    def _parse_bar(interval: str, bar: CandleBar | list[Any] | dict[str, Any]) -> CandleBar | None:
        if isinstance(bar, CandleBar):
            return bar
        if isinstance(bar, dict):
            t_ms = int(bar.get("t") or 0)
            o = float(bar.get("o") or 0)
            h = float(bar.get("h") or 0)
            l = float(bar.get("l") or 0)
            c = float(bar.get("c") or 0)
            v = float(bar.get("v") or 0)
        elif isinstance(bar, (list, tuple)) and len(bar) >= 5:
            t_ms = int(bar[0] or 0)
            o, h, l, c = float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4])
            v = float(bar[5]) if len(bar) > 5 else 0.0
        else:
            return None
        if t_ms <= 0 or c <= 0:
            return None
        return CandleBar(t_open_ms=t_ms, o=o, h=h, l=l, c=c, v=v, interval=interval)

    def has_coin(self, coin: str) -> bool:
        return bool(self._marks.get(coin)) or any(self._bars.get(coin, {}).get(iv) for iv in CANDLE_INTERVALS)

    def span_s(self, coin: str) -> float:
        marks = self._marks.get(coin) or []
        if len(marks) >= 2:
            return float(marks[-1][0] - marks[0][0])
        for iv in ("1m", "15m", "1h"):
            bars = self._bars.get(coin, {}).get(iv) or []
            if len(bars) >= 2:
                return float(bars[-1].t_close - bars[0].t_close)
        return 0.0

    def candle_span_s(self, coin: str, interval: str) -> float:
        bars = self._bars.get(coin, {}).get(interval) or []
        if len(bars) < 2:
            return 0.0
        return float(bars[-1].t_close - bars[0].t_close)

    # ----- lookups -----

    def _last_mark(self, coin: str, asof: float) -> float | None:
        series = self._marks.get(coin) or []
        i = _bisect_right_ts(series, asof) - 1
        if i < 0:
            return None
        px = float(series[i][1])
        return px if px > 0 else None

    def _last_close(self, coin: str, interval: str, asof: float) -> float | None:
        bars = self._bars.get(coin, {}).get(interval) or []
        # binary search on t_close
        lo, hi = 0, len(bars)
        while lo < hi:
            mid = (lo + hi) // 2
            if bars[mid].t_close <= asof:
                lo = mid + 1
            else:
                hi = mid
        i = lo - 1
        if i < 0:
            return None
        c = float(bars[i].c)
        return c if c > 0 else None

    def _bar_end(self, coin: str, interval: str, asof: float) -> int:
        """Index past last bar with t_close ≤ asof (bisect_right)."""
        bars = self._bars.get(coin, {}).get(interval) or []
        lo, hi = 0, len(bars)
        t = float(asof)
        while lo < hi:
            mid = (lo + hi) // 2
            if bars[mid].t_close <= t:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _bars_asof(self, coin: str, interval: str, asof: float) -> list[CandleBar]:
        bars = self._bars.get(coin, {}).get(interval) or []
        end = self._bar_end(coin, interval, asof)
        return bars[:end]

    def _ensure_close_ema(self, coin: str, interval: str, span: int) -> list[float]:
        key = (coin, interval, int(span))
        cached = self._close_ema.get(key)
        if cached is not None:
            return cached
        bars = self._bars.get(coin, {}).get(interval) or []
        alpha = 2.0 / (float(span) + 1.0)
        out: list[float] = []
        ema = 0.0
        for i, b in enumerate(bars):
            c = float(b.c)
            ema = c if i == 0 else alpha * c + (1.0 - alpha) * ema
            out.append(ema)
        self._close_ema[key] = out
        return out

    def candles_as_hl_dicts(
        self,
        coin: str,
        interval: str,
        asof: float,
        *,
        max_bars: int = 160,
    ) -> list[dict[str, Any]]:
        """Last closed bars ≤ asof in Hyperliquid candle dict shape (no lookahead)."""
        coin_s = str(coin)
        iv = str(interval).strip().lower()
        bars = self._bars.get(coin_s, {}).get(iv) or []
        end = self._bar_end(coin_s, iv, asof)
        if end <= 0:
            return []
        mb = max(1, int(max_bars))
        end_t_open = int(bars[end - 1].t_open_ms)
        cache_key = (coin_s, iv, end_t_open, mb)
        cached = self._hl_cache.get(cache_key)
        if cached is not None:
            return cached
        start = max(0, end - mb)
        out: list[dict[str, Any]] = []
        for b in bars[start:end]:
            close_ms = int(b.t_close * 1000.0) - 1
            out.append(
                {
                    "t": int(b.t_open_ms),
                    "T": close_ms,
                    "o": float(b.o),
                    "h": float(b.h),
                    "l": float(b.l),
                    "c": float(b.c),
                    "v": float(b.v),
                }
            )
        self._hl_cache[cache_key] = out
        return out

    def price_at(self, coin: str, asof: float) -> float:
        """Canonical price: last mark ≤ asof, else densest candle close ≤ asof."""
        m = self._last_mark(coin, asof)
        if m is not None:
            return m
        for iv in ("1m", "15m", "1h"):
            c = self._last_close(coin, iv, asof)
            if c is not None:
                return c
        return 0.0

    def _tf_for_lookback(self, lookback_s: float) -> str:
        if lookback_s <= 1800.0:
            return "1m"
        if lookback_s <= 6 * 3600.0:
            return "15m"
        return "1h"

    def ret(self, coin: str, lookback_s: float, asof: float | None = None) -> float:
        t = float(asof if asof is not None else self._now_fallback(coin))
        now_px = self.price_at(coin, t)
        if now_px <= 0:
            return 0.0
        base_px = self.price_at(coin, t - float(lookback_s))
        if base_px <= 0:
            return 0.0
        return now_px / base_px - 1.0

    def ema_bias(self, coin: str, asof: float | None = None, *, tf: str = "15m", span: int = 20) -> float:
        t = float(asof if asof is not None else self._now_fallback(coin))
        end = self._bar_end(coin, tf, t)
        need = max(3, span // 2)
        if end < need:
            # fallback: mark EMA over last ~span marks
            marks = self._marks.get(coin) or []
            i = _bisect_right_ts(marks, t)
            window = [px for ts, px in marks[:i] if px > 0][-span:]
            if len(window) < 3:
                return 0.0
            alpha = 2.0 / (span + 1.0)
            ema = window[0]
            for px in window[1:]:
                ema = alpha * px + (1.0 - alpha) * ema
            last = window[-1]
            return last / ema - 1.0 if ema > 0 else 0.0
        emas = self._ensure_close_ema(coin, tf, span)
        ema = float(emas[end - 1])
        last = self.price_at(coin, t) or float((self._bars.get(coin, {}).get(tf) or [])[end - 1].c)
        return last / ema - 1.0 if ema > 0 and last > 0 else 0.0

    def atr_pct(self, coin: str, asof: float | None = None, *, tf: str = "15m", period: int = 14) -> float:
        t = float(asof if asof is not None else self._now_fallback(coin))
        bars = self._bars.get(coin, {}).get(tf) or []
        end = self._bar_end(coin, tf, t)
        if end < period + 1:
            # mark range fallback over ~period * tf seconds
            window_s = float(INTERVAL_S.get(tf, 900.0)) * period
            marks = self._marks.get(coin) or []
            i = _bisect_right_ts(marks, t)
            lo = t - window_s
            vals = [px for ts, px in marks[:i] if ts >= lo and px > 0]
            if len(vals) < 2:
                return 0.0
            last = vals[-1]
            return (max(vals) - min(vals)) / last if last > 0 else 0.0
        # Only need the last `period` true ranges — identical to scanning all then [-period:].
        start = max(0, end - (period + 1))
        prev_c = float(bars[start].c)
        trs: list[float] = []
        for b in bars[start + 1 : end]:
            tr = max(b.h - b.l, abs(b.h - prev_c), abs(b.l - prev_c))
            trs.append(tr)
            prev_c = float(b.c)
        use = trs[-period:]
        if not use:
            return 0.0
        atr = sum(use) / len(use)
        last = float(bars[end - 1].c)
        return atr / last if last > 0 else 0.0

    def rsi(self, coin: str, asof: float | None = None, *, tf: str = "15m", period: int = 14) -> float:
        """Wilder RSI; 50 = neutral when not enough data."""
        t = float(asof if asof is not None else self._now_fallback(coin))
        bars = self._bars.get(coin, {}).get(tf) or []
        end = self._bar_end(coin, tf, t)
        if end < period + 2:
            return 50.0
        # Identical to using all closes then closes[-period:] deltas.
        start = end - (period + 1)
        closes = [float(bars[i].c) for i in range(start, end)]
        gains = 0.0
        losses = 0.0
        for i in range(1, len(closes)):
            d = closes[i] - closes[i - 1]
            if d >= 0:
                gains += d
            else:
                losses -= d
        avg_g = gains / period
        avg_l = losses / period
        if avg_l <= 1e-12:
            return 100.0 if avg_g > 0 else 50.0
        rs = avg_g / avg_l
        return 100.0 - (100.0 / (1.0 + rs))

    def range_dump(self, coin: str, lookback_s: float, asof: float | None = None, *, tf: str = "1m") -> float:
        """(close - high) / high over lookback — negative when sold off from local high."""
        t = float(asof if asof is not None else self._now_fallback(coin))
        bars = self._bars.get(coin, {}).get(tf) or []
        end = self._bar_end(coin, tf, t)
        if end <= 0:
            return 0.0
        lo_t = t - float(lookback_s)
        # Walk backward from end (usually short lookback) instead of scanning all bars.
        window: list[CandleBar] = []
        for i in range(end - 1, -1, -1):
            b = bars[i]
            if b.t_close < lo_t:
                break
            window.append(b)
        if not window:
            take = max(3, end // 10)
            window = list(reversed(bars[max(0, end - take) : end]))
        else:
            window.reverse()
        hi = max(float(b.h) for b in window)
        close = float(window[-1].c)
        if hi <= 0:
            return 0.0
        return close / hi - 1.0

    def funding_at(self, coin: str, asof: float) -> float:
        series = self._funding.get(coin) or []
        i = _bisect_right_ts(series, asof) - 1
        return float(series[i][1]) if i >= 0 else 0.0

    def basis_at(self, coin: str, asof: float) -> float:
        series = self._basis.get(coin) or []
        i = _bisect_right_ts(series, asof) - 1
        return float(series[i][1]) if i >= 0 else 0.0

    def oi_at(self, coin: str, asof: float) -> float:
        series = self._oi.get(coin) or []
        i = _bisect_right_ts(series, asof) - 1
        return float(series[i][1]) if i >= 0 else 0.0

    def day_vol_at(self, coin: str, asof: float) -> float:
        series = self._day_vol.get(coin) or []
        i = _bisect_right_ts(series, asof) - 1
        return float(series[i][1]) if i >= 0 else 0.0

    def market_ctx_at(self, coin: str, asof: float, *, default_day_vol: float = 5_000_000.0) -> Any:
        """Build live-shaped MarketCtx from gathered marks (identical fields)."""
        from .types import MarketCtx

        mark = self.price_at(coin, asof)
        day_vol = self.day_vol_at(coin, asof)
        oi = self.oi_at(coin, asof)
        return MarketCtx(
            coin=str(coin),
            day_volume=float(day_vol if day_vol > 0 else default_day_vol),
            funding=self.funding_at(coin, asof),
            open_interest=float(oi if oi > 0 else default_day_vol),
            basis=self.basis_at(coin, asof),
            mark=float(mark or 0.0),
        )

    def _now_fallback(self, coin: str) -> float:
        marks = self._marks.get(coin) or []
        if marks:
            return float(marks[-1][0])
        for iv in CANDLE_INTERVALS:
            bars = self._bars.get(coin, {}).get(iv) or []
            if bars:
                return float(bars[-1].t_close)
        return 0.0

    # ----- live candle fill (rate-limited) -----

    def queue_candle_jobs(self, coins: Iterable[str], intervals: Iterable[str] | None = None) -> None:
        ivs = tuple(intervals) if intervals is not None else CANDLE_INTERVALS
        jobs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for c in coins:
            if not c:
                continue
            for iv in ivs:
                key = (c, iv)
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(key)
        # Prefer BTC + cold jobs first
        def _key(job: tuple[str, str]) -> tuple:
            c, iv = job
            need = 3600.0 if iv == "1h" else (1800.0 if iv == "15m" else 900.0)
            cold = 0 if self.candle_span_s(c, iv) + 30 < need else 1
            return (0 if c == "BTC" else 1, cold, 0 if job not in self._seeded else 1, iv, c)

        jobs.sort(key=_key)
        self._candle_jobs = jobs
        self._candle_job_i = 0

    def maybe_fetch_candles(
        self,
        info: Any,
        *,
        now: float,
        data_dir: Any,
        per_tick: int = 1,
        cooldown_s: float = 8.0,
        bars_1m: int = 120,
        bars_15m: int = 64,
        bars_1h: int = 48,
    ) -> int:
        if per_tick <= 0 or not self._candle_jobs:
            return 0
        if now - self._last_candle_fetch < float(cooldown_s):
            return 0
        from src.candles import fetch_closed_candles

        calls = 0
        n = min(int(per_tick), len(self._candle_jobs))
        for _ in range(n):
            coin, iv = self._candle_jobs[self._candle_job_i % len(self._candle_jobs)]
            self._candle_job_i += 1
            need = 3600.0 if iv == "1h" else (1800.0 if iv == "15m" else 900.0)
            if self.candle_span_s(coin, iv) + 30 >= need and (coin, iv) in self._seeded:
                continue
            req = bars_1m if iv == "1m" else (bars_15m if iv == "15m" else bars_1h)
            try:
                rows = fetch_closed_candles(
                    info, coin, iv, max(16, int(req)), data_dir=data_dir, logger=self.log
                )
            except Exception as exc:
                self.log.debug("Candle fetch %s %s: %s", coin, iv, exc)
                continue
            calls += 1
            for row in rows or []:
                self.ingest_bar(coin, iv, row)
            self._seeded.add((coin, iv))
            self.log.info(
                "Live candles %s %s bars=%s span=%.0fs",
                coin,
                iv,
                len(rows or []),
                self.candle_span_s(coin, iv),
            )
        self._last_candle_fetch = now
        return calls


def load_research_into_engine(
    research_dir: Path,
    *,
    max_days: int,
    coins: Iterable[str] | None = None,
    engine: PriceEngine | None = None,
    progress: bool = False,
    load_book_mkt: bool = True,
) -> PriceEngine:
    """Load marks + all candle intervals from research gather into a PriceEngine.

    load_book_mkt: also scan books.jsonl mkt vectors. Set False when marks.jsonl
    already loaded (avoids a second full pass over huge books files).
    """
    import time as _time
    import sys

    eng = engine or PriceEngine()
    want = {str(c) for c in (coins or [])} if coins is not None else None
    day_dirs = _research_day_dirs(research_dir, max_days=max_days)

    # Work units: mark rows + optional book rows + candle files
    def _nlines(path: Path) -> int:
        if not path.exists():
            return 0
        n = 0
        with path.open("rb") as fh:
            for _ in fh:
                n += 1
        return n

    total = 0
    candle_files: list[tuple[Path, Path]] = []  # (cdir, path) — cdir for coin name
    for day_dir in day_dirs:
        total += _nlines(day_dir / "marks.jsonl")
        if load_book_mkt:
            total += _nlines(day_dir / "books.jsonl")
        root = day_dir / "candles"
        if root.exists():
            for cdir in root.iterdir():
                if not cdir.is_dir():
                    continue
                for iv in CANDLE_INTERVALS:
                    path = cdir / f"{iv}.jsonl"
                    if path.exists():
                        candle_files.append((cdir, path))
                        total += max(1, _nlines(path))
    total = max(1, total)
    done = 0
    t0 = _time.time()

    def _bump(n: int = 1, label: str = "price engine") -> None:
        nonlocal done
        done = min(total, done + n)
        if progress and (done % 40 == 0 or done >= total):
            # inline progress (avoid importing research_load)
            elapsed = _time.time() - t0
            frac = done / total
            eta = elapsed * (total - done) / done if done else 0
            eta_s = f"{int(eta)}s" if done else "?"
            width = 28
            filled = int(width * frac)
            bar = "#" * filled + "-" * (width - filled)
            sys.stderr.write(
                f"\r[{bar}] {done}/{total} {frac:5.1%}  elapsed {int(elapsed)}s  eta {eta_s} {label}   "
            )
            sys.stderr.flush()
            if done >= total:
                sys.stderr.write("\n")
                sys.stderr.flush()

    if progress:
        _bump(0)

    for day_dir in day_dirs:
        for row in _iter_jsonl(day_dir / "marks.jsonl"):
            ts = float(row.get("ts") or 0)
            mkt = row.get("mkt") or {}
            _bump(1, "marks")
            if ts <= 0 or not isinstance(mkt, dict):
                continue
            for coin, vec in mkt.items():
                if want is not None and coin not in want:
                    continue
                if not isinstance(vec, (list, tuple)) or not vec:
                    continue
                mark = float(vec[0] or 0)
                funding = float(vec[1] or 0) if len(vec) > 1 else 0.0
                oi = float(vec[2] or 0) if len(vec) > 2 else 0.0
                basis = float(vec[3] or 0) if len(vec) > 3 else 0.0
                day_vol = float(vec[4] or 0) if len(vec) > 4 else 0.0
                eng.ingest_mark(coin, ts, mark, funding=funding, basis=basis, oi=oi, day_vol=day_vol)
        if load_book_mkt:
            for row in _iter_jsonl(day_dir / "books.jsonl"):
                ts = float(row.get("ts") or 0)
                mkt = row.get("mkt") or {}
                _bump(1, "book-mkt")
                if ts <= 0 or not isinstance(mkt, dict):
                    continue
                for coin, vec in mkt.items():
                    if want is not None and coin not in want:
                        continue
                    if not isinstance(vec, (list, tuple)) or not vec:
                        continue
                    eng.ingest_mark(
                        coin,
                        ts,
                        float(vec[0] or 0),
                        funding=float(vec[1] or 0) if len(vec) > 1 else 0.0,
                        oi=float(vec[2] or 0) if len(vec) > 2 else 0.0,
                        basis=float(vec[3] or 0) if len(vec) > 3 else 0.0,
                        day_vol=float(vec[4] or 0) if len(vec) > 4 else 0.0,
                    )

    safe_want = {_coin_dir(c): c for c in (want or [])} if want is not None else None
    for cdir, path in candle_files:
        iv = path.stem  # 1m / 15m / 1h
        for row in _iter_jsonl(path):
            coin = str(row.get("coin") or "").strip()
            if not coin:
                coin = safe_want.get(cdir.name, cdir.name) if safe_want is not None else cdir.name
            if want is not None and coin not in want:
                alt = safe_want.get(cdir.name) if safe_want else None
                if alt:
                    coin = alt
                else:
                    _bump(1, f"candles {iv}")
                    continue
            eng.ingest_bar(coin, iv, row.get("bar"))
            _bump(1, f"candles {iv}")
    if progress and done < total:
        done = total - 1
        _bump(1, "price engine")
    return eng


def load_candle_closes(
    research_dir: Path,
    coin_index: dict[str, int],
    *,
    max_days: int,
    interval: str = "15m",
) -> dict[str, list[tuple[float, float]]]:
    """Backward-compat helper used by older tests."""
    eng = load_research_into_engine(research_dir, max_days=max_days, coins=coin_index.keys())
    out: dict[str, list[tuple[float, float]]] = {c: [] for c in coin_index}
    for coin in coin_index:
        for b in eng._bars.get(coin, {}).get(interval) or []:
            out[coin].append((b.t_close, b.c))
    return out
