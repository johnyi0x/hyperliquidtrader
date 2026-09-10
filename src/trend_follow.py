"""Ride the list's side: stay with the trend, flatten dumps.

Not the 18 MTF 1m scalps. Side comes from the pair list (majority / movers).
Entry on 15m when price is on the right side of EMA and not stretched too far.
Exit on the tighter of: close back through EMA, or ATR chandelier from the
extreme since entry. No take-profit. List drop is handled by the live bot.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .candles import INTERVAL_MS, fetch_closed_candles, probe_max_candles
from .indicators import atr, ema
from .mtf import align_to_ltf, closes_ms


@dataclass(frozen=True)
class TrendParams:
    interval: str = "15m"
    htf_interval: str = "1h"
    ema_period: int = 50
    atr_period: int = 14
    atr_k: float = 3.0
    chase_pct: float = 8.0
    entry_buf_atr: float = 0.35
    cooldown_bars: int = 3
    min_atr_pct: float = 0.20
    taker_fee_pct: float = 0.045

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_params(cfg: Any | None = None) -> TrendParams:
    if cfg is None:
        return TrendParams()
    return TrendParams(
        interval=str(getattr(cfg, "TREND_INTERVAL", "15m") or "15m"),
        htf_interval=str(getattr(cfg, "TREND_HTF_INTERVAL", "1h") or "1h"),
        ema_period=int(getattr(cfg, "TREND_EMA_PERIOD", 50) or 50),
        atr_period=int(getattr(cfg, "TREND_ATR_PERIOD", 14) or 14),
        atr_k=float(getattr(cfg, "TREND_ATR_K", 3.0) or 3.0),
        chase_pct=float(getattr(cfg, "TREND_CHASE_PCT", 8.0) or 8.0),
        entry_buf_atr=float(getattr(cfg, "TREND_ENTRY_BUF_ATR", 0.35) or 0.35),
        cooldown_bars=int(getattr(cfg, "TREND_COOLDOWN_BARS", 3) or 3),
        min_atr_pct=float(getattr(cfg, "TREND_MIN_ATR_PCT", 0.20) or 0.20),
        taker_fee_pct=float(getattr(cfg, "TAKER_FEE_PCT", 0.045) or 0.045),
    )


def params_from_dict(raw: dict[str, Any] | None, fallback: TrendParams) -> TrendParams:
    d = raw if isinstance(raw, dict) else {}
    return TrendParams(
        interval=str(d.get("interval") or fallback.interval),
        htf_interval=str(d.get("htf_interval") or fallback.htf_interval),
        ema_period=int(d.get("ema_period") or fallback.ema_period),
        atr_period=int(d.get("atr_period") or fallback.atr_period),
        atr_k=float(d.get("atr_k") or fallback.atr_k),
        chase_pct=float(d.get("chase_pct") or fallback.chase_pct),
        entry_buf_atr=float(d.get("entry_buf_atr") or fallback.entry_buf_atr),
        cooldown_bars=int(d.get("cooldown_bars") or fallback.cooldown_bars),
        min_atr_pct=float(d.get("min_atr_pct") or fallback.min_atr_pct),
        taker_fee_pct=float(d.get("taker_fee_pct") or fallback.taker_fee_pct),
    )


def _ohlc(candles: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.array([int(c["t"]) for c in candles], dtype=np.int64)
    h = np.array([float(c["h"]) for c in candles], dtype=np.float64)
    l = np.array([float(c["l"]) for c in candles], dtype=np.float64)
    c = np.array([float(c["c"]) for c in candles], dtype=np.float64)
    return t, h, l, c


def _htf_ema_on_ltf(
    exec_candles: list[dict],
    htf_candles: list[dict],
    *,
    exec_iv: str,
    htf_iv: str,
    ema_period: int,
) -> np.ndarray:
    if not htf_candles or not exec_candles:
        return np.full(len(exec_candles), np.nan)
    _t, _h, _l, hc = _ohlc(htf_candles)
    he = ema(hc, ema_period)
    ltf_cms = closes_ms(exec_candles, exec_iv)
    htf_cms = closes_ms(htf_candles, htf_iv)
    return align_to_ltf(ltf_cms, htf_cms, he, fill=np.nan)


def chandelier_stop(
    side: int,
    extreme: float,
    atr_v: float,
    k: float,
) -> float:
    a = max(0.0, float(atr_v)) * max(0.5, float(k))
    if side > 0:
        return float(extreme) - a
    return float(extreme) + a


def live_stop_px(
    side: int,
    ema_v: float,
    atr_v: float,
    extreme: float,
    k: float,
) -> float:
    """Tighter of EMA line and chandelier. Long → higher stop; short → lower."""
    ch = chandelier_stop(side, extreme, atr_v, k)
    ema_v = float(ema_v)
    if side > 0:
        return max(float(ch), ema_v)
    return min(float(ch), ema_v)


def entry_reason(
    *,
    side: int,
    close: float,
    ema_v: float,
    atr_v: float,
    htf_ema: float,
    chase_pct: float,
    entry_buf_atr: float,
    min_atr_pct: float,
) -> str:
    """Empty string = allowed. Else skip why."""
    if not np.isfinite(close) or close <= 0:
        return "bad_px"
    if not np.isfinite(ema_v) or ema_v <= 0:
        return "no_ema"
    if not np.isfinite(atr_v) or atr_v <= 0:
        return "no_atr"
    atr_pct = atr_v / close * 100.0
    if atr_pct + 1e-12 < float(min_atr_pct):
        return f"dead_atr {atr_pct:.2f}%<{min_atr_pct}"
    if np.isfinite(htf_ema) and htf_ema > 0:
        if side > 0 and close < htf_ema:
            return "htf_against"
        if side < 0 and close > htf_ema:
            return "htf_against"
    buf = float(entry_buf_atr) * atr_v
    if side > 0:
        if close < ema_v + buf:
            return "below_ema"
        ext = (close - ema_v) / ema_v * 100.0
        if ext > float(chase_pct):
            return f"chase {ext:.1f}%>{chase_pct}"
        return ""
    if close > ema_v - buf:
        return "above_ema"
    ext = (ema_v - close) / ema_v * 100.0
    if ext > float(chase_pct):
        return f"chase {ext:.1f}%>{chase_pct}"
    return ""


def exit_reason(
    *,
    side: int,
    close: float,
    ema_v: float,
    atr_v: float,
    extreme: float,
    k: float,
) -> str:
    if not np.isfinite(close) or not np.isfinite(ema_v) or not np.isfinite(atr_v):
        return ""
    sl = live_stop_px(side, ema_v, atr_v, extreme, k)
    slack = 0.10 * max(atr_v, 1e-12)
    if side > 0:
        if close <= sl:
            return "trail" if sl >= ema_v - 1e-12 else "ema"
        if close < ema_v - slack:
            return "ema"
        return ""
    if close >= sl:
        return "trail" if sl <= ema_v + 1e-12 else "ema"
    if close > ema_v + slack:
        return "ema"
    return ""


def simulate(
    exec_candles: list[dict],
    htf_candles: list[dict],
    *,
    side: int,
    params: TrendParams,
) -> dict[str, Any]:
    n = len(exec_candles)
    empty = {
        "trades": 0,
        "wins": 0,
        "return_pct": 0.0,
        "max_dd_pct": 0.0,
        "win_rate_pct": 0.0,
        "trades_per_day": 0.0,
        "score": -1e9,
    }
    need = max(params.ema_period, params.atr_period) + 5
    if n < need + 10:
        return empty
    _t, high, low, close = _ohlc(exec_candles)
    e = ema(close, params.ema_period)
    a = atr(high, low, close, params.atr_period)
    htf_e = _htf_ema_on_ltf(
        exec_candles,
        htf_candles,
        exec_iv=params.interval,
        htf_iv=params.htf_interval,
        ema_period=params.ema_period,
    )
    step_ms = float(INTERVAL_MS.get(params.interval, 900_000))
    days = max(1.0, n * step_ms / 86400_000.0)
    fee = float(params.taker_fee_pct) / 100.0
    side = 1 if int(side) > 0 else -1
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    trades = 0
    wins = 0
    in_pos = False
    entry = 0.0
    extreme = 0.0
    cool_until = -1
    start = need
    for i in range(start, n):
        if in_pos:
            if side > 0:
                extreme = max(extreme, float(high[i]))
            else:
                extreme = min(extreme, float(low[i]))
            why = exit_reason(
                side=side,
                close=float(close[i]),
                ema_v=float(e[i]),
                atr_v=float(a[i]),
                extreme=extreme,
                k=params.atr_k,
            )
            if not why:
                continue
            pnl = (float(close[i]) / entry - 1.0) * side
            pnl -= 2.0 * fee
            equity *= 1.0 + pnl
            trades += 1
            if pnl > 0:
                wins += 1
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)
            in_pos = False
            cool_until = i + max(0, int(params.cooldown_bars))
            continue
        if i < cool_until:
            continue
        why = entry_reason(
            side=side,
            close=float(close[i]),
            ema_v=float(e[i]),
            atr_v=float(a[i]),
            htf_ema=float(htf_e[i]) if i < len(htf_e) else float("nan"),
            chase_pct=params.chase_pct,
            entry_buf_atr=params.entry_buf_atr,
            min_atr_pct=params.min_atr_pct,
        )
        if why:
            continue
        in_pos = True
        entry = float(close[i])
        extreme = float(high[i]) if side > 0 else float(low[i])
    if in_pos:
        pnl = (float(close[-1]) / entry - 1.0) * side
        pnl -= 2.0 * fee
        equity *= 1.0 + pnl
        trades += 1
        if pnl > 0:
            wins += 1
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    wr = (wins / trades * 100.0) if trades else 0.0
    ret = (equity - 1.0) * 100.0
    tpd = trades / days
    score = ret - 0.8 * max_dd + 0.05 * wr
    if trades < 2:
        score = -1e9
    return {
        "trades": trades,
        "wins": wins,
        "return_pct": round(ret, 3),
        "max_dd_pct": round(max_dd, 3),
        "win_rate_pct": round(wr, 2),
        "trades_per_day": round(tpd, 3),
        "score": round(score, 4),
    }


def param_grid(cfg: Any | None = None) -> list[TrendParams]:
    base = default_params(cfg)
    emas = (50, 100)
    ks = (2.5, 3.0, 3.5)
    chases = (6.0, 10.0)
    out: list[TrendParams] = []
    for ema_p in emas:
        for k in ks:
            for chase in chases:
                out.append(
                    TrendParams(
                        interval=base.interval,
                        htf_interval=base.htf_interval,
                        ema_period=int(ema_p),
                        atr_period=base.atr_period,
                        atr_k=float(k),
                        chase_pct=float(chase),
                        entry_buf_atr=base.entry_buf_atr,
                        cooldown_bars=base.cooldown_bars,
                        min_atr_pct=base.min_atr_pct,
                        taker_fee_pct=base.taker_fee_pct,
                    )
                )
    return out


def tune_coin(
    info: Any,
    coin: str,
    *,
    side: int,
    data_dir: Path,
    requested_candles: int,
    logger: logging.Logger | None = None,
    cfg: Any | None = None,
) -> dict[str, Any] | None:
    log = logger or logging.getLogger("hl-multi")
    base = default_params(cfg)
    max_n = probe_max_candles(info, coin, base.interval, data_dir=data_dir, logger=log)
    exec_c = fetch_closed_candles(
        info,
        coin,
        base.interval,
        requested_candles,
        max_candles=max_n,
        data_dir=data_dir,
        logger=log,
    )
    htf_c = fetch_closed_candles(
        info,
        coin,
        base.htf_interval,
        min(requested_candles, 3000),
        max_candles=max_n,
        data_dir=data_dir,
        logger=log,
    )
    if len(exec_c) < 80:
        log.warning("TREND tune %s — not enough %s bars (%s)", coin, base.interval, len(exec_c))
        return None
    best: dict[str, Any] | None = None
    for p in param_grid(cfg):
        stats = simulate(exec_c, htf_c, side=side, params=p)
        if best is None or float(stats["score"]) > float(best["score"]):
            row = {**stats, "params": p.as_dict(), "side": int(side), "coin": coin}
            best = row
    if best is None or float(best.get("score", -1e9)) <= -1e8:
        log.warning("TREND tune %s — no usable combo, using defaults", coin)
        stats = simulate(exec_c, htf_c, side=side, params=base)
        return {
            **stats,
            "params": base.as_dict(),
            "side": int(side),
            "coin": coin,
            "fallback": True,
        }
    log.info(
        "BEST %s TREND@%s ema=%s atr_k=%s chase=%.0f%% | ret=%.1f%% wr=%.0f%% "
        "trades=%s (%.2f/d) dd=%.1f%%",
        coin,
        best["params"]["interval"],
        best["params"]["ema_period"],
        best["params"]["atr_k"],
        best["params"]["chase_pct"],
        best["return_pct"],
        best["win_rate_pct"],
        best["trades"],
        best["trades_per_day"],
        best["max_dd_pct"],
    )
    return best


def bars_need(params: TrendParams) -> int:
    return max(80, int(params.ema_period) + int(params.atr_period) + 10)


def last_state(
    exec_candles: list[dict],
    htf_candles: list[dict],
    params: TrendParams,
) -> dict[str, Any] | None:
    need = bars_need(params)
    if len(exec_candles) < need:
        return None
    _t, high, low, close = _ohlc(exec_candles)
    e = ema(close, params.ema_period)
    a = atr(high, low, close, params.atr_period)
    htf_e = _htf_ema_on_ltf(
        exec_candles,
        htf_candles,
        exec_iv=params.interval,
        htf_iv=params.htf_interval,
        ema_period=params.ema_period,
    )
    i = len(close) - 1
    if not np.isfinite(e[i]) or not np.isfinite(a[i]):
        return None
    htf_v = float(htf_e[i]) if i < len(htf_e) else float("nan")
    return {
        "bar_t": int(exec_candles[-1]["t"]),
        "close": float(close[i]),
        "high": float(high[i]),
        "low": float(low[i]),
        "ema": float(e[i]),
        "atr": float(a[i]),
        "htf_ema": htf_v,
        "ext_pct": abs(float(close[i]) - float(e[i])) / max(float(e[i]), 1e-12) * 100.0,
    }


@dataclass
class TrendTrade:
    coin: str
    side: str
    entry_px: float
    extreme: float
    sl_px: float
    opened_bar_t: int
    opened_at: float
    last_exit_bar_t: int = 0
    params: dict[str, Any] = field(default_factory=dict)


class TrendStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.params_by_coin: dict[str, dict[str, Any]] = {}
        self.trades: dict[str, TrendTrade] = {}
        self.last_exit_bar: dict[str, int] = {}
        self.updated_at: float = 0.0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        self.updated_at = float(raw.get("updated_at") or 0)
        per = raw.get("per_coin") or {}
        if isinstance(per, dict):
            self.params_by_coin = {str(k): dict(v) for k, v in per.items() if isinstance(v, dict)}
        exits = raw.get("last_exit_bar") or {}
        if isinstance(exits, dict):
            self.last_exit_bar = {
                str(k): int(v) for k, v in exits.items() if str(k)
            }
        for coin, row in (raw.get("open") or {}).items():
            if not isinstance(row, dict):
                continue
            self.trades[str(coin)] = TrendTrade(
                coin=str(coin),
                side=str(row.get("side") or "long"),
                entry_px=float(row.get("entry_px") or 0),
                extreme=float(row.get("extreme") or 0),
                sl_px=float(row.get("sl_px") or 0),
                opened_bar_t=int(row.get("opened_bar_t") or 0),
                opened_at=float(row.get("opened_at") or 0),
                last_exit_bar_t=int(row.get("last_exit_bar_t") or 0),
                params=dict(row.get("params") or {}),
            )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": self.updated_at,
            "updated_at_iso": datetime.fromtimestamp(
                self.updated_at or time.time(), tz=timezone.utc
            ).isoformat(),
            "per_coin": self.params_by_coin,
            "last_exit_bar": {c: int(v) for c, v in self.last_exit_bar.items()},
            "open": {
                c: {
                    "side": t.side,
                    "entry_px": t.entry_px,
                    "extreme": t.extreme,
                    "sl_px": t.sl_px,
                    "opened_bar_t": t.opened_bar_t,
                    "opened_at": t.opened_at,
                    "last_exit_bar_t": t.last_exit_bar_t,
                    "params": t.params,
                }
                for c, t in self.trades.items()
            },
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def save_tune(self, results: dict[str, dict[str, Any]], *, merge: bool = True) -> None:
        if merge:
            self.params_by_coin.update(results)
        else:
            self.params_by_coin = dict(results)
        self.updated_at = time.time()
        self._save()

    def params_for(self, coin: str, cfg: Any | None = None) -> TrendParams:
        base = default_params(cfg)
        row = self.params_by_coin.get(coin) or {}
        inner = row.get("params") if isinstance(row.get("params"), dict) else row
        return params_from_dict(inner if isinstance(inner, dict) else {}, base)

    def side_for(self, coin: str) -> int | None:
        row = self.params_by_coin.get(coin) or {}
        if "side" in row:
            try:
                return int(row["side"])
            except (TypeError, ValueError):
                return None
        return None

    def open_trade(self, trade: TrendTrade) -> None:
        self.trades[trade.coin] = trade
        self._save()

    def close_trade(self, coin: str, bar_t: int) -> None:
        t = self.trades.pop(coin, None)
        stamp = int(bar_t or 0)
        if stamp <= 0 and t is not None:
            stamp = int(t.opened_bar_t or 0)
        if stamp > 0:
            self.last_exit_bar[str(coin)] = stamp
        self._save()

    def mark_extreme(self, coin: str, extreme: float, sl_px: float) -> None:
        t = self.trades.get(coin)
        if t is None:
            return
        t.extreme = float(extreme)
        t.sl_px = float(sl_px)
        self._save()
