"""
Pick the watch-list coin farthest from EMA. No backtest / tune.

D = abs(close - EMA) / EMA locked at entry.
Mean-revert (REVERSE off): below EMA => LONG, above => SHORT.
Momentum (REVERSE on): below EMA => SHORT, above => LONG.
Both: fixed TP = D% in favor from fill, fixed SL = D% against from fill.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ema import compute_ema_series


@dataclass(frozen=True)
class EmaDevSnap:
    coin: str
    close: float
    ema: float
    abs_dev_pct: float
    signed_dev_pct: float
    signal_side: int  # +1 long (price below EMA), -1 short (price above)
    bar_t: int
    cross_bars: int = 1  # closed bars on this side of EMA (1 = just crossed)


@dataclass
class EmaDevTrade:
    coin: str
    side: str  # "long" | "short"
    dev_pct: float
    entry_px: float
    last_fill_px: float
    dca_done: bool
    entry_ema: float
    opened_bar_t: int
    last_exit_coin: str = ""
    last_exit_bar_t: int = 0
    last_exit_by_coin: dict = field(default_factory=dict)
    opened_at: float = 0.0  # unix seconds of fill; 0 = unknown (use bar / trade_store)
    mfe_pct: float = 0.0  # max favorable excursion from entry, %
    mae_pct: float = 0.0  # max adverse excursion from entry, %
    entry_ctx: dict = field(default_factory=dict)


def bar_t_to_ms(t: int) -> int:
    raw = int(t or 0)
    if raw > 10_000_000_000:
        return raw
    if raw > 1_000_000_000:
        return raw * 1000
    return raw


def exit_map_from_trade(trade: EmaDevTrade | None) -> dict[str, int]:
    if trade is None:
        return {}
    out: dict[str, int] = {}
    for key, val in (trade.last_exit_by_coin or {}).items():
        coin = str(key or "")
        if not coin:
            continue
        try:
            out[coin] = int(val)
        except (TypeError, ValueError):
            continue
    coin = str(trade.last_exit_coin or "")
    bar = int(trade.last_exit_bar_t or 0)
    if coin and bar > 0:
        out.setdefault(coin, bar)
    return out


def _load_exit_map(raw: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    blob = raw.get("last_exit_by_coin")
    if isinstance(blob, dict):
        for key, val in blob.items():
            coin = str(key or "")
            if not coin:
                continue
            try:
                out[coin] = int(val)
            except (TypeError, ValueError):
                continue
    coin = str(raw.get("last_exit_coin", "") or "")
    try:
        bar = int(raw.get("last_exit_bar_t", 0) or 0)
    except (TypeError, ValueError):
        bar = 0
    if coin and bar > 0:
        out.setdefault(coin, bar)
    return out


def coin_on_cooldown(
    coin: str,
    bar_t: int,
    last_exit_by_coin: dict[str, int] | None,
    cooldown_ms: int,
) -> bool:
    if cooldown_ms <= 0 or not coin:
        return False
    prev = int((last_exit_by_coin or {}).get(coin, 0) or 0)
    if prev <= 0:
        return False
    return abs(bar_t_to_ms(bar_t) - bar_t_to_ms(prev)) < cooldown_ms


def ema_entry_reject(
    *,
    d_pct: float,
    cross_bars: int,
    order_side: str,
    tape: dict,
    min_dev_pct: float = 0.0,
    min_cross_bars: int = 0,
    max_last_bar_bps: float = 0.0,
    min_er: float = 0.0,
    max_er: float = 1.0,
    min_d_atr_mult: float = 0.0,
    min_touches: int = 0,
    skip_mid_lo: float = 0.0,
    skip_mid_hi: float = 1.0,
    chase_high: float = 1.0,
    chase_low: float = 0.0,
    bucket: str = "",
    skip_gainer_long: bool = False,
    gainer_short_min_loc: float = 0.0,
    max_adverse_last_bps: float = 0.0,
    max_range_bps: float = 0.0,
) -> str | None:
    """Why this EMA snap should not be taken. None = ok.

    Rails = last N-bar high/low (tape loc / touches). Momentum still uses EMA
    side; this only blocks mid-range, chasing the far rail, and 24h-gainer
    longs / dip-shorts that the paper log stopped out.
    """
    d = float(d_pct)
    if d + 1e-12 < float(min_dev_pct):
        return "d"
    atr = float(tape.get("atr_pct") or 0.0)
    if min_d_atr_mult > 0 and atr > 0 and d + 1e-12 < float(min_d_atr_mult) * atr:
        return "d<atr"
    if int(cross_bars) < int(min_cross_bars):
        return "xbars"
    last_bps = float(tape.get("last_bar_bps") or 0.0)
    if max_last_bar_bps > 0 and abs(last_bps) > float(max_last_bar_bps) + 1e-12:
        return "last"
    side = str(order_side or "").lower()
    if max_adverse_last_bps > 0:
        lim = float(max_adverse_last_bps)
        if side == "long" and last_bps < -lim:
            return "last_adv"
        if side == "short" and last_bps > lim:
            return "last_adv"
    er = float(tape.get("er_20") or 0.0)
    if er + 1e-12 < float(min_er):
        return "er_low"
    if er > float(max_er) + 1e-12:
        return "er_high"
    touches = int(tape.get("touches") or 0)
    if min_touches > 0 and touches < int(min_touches):
        return "touches"
    box_bps = float(tape.get("box_bps") or tape.get("range_45_bps") or 0.0)
    if box_bps + 1e-12 < 50.0:
        return "box"
    rng = float(tape.get("range_45_bps") or box_bps or 0.0)
    if max_range_bps > 0 and rng > float(max_range_bps) + 1e-12:
        return "range"
    loc = tape.get("loc")
    if loc is None:
        return None
    loc_f = float(loc)
    if loc_f < 0.08 or loc_f > 0.92:
        return "through_rail"
    if float(skip_mid_lo) < loc_f < float(skip_mid_hi):
        return "mid"
    if side == "long" and loc_f > float(chase_high) + 1e-12:
        return "chase_high"
    if side == "short" and loc_f + 1e-12 < float(chase_low):
        return "chase_low"
    bkt = str(bucket or "").lower()
    if bkt == "gainer":
        if skip_gainer_long and side == "long":
            return "gainer_long"
        if side == "short" and loc_f + 1e-12 < float(gainer_short_min_loc):
            return "gainer_low"
    return None


def last_ema(closes: list[float], period: int) -> float | None:
    if period < 2 or len(closes) < period:
        return None
    series = compute_ema_series(closes, period)
    if not series:
        return None
    val = float(series[-1])
    if val <= 0:
        return None
    return val


def signed_dev_pct(price: float, ema: float) -> float:
    if ema <= 0:
        return 0.0
    return (price - ema) / ema * 100.0


def signal_side_from_price(price: float, ema: float) -> int:
    if price < ema:
        return 1
    if price > ema:
        return -1
    return 0


def snap_from_candles(
    coin: str,
    candles: list[dict],
    period: int,
) -> EmaDevSnap | None:
    if len(candles) < period:
        return None
    closes = [float(c["c"]) for c in candles]
    ema_tail = compute_ema_series(closes, period)
    if not ema_tail:
        return None
    ema = float(ema_tail[-1])
    if ema <= 0:
        return None
    close = closes[-1]
    if close <= 0:
        return None
    signed = signed_dev_pct(close, ema)
    side = signal_side_from_price(close, ema)
    if side == 0:
        return None
    return EmaDevSnap(
        coin=coin,
        close=close,
        ema=ema,
        abs_dev_pct=abs(signed),
        signed_dev_pct=signed,
        signal_side=side,
        bar_t=int(candles[-1]["t"]),
        cross_bars=bars_on_ema_side(closes, ema_tail),
    )


def bars_on_ema_side(closes: list[float], ema_tail: list[float]) -> int:
    """Closed bars the last close has stayed on the same side of EMA (min 1)."""
    n_ema = len(ema_tail)
    if n_ema == 0 or len(closes) < n_ema:
        return 1
    offset = len(closes) - n_ema
    last_side = signal_side_from_price(closes[-1], ema_tail[-1])
    if last_side == 0:
        return 1
    count = 1
    for j in range(n_ema - 2, -1, -1):
        side = signal_side_from_price(closes[offset + j], ema_tail[j])
        if side != last_side:
            break
        count += 1
    return count


def _dense_ranks(values: list[float], *, higher_is_better: bool) -> list[int]:
    ordered = sorted(set(values), reverse=higher_is_better)
    rank_of = {v: i + 1 for i, v in enumerate(ordered)}
    return [rank_of[v] for v in values]


def rank_snaps(
    snaps: list[EmaDevSnap],
    *,
    min_dev_pct: float = 0.0,
    skip_coin: str | None = None,
    skip_bar_t: int = 0,
    last_exit_by_coin: dict[str, int] | None = None,
    cooldown_ms: int = 0,
    rank_cross_age: bool = False,
    reverse: bool = False,
) -> list[EmaDevSnap]:
    eligible: list[EmaDevSnap] = []
    for snap in snaps:
        if skip_coin and snap.coin == skip_coin and snap.bar_t == skip_bar_t:
            continue
        if coin_on_cooldown(snap.coin, snap.bar_t, last_exit_by_coin, cooldown_ms):
            continue
        if snap.abs_dev_pct + 1e-12 < min_dev_pct:
            continue
        eligible.append(snap)
    if not eligible:
        return []
    if not rank_cross_age:
        return sorted(eligible, key=lambda s: (-float(s.abs_dev_pct), s.coin))
    # Equal-weight ranks: 1 is best. Mean-revert wants old crosses; momentum wants new.
    dev_ranks = _dense_ranks([s.abs_dev_pct for s in eligible], higher_is_better=True)
    age_ranks = _dense_ranks(
        [float(s.cross_bars) for s in eligible],
        higher_is_better=not reverse,
    )
    scored = list(zip(eligible, dev_ranks, age_ranks))
    scored.sort(key=lambda row: (row[1] + row[2], row[0].coin))
    return [row[0] for row in scored]


def pick_farthest(
    snaps: list[EmaDevSnap],
    *,
    min_dev_pct: float = 0.0,
    skip_coin: str | None = None,
    skip_bar_t: int = 0,
    last_exit_by_coin: dict[str, int] | None = None,
    cooldown_ms: int = 0,
    rank_cross_age: bool = False,
    reverse: bool = False,
) -> EmaDevSnap | None:
    ranked = rank_snaps(
        snaps,
        min_dev_pct=min_dev_pct,
        skip_coin=skip_coin,
        skip_bar_t=skip_bar_t,
        last_exit_by_coin=last_exit_by_coin,
        cooldown_ms=cooldown_ms,
        rank_cross_age=rank_cross_age,
        reverse=reverse,
    )
    return ranked[0] if ranked else None


def adverse_pct(side: str, from_px: float, now: float) -> float:
    """Percent move against the position from `from_px` to `now`."""
    if from_px <= 0:
        return 0.0
    if side == "long":
        return (from_px - now) / from_px * 100.0
    return (now - from_px) / from_px * 100.0


def should_tp(side: str, price: float, ema: float) -> bool:
    if ema <= 0 or price <= 0:
        return False
    if side == "long":
        return price >= ema
    return price <= ema


def tp_through_pct(side: str, price: float, ema: float) -> float:
    """How far price has gone through EMA in the TP direction (percent)."""
    if ema <= 0 or price <= 0:
        return 0.0
    if side == "long":
        if price < ema:
            return 0.0
        return (price - ema) / ema * 100.0
    if price > ema:
        return 0.0
    return (ema - price) / ema * 100.0


def sl_price(side: str, from_px: float, d_pct: float) -> float:
    m = max(0.0, d_pct) / 100.0
    if side == "long":
        return from_px * (1.0 - m)
    return from_px * (1.0 + m)


def tp_price_from_dev(side: str, from_px: float, d_pct: float) -> float:
    """Favorable target: D% beyond the fill."""
    m = max(0.0, d_pct) / 100.0
    if side == "long":
        return from_px * (1.0 + m)
    return from_px * (1.0 - m)


def should_sl_ema(side: str, price: float, ema: float) -> bool:
    """Momentum SL: mark has come back through EMA."""
    if ema <= 0 or price <= 0:
        return False
    if side == "long":
        return price <= ema
    return price >= ema


def sl_pct_to_ema(side: str, avg_entry: float, ema: float) -> float | None:
    """None when already through EMA (caller should flatten)."""
    if avg_entry <= 0 or ema <= 0:
        return None
    if side == "long":
        if ema >= avg_entry:
            return None
        return max(0.05, (avg_entry - ema) / avg_entry * 100.0)
    if ema <= avg_entry:
        return None
    return max(0.05, (ema - avg_entry) / avg_entry * 100.0)


def pct_from_avg_to_price(side: str, avg_entry: float, target: float) -> float:
    """Spot-% from average entry to a target price (for exchange TP/SL)."""
    if avg_entry <= 0 or target <= 0:
        return 0.05
    if side == "long":
        return max(0.05, abs(target - avg_entry) / avg_entry * 100.0)
    return max(0.05, abs(avg_entry - target) / avg_entry * 100.0)


def tp_pct_to_ema(side: str, avg_entry: float, ema: float) -> float | None:
    """None when price is already through EMA (caller should market-close)."""
    if avg_entry <= 0 or ema <= 0:
        return None
    if side == "long":
        if ema <= avg_entry:
            return None
        return max(0.05, (ema - avg_entry) / avg_entry * 100.0)
    if ema >= avg_entry:
        return None
    return max(0.05, (avg_entry - ema) / avg_entry * 100.0)


def protect_pcts(
    trade: EmaDevTrade,
    avg_entry: float,
    ema: float,
    *,
    dca_on: bool,
    reverse: bool = False,
) -> tuple[float, float] | None:
    """
    Fixed (tp_pct, sl_pct) = (D, D) from the entry gap. Not live EMA.
    None only when D is unusable.
    """
    del avg_entry, ema, dca_on, reverse
    d = max(0.0, float(trade.dev_pct))
    if d <= 0:
        return None
    pct = max(0.05, d)
    return pct, pct


def fill_pcts(
    entry_pct: float,
    dca_equity_pct: float,
    *,
    dca_on: bool,
    cap: float = 99.0,
) -> tuple[float, float]:
    """(entry_pct, dca_pct) of equity at that moment (entry now, DCA later)."""
    x = min(cap, max(0.1, float(entry_pct)))
    y = min(cap, max(0.1, float(dca_equity_pct)))
    if not dca_on:
        return x, 0.0
    return x, y


class EmaDevStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.trade: EmaDevTrade | None = None
        self._load()

    def _load(self) -> None:
        self.trade = None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        try:
            self.trade = EmaDevTrade(
                coin=str(raw["coin"]),
                side=str(raw["side"]),
                dev_pct=float(raw["dev_pct"]),
                entry_px=float(raw["entry_px"]),
                last_fill_px=float(raw.get("last_fill_px", raw["entry_px"])),
                dca_done=bool(raw.get("dca_done", False)),
                entry_ema=float(raw.get("entry_ema", 0.0)),
                opened_bar_t=int(raw.get("opened_bar_t", 0)),
                last_exit_coin=str(raw.get("last_exit_coin", "") or ""),
                last_exit_bar_t=int(raw.get("last_exit_bar_t", 0) or 0),
                last_exit_by_coin=_load_exit_map(raw),
                opened_at=float(raw.get("opened_at", 0.0) or 0.0),
                mfe_pct=float(raw.get("mfe_pct", 0.0) or 0.0),
                mae_pct=float(raw.get("mae_pct", 0.0) or 0.0),
                entry_ctx=dict(raw["entry_ctx"])
                if isinstance(raw.get("entry_ctx"), dict)
                else {},
            )
        except (KeyError, TypeError, ValueError):
            self.trade = None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.trade is None:
            if self.path.exists():
                try:
                    self.path.unlink()
                except OSError:
                    pass
            return
        tmp = self.path.with_suffix(".tmp")
        payload = asdict(self.trade)
        last_err: OSError | None = None
        for attempt in range(5):
            try:
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(tmp, self.path)
                return
            except OSError as exc:
                last_err = exc
                time.sleep(0.05 * (attempt + 1))
        if last_err is not None:
            raise last_err

    def open_trade(self, trade: EmaDevTrade) -> EmaDevTrade:
        prev = self.trade
        if prev is not None:
            trade.last_exit_coin = prev.last_exit_coin
            trade.last_exit_bar_t = prev.last_exit_bar_t
            trade.last_exit_by_coin = exit_map_from_trade(prev)
        self.trade = trade
        self._save()
        return trade

    def mark_dca(self, fill_px: float) -> None:
        if self.trade is None:
            return
        self.trade.last_fill_px = float(fill_px)
        self.trade.dca_done = True
        self._save()

    def close(self, *, coin: str, bar_t: int) -> None:
        exit_coin = coin
        exit_bar = bar_t
        exits = exit_map_from_trade(self.trade)
        if self.trade is not None:
            exit_coin = self.trade.coin or coin
            if exit_bar <= 0:
                exit_bar = self.trade.opened_bar_t
        if exit_coin and int(exit_bar) > 0:
            exits[str(exit_coin)] = int(exit_bar)
        self.trade = EmaDevTrade(
            coin="",
            side="",
            dev_pct=0.0,
            entry_px=0.0,
            last_fill_px=0.0,
            dca_done=False,
            entry_ema=0.0,
            opened_bar_t=0,
            last_exit_coin=exit_coin,
            last_exit_bar_t=int(exit_bar),
            last_exit_by_coin=exits,
        )
        self._save()

    def active(self) -> EmaDevTrade | None:
        t = self.trade
        if t is None or not t.coin:
            return None
        return t
