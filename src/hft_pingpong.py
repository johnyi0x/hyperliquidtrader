"""
Maker ping-pong for Hyperliquid perps (Railway-safe, REST poll).

Picks a choppy wide-enough book, quotes one clip post-only on both sides when
flat, skews to flatten when filled, and market-exits on trend / box break /
whip / inventory timeout so size is not left behind when 3x books run.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BookSnap:
    bid: float
    ask: float
    bid_sz: float
    ask_sz: float
    mid: float
    spread_bps: float
    imbalance: float  # +1 bid-heavy, -1 ask-heavy
    tick: float


@dataclass(frozen=True)
class ChopSnap:
    coin: str
    close: float
    atr_bps: float
    er: float  # Kaufman efficiency (1 = one-way, 0 = chop)
    path_over_net: float
    range_bps: float
    box_high: float
    box_low: float
    box_high_fast: float
    box_low_fast: float
    up_frac: float
    burst: bool  # last bars one-way
    last_bar_bps: float
    bar_t: int
    score: float


@dataclass
class HftState:
    coin: str
    opened_at: float = 0.0
    last_fill_at: float = 0.0
    last_fill_buy: bool = False
    last_score_at: float = 0.0
    last_exit_coin: str = ""
    last_exit_until: float = 0.0
    cleared_legacy: bool = False
    pause_until: float = 0.0
    fav_px: float = 0.0  # best mid in our favor since fill (give-back flatten)
    flat_streak: int = 0  # consecutive empty position polls after a fill


@dataclass(frozen=True)
class HftDecision:
    flatten: str | None
    pause: str | None
    quote_bid: bool
    quote_ask: bool
    bid_reduce: bool
    ask_reduce: bool
    timeout_s: float
    vol_scale: float
    note: str


def book_from_l2(l2: dict, sz_decimals: int) -> BookSnap | None:
    try:
        bids = l2["levels"][0]
        asks = l2["levels"][1]
    except (KeyError, IndexError, TypeError):
        return None
    if not bids or not asks:
        return None
    try:
        bid = float(bids[0]["px"])
        ask = float(asks[0]["px"])
        bid_sz = float(bids[0].get("sz") or 0)
        ask_sz = float(asks[0].get("sz") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if bid <= 0 or ask <= 0 or ask <= bid:
        return None
    mid = (bid + ask) / 2.0
    spread_bps = (ask - bid) / mid * 10_000.0
    tot = bid_sz + ask_sz
    imb = 0.0 if tot <= 0 else (bid_sz - ask_sz) / tot
    tick = 0.0
    if len(bids) >= 2:
        tick = abs(float(bids[0]["px"]) - float(bids[1]["px"]))
    elif len(asks) >= 2:
        tick = abs(float(asks[0]["px"]) - float(asks[1]["px"]))
    if tick <= 0:
        tick = 10 ** (-max(1, 6 - int(sz_decimals)))
    return BookSnap(
        bid=bid,
        ask=ask,
        bid_sz=bid_sz,
        ask_sz=ask_sz,
        mid=mid,
        spread_bps=spread_bps,
        imbalance=imb,
        tick=tick,
    )


def chop_from_candles(coin: str, candles: list[dict], lookback: int) -> ChopSnap | None:
    n = max(12, int(lookback))
    if len(candles) < n:
        return None
    tail = candles[-n:]
    closes = [float(c["c"]) for c in tail]
    highs = [float(c["h"]) for c in tail]
    lows = [float(c["l"]) for c in tail]
    if any(x <= 0 for x in closes + highs + lows):
        return None
    close = closes[-1]
    path = 0.0
    up = 0
    down = 0
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        path += abs(d)
        if d > 0:
            up += 1
        elif d < 0:
            down += 1
    net = abs(closes[-1] - closes[0])
    er = 1.0 if path <= 1e-12 else min(1.0, net / path)
    path_over_net = path / max(net, close * 1e-8)
    box_high = max(highs)
    box_low = min(lows)
    if box_high <= box_low:
        return None
    range_bps = (box_high - box_low) / close * 10_000.0
    trs: list[float] = []
    for i in range(1, len(closes)):
        trs.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    atr = sum(trs[-14:]) / max(1, min(14, len(trs)))
    atr_bps = atr / close * 10_000.0
    sides = up + down
    up_frac = (up / sides) if sides else 0.5
    last_bar_bps = 0.0
    if len(closes) >= 2 and closes[-2] > 0:
        last_bar_bps = abs(closes[-1] - closes[-2]) / closes[-2] * 10_000.0
    # Two consecutive 1m closes same way + a real move = tape is running.
    burst = False
    if len(closes) >= 3 and closes[-3] > 0:
        same_up = closes[-1] > closes[-2] > closes[-3]
        same_dn = closes[-1] < closes[-2] < closes[-3]
        burst = (same_up or same_dn) and last_bar_bps >= max(10.0, 0.65 * atr_bps)
    fast_n = min(12, len(highs))
    box_high_fast = max(highs[-fast_n:])
    box_low_fast = min(lows[-fast_n:])
    # Prefer two-sided calm tape. High ATR names get picked off.
    balance = 1.0 - abs(up_frac - 0.5) * 2.0
    score = (
        min(path_over_net, 8.0)
        * max(0.05, balance)
        * min(range_bps, 80.0)
        * (1.0 - er)
        * max(0.2, 1.0 - min(atr_bps, 50.0) / 50.0)
    )
    if burst:
        score *= 0.15
    return ChopSnap(
        coin=coin,
        close=close,
        atr_bps=max(0.0, atr_bps),
        er=er,
        path_over_net=path_over_net,
        range_bps=range_bps,
        box_high=box_high,
        box_low=box_low,
        box_high_fast=box_high_fast,
        box_low_fast=box_low_fast,
        up_frac=up_frac,
        burst=burst,
        last_bar_bps=last_bar_bps,
        bar_t=int(tail[-1]["t"]),
        score=float(score),
    )


def chop_reject_reason(
    s: ChopSnap,
    *,
    max_er: float,
    max_range_bps: float,
    skip_coin: str | None = None,
    skip_until: float = 0.0,
    now: float = 0.0,
) -> str | None:
    """Why this chop snapshot is not quotable, or None if it is eligible."""
    if skip_coin and s.coin == skip_coin and now < skip_until:
        return "cooldown"
    if s.er > max_er + 1e-12:
        return f"er={s.er:.2f}>{max_er:.2f}"
    if s.burst:
        return f"burst last={s.last_bar_bps:.0f}b"
    if s.range_bps < 10.0:
        return f"rng={s.range_bps:.0f}b<10"
    if s.range_bps > max_range_bps:
        return f"rng={s.range_bps:.0f}b>{max_range_bps:.0f}"
    if s.atr_bps < 4.0:
        return f"atr={s.atr_bps:.1f}b"
    if s.atr_bps > 32.0:
        return f"atr={s.atr_bps:.0f}b>32"
    if s.up_frac < 0.28 or s.up_frac > 0.72:
        return f"one-way={s.up_frac:.0%}"
    return None


def rank_chop(
    snaps: list[ChopSnap],
    *,
    max_er: float,
    skip_coin: str | None = None,
    skip_until: float = 0.0,
    now: float = 0.0,
    max_range_bps: float = 900.0,
) -> list[ChopSnap]:
    eligible: list[ChopSnap] = []
    for s in snaps:
        if chop_reject_reason(
            s,
            max_er=max_er,
            max_range_bps=max_range_bps,
            skip_coin=skip_coin,
            skip_until=skip_until,
            now=now,
        ):
            continue
        eligible.append(s)
    eligible.sort(key=lambda s: (-s.score, s.coin))
    return eligible


def pick_chop(
    snaps: list[ChopSnap],
    *,
    max_er: float,
    skip_coin: str | None = None,
    skip_until: float = 0.0,
    now: float = 0.0,
) -> ChopSnap | None:
    ranked = rank_chop(
        snaps,
        max_er=max_er,
        skip_coin=skip_coin,
        skip_until=skip_until,
        now=now,
    )
    return ranked[0] if ranked else None


def filter_hft_entries(entries: list, *, max_max_leverage: int) -> tuple[list, list[str]]:
    """Keep only names whose exchange maxLev is ≤ cap (HFT-only pair cut)."""
    cap = max(1, int(max_max_leverage or 1))
    kept: list = []
    dropped: list[str] = []
    for e in entries:
        market = getattr(e, "market", None)
        mx = int(getattr(market, "max_leverage", 0) or 0)
        label = str(getattr(e, "api_coin", None) or getattr(e, "coin", "?") or "?")
        if mx <= cap:
            kept.append(e)
        else:
            dropped.append(f"{label}({mx}x)")
    return kept, dropped


def target_clip_notional(
    *,
    equity: float,
    available: float,
    leverage: int,
    min_notional: float,
    max_notional: float,
) -> float | None:
    """
    One small clip: at least exchange min, never a fat order.
    Tiny accounts use the min notional. Returns None if free margin cannot fund it.
    """
    lev = max(1, int(leverage))
    floor = max(float(min_notional) * 1.08, float(min_notional))
    cap = max(floor, float(max_notional))
    if equity > 0:
        cap = min(cap, max(floor, equity * 0.20))
    if equity > 0 and equity >= 50.0:
        target = min(cap, max(floor, equity * 0.12))
    else:
        target = floor
    margin = target / lev
    if available + 1e-9 >= margin:
        return target
    affordable = max(0.0, available) * 0.90 * lev
    if affordable + 1e-9 >= float(min_notional):
        return max(float(min_notional), affordable)
    return None


def vol_scale(atr_bps: float) -> float:
    return max(0.70, min(2.8, (max(4.0, atr_bps) / 16.0)))


def decide(
    *,
    book: BookSnap,
    chop: ChopSnap | None,
    side: str | None,
    size: float,
    entry_px: float,
    last_fill_at: float,
    now: float,
    min_spread_bps: float,
    max_spread_bps: float,
    max_er: float,
    box_break_bps: float,
    base_timeout_s: float,
    clip_sz: float,
    fav_px: float = 0.0,
    holding_quotes: bool = False,
) -> HftDecision:
    """
    Flat: two-sided maker quotes on a calm, fee-wide book.
    In a clip: only reduce-only the other side. Market flatten is a last resort
    (hard stop / stale loser). Immediate taker exits after a maker fill lose
    taker fee (0.045%) plus spread and turn MM into a bleed.
    """
    del clip_sz, box_break_bps, fav_px
    vs = vol_scale(chop.atr_bps if chop else 16.0)
    timeout = max(20.0, min(60.0, float(base_timeout_s)))
    in_pos = bool(side) and size > 1e-12
    hold = (now - last_fill_at) if last_fill_at > 0 else 0.0
    last_bar = float(chop.last_bar_bps if chop else 0.0)
    atr = float(chop.atr_bps if chop else 16.0)
    loc = 0.5
    if chop is not None and chop.box_high > chop.box_low:
        loc = (book.mid - chop.box_low) / (chop.box_high - chop.box_low)

    def _reduce() -> HftDecision:
        if side == "long":
            return HftDecision(
                None, None, False, True, False, True, timeout, vs, "reduce long"
            )
        return HftDecision(
            None, None, True, False, True, False, timeout, vs, "reduce short"
        )

    if (not in_pos) and last_fill_at > 0:
        return HftDecision(
            None, None, False, False, False, False, timeout, vs, "wait flat"
        )

    if in_pos and book.spread_bps > max_spread_bps:
        return _reduce()

    if book.spread_bps > max_spread_bps:
        return HftDecision(
            None, "spread_wide", False, False, False, False, timeout, vs, "spread wide"
        )

    # Do not keep two-sided quotes on a book that compressed inside the fee floor.
    # holding_quotes used to skip this and rest ZRO at 0.9bps.
    if (not in_pos) and book.spread_bps + 1e-12 < min_spread_bps:
        return HftDecision(
            None, "spread_tight", False, False, False, False, timeout, vs, "spread tight"
        )

    trending = bool(chop and (chop.er > max_er or chop.burst))
    if (not in_pos) and (not holding_quotes) and trending:
        return HftDecision(
            None, "trend", False, False, False, False, timeout, vs, "trend pause"
        )
    if (not in_pos) and (not holding_quotes) and last_bar >= max(18.0, 2.0 * book.spread_bps):
        return HftDecision(
            None, "whip_bar", False, False, False, False, timeout, vs, "whip pause"
        )
    if (not in_pos) and (not holding_quotes) and atr >= 32.0:
        return HftDecision(
            None, "atr_spike", False, False, False, False, timeout, vs, "atr spike"
        )
    if (not in_pos) and (not holding_quotes) and (loc < 0.22 or loc > 0.78):
        return HftDecision(
            None, "edge", False, False, False, False, timeout, vs, "edge of box"
        )
    if (not in_pos) and (not holding_quotes) and abs(book.imbalance) > 0.88:
        return HftDecision(
            None, "imbalance", False, False, False, False, timeout, vs, "one-sided book"
        )

    pnl_bps = 0.0
    if in_pos and entry_px > 0:
        if side == "long":
            pnl_bps = (book.mid - entry_px) / entry_px * 10_000.0
        else:
            pnl_bps = (entry_px - book.mid) / entry_px * 10_000.0

    # Hard stop only. Taker fee is 4.5bps — do not market-out for a 1–5bps dip.
    stop_bps = 20.0
    if in_pos and pnl_bps <= -stop_bps:
        return HftDecision(
            "inv_stop", None, False, False, False, False, timeout, vs, "inv stop"
        )
    if in_pos:
        return _reduce()

    return HftDecision(
        None,
        None,
        True,
        True,
        False,
        False,
        timeout,
        vs,
        "two-sided",
    )


def quote_px_ok(existing_px: float, target_px: float, tick: float, mid: float) -> bool:
    if existing_px <= 0 or target_px <= 0:
        return False
    tol = max(tick * 1.1, mid * 4.0e-4)
    return abs(existing_px - target_px) <= tol


class HftStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state: HftState | None = None
        self._load()

    def _load(self) -> None:
        self.state = None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return
        if not isinstance(raw, dict) or not raw.get("coin"):
            return
        try:
            self.state = HftState(
                coin=str(raw["coin"]),
                opened_at=float(raw.get("opened_at", 0) or 0),
                last_fill_at=float(raw.get("last_fill_at", 0) or 0),
                last_fill_buy=bool(raw.get("last_fill_buy", False)),
                last_score_at=float(raw.get("last_score_at", 0) or 0),
                last_exit_coin=str(raw.get("last_exit_coin", "") or ""),
                last_exit_until=float(raw.get("last_exit_until", 0) or 0),
                cleared_legacy=bool(raw.get("cleared_legacy", False)),
                pause_until=float(raw.get("pause_until", 0) or 0),
                fav_px=float(raw.get("fav_px", 0) or 0),
                flat_streak=int(raw.get("flat_streak", 0) or 0),
            )
        except (KeyError, TypeError, ValueError):
            self.state = None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.state is None or not self.state.coin:
            if self.path.exists():
                try:
                    self.path.unlink()
                except OSError:
                    pass
            return
        tmp = self.path.with_suffix(".tmp")
        payload = asdict(self.state)
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

    def set_coin(self, coin: str, *, now: float) -> HftState:
        prev = self.state
        st = HftState(coin=coin, last_score_at=now, cleared_legacy=False)
        if prev is not None:
            st.last_exit_coin = prev.last_exit_coin
            st.last_exit_until = prev.last_exit_until
        self.state = st
        self._save()
        return st

    def touch_fill(self, *, now: float, is_buy: bool | None = None) -> None:
        if self.state is None:
            return
        if self.state.opened_at <= 0:
            self.state.opened_at = now
        self.state.last_fill_at = now
        if is_buy is not None:
            self.state.last_fill_buy = bool(is_buy)
        self.state.fav_px = 0.0
        self.state.flat_streak = 0
        self._save()

    def mark_fav(self, mid: float, side: str) -> None:
        if self.state is None or mid <= 0:
            return
        cur = float(self.state.fav_px or 0)
        if side == "long":
            self.state.fav_px = mid if cur <= 0 else max(cur, mid)
        elif side == "short":
            self.state.fav_px = mid if cur <= 0 else min(cur, mid)
        else:
            return
        self._save()

    def mark_flat(self) -> None:
        if self.state is None:
            return
        self.state.last_fill_at = 0.0
        self.state.fav_px = 0.0
        self.state.flat_streak = 0
        self._save()

    def bump_flat(self) -> int:
        if self.state is None:
            return 0
        self.state.flat_streak = int(self.state.flat_streak or 0) + 1
        self._save()
        return self.state.flat_streak

    def reset_flat_streak(self) -> None:
        if self.state is None:
            return
        if int(self.state.flat_streak or 0) == 0:
            return
        self.state.flat_streak = 0
        self._save()

    def mark_scored(self, now: float) -> None:
        if self.state is None:
            return
        self.state.last_score_at = now
        self._save()

    def mark_cleared_legacy(self) -> None:
        if self.state is None:
            return
        self.state.cleared_legacy = True
        self._save()

    def pause(self, until: float) -> None:
        if self.state is None:
            return
        self.state.pause_until = until
        self._save()

    def close(self, *, coin: str, until: float) -> None:
        prev = self.state
        self.state = HftState(
            coin="",
            last_exit_coin=coin or (prev.coin if prev else ""),
            last_exit_until=until,
        )
        self._save()

    def active(self) -> HftState | None:
        if self.state is None or not self.state.coin:
            return None
        return self.state
