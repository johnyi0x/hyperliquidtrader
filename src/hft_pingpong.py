"""
Maker ping-pong for Hyperliquid perps (Railway-safe, REST poll).

Picks a choppy wide-enough book, quotes one clip post-only on both sides when
flat, skews to flatten when filled, and market-exits on trend / box break /
inventory timeout so size is not left behind when price walks away.
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
    up_frac: float
    burst: bool  # last bars one-way
    bar_t: int
    score: float


@dataclass
class HftState:
    coin: str
    opened_at: float = 0.0
    last_fill_at: float = 0.0
    last_score_at: float = 0.0
    last_exit_coin: str = ""
    last_exit_until: float = 0.0
    cleared_legacy: bool = False
    pause_until: float = 0.0


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
    last3 = closes[-4:]
    burst = False
    if len(last3) >= 4:
        same_up = all(last3[i] > last3[i - 1] for i in range(1, 4))
        same_dn = all(last3[i] < last3[i - 1] for i in range(1, 4))
        move_bps = abs(last3[-1] - last3[0]) / last3[0] * 10_000.0
        burst = (same_up or same_dn) and move_bps >= max(12.0, 0.9 * atr_bps)
    # Prefer lots of path, two-sided tape, usable range; penalize trend.
    balance = 1.0 - abs(up_frac - 0.5) * 2.0
    score = path_over_net * max(0.05, balance) * min(range_bps, 80.0) * (1.0 - er)
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
        up_frac=up_frac,
        burst=burst,
        bar_t=int(tail[-1]["t"]),
        score=float(score),
    )


def pick_chop(
    snaps: list[ChopSnap],
    *,
    max_er: float,
    skip_coin: str | None = None,
    skip_until: float = 0.0,
    now: float = 0.0,
) -> ChopSnap | None:
    eligible: list[ChopSnap] = []
    for s in snaps:
        if skip_coin and s.coin == skip_coin and now < skip_until:
            continue
        if s.er > max_er + 1e-12:
            continue
        if s.burst:
            continue
        if s.range_bps < 8.0:
            continue
        if s.up_frac < 0.22 or s.up_frac > 0.78:
            continue
        eligible.append(s)
    if not eligible:
        return None
    best = eligible[0]
    for s in eligible[1:]:
        if s.score > best.score + 1e-12:
            best = s
        elif abs(s.score - best.score) <= 1e-12 and s.coin < best.coin:
            best = s
    return best


def vol_scale(atr_bps: float) -> float:
    return max(0.55, min(2.6, (max(4.0, atr_bps) / 16.0)))


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
) -> HftDecision:
    del clip_sz
    vs = vol_scale(chop.atr_bps if chop else 16.0)
    timeout = max(8.0, min(48.0, base_timeout_s / vs))
    in_pos = bool(side) and size > 1e-12

    if book.spread_bps > max_spread_bps:
        if in_pos:
            return HftDecision(
                "spread_toxic", None, False, False, False, False, timeout, vs, "spread wide"
            )
        return HftDecision(
            None, "spread_wide", False, False, False, False, timeout, vs, "spread wide"
        )

    if (not in_pos) and book.spread_bps + 1e-12 < min_spread_bps:
        return HftDecision(
            None, "spread_tight", False, False, False, False, timeout, vs, "spread tight"
        )

    trending = bool(chop and (chop.er > max_er or chop.burst))
    if trending:
        if in_pos:
            return HftDecision(
                "trend", None, False, False, False, False, timeout, vs, "trend flatten"
            )
        return HftDecision(
            None, "trend", False, False, False, False, timeout, vs, "trend pause"
        )

    if chop is not None and chop.box_high > chop.box_low:
        span = chop.box_high - chop.box_low
        pad = book.mid * (box_break_bps / 10_000.0)
        if book.mid > chop.box_high + pad or book.mid < chop.box_low - pad:
            if in_pos:
                return HftDecision(
                    "box_break", None, False, False, False, False, timeout, vs, "box break"
                )
            return HftDecision(
                None, "box_break", False, False, False, False, timeout, vs, "outside box"
            )
        loc = (book.mid - chop.box_low) / span
    else:
        loc = 0.5

    if in_pos and last_fill_at > 0 and now - last_fill_at >= timeout:
        return HftDecision(
            "inventory_timeout",
            None,
            False,
            False,
            False,
            False,
            timeout,
            vs,
            "timeout",
        )

    if in_pos and entry_px > 0:
        stop_bps = max(14.0, 2.2 * book.spread_bps, 0.55 * (chop.atr_bps if chop else 20.0))
        if side == "long" and (entry_px - book.mid) / entry_px * 10_000.0 >= stop_bps:
            return HftDecision(
                "inv_stop", None, False, False, False, False, timeout, vs, "long stop"
            )
        if side == "short" and (book.mid - entry_px) / entry_px * 10_000.0 >= stop_bps:
            return HftDecision(
                "inv_stop", None, False, False, False, False, timeout, vs, "short stop"
            )
        # Winner: don't sit long into a lift or short into a dump waiting on a stale quote.
        take_bps = max(book.spread_bps * 0.85, 3.0)
        if side == "long" and (book.mid - entry_px) / entry_px * 10_000.0 >= take_bps:
            return HftDecision(
                None,
                None,
                False,
                True,
                False,
                True,
                timeout,
                vs,
                "exit long",
            )
        if side == "short" and (entry_px - book.mid) / entry_px * 10_000.0 >= take_bps:
            return HftDecision(
                None,
                None,
                True,
                False,
                True,
                False,
                timeout,
                vs,
                "exit short",
            )

    if in_pos:
        # Only the flattening side. Never add when already in.
        if side == "long":
            return HftDecision(
                None, None, False, True, False, True, timeout, vs, "reduce long"
            )
        return HftDecision(
            None, None, True, False, True, False, timeout, vs, "reduce short"
        )

    # Flat: two-sided unless book/box says one side is toxic.
    quote_bid = loc < 0.90 and book.imbalance < 0.62
    quote_ask = loc > 0.10 and book.imbalance > -0.62
    if loc > 0.88:
        quote_bid = False
    if loc < 0.12:
        quote_ask = False
    if not quote_bid and not quote_ask:
        return HftDecision(
            None, "edge", False, False, False, False, timeout, vs, "no safe side"
        )
    return HftDecision(
        None,
        None,
        quote_bid,
        quote_ask,
        False,
        False,
        timeout,
        vs,
        "two-sided" if quote_bid and quote_ask else "one-sided",
    )


def quote_px_ok(existing_px: float, target_px: float, tick: float, mid: float) -> bool:
    if existing_px <= 0 or target_px <= 0:
        return False
    tol = max(tick * 0.51, mid * 1.2e-4)
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
                last_score_at=float(raw.get("last_score_at", 0) or 0),
                last_exit_coin=str(raw.get("last_exit_coin", "") or ""),
                last_exit_until=float(raw.get("last_exit_until", 0) or 0),
                cleared_legacy=bool(raw.get("cleared_legacy", False)),
                pause_until=float(raw.get("pause_until", 0) or 0),
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

    def touch_fill(self, *, now: float) -> None:
        if self.state is None:
            return
        if self.state.opened_at <= 0:
            self.state.opened_at = now
        self.state.last_fill_at = now
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
