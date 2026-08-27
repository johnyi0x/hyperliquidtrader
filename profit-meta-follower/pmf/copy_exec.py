"""Mirror top copy-leader books into our target positions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .consensus import _clamp_leverage, in_scope
from .copy_score import CopyLeader
from .types import TargetPos, WalletSnapshot


def _leader_budget_pct(cfg: Any, n_leaders: int) -> float:
    gross = float(getattr(cfg, "OUR_GROSS_MARGIN_PCT", 90.0) or 90.0)
    return gross / max(1, n_leaders)


def _pos_margin_pct(pos, equity: float, budget_pct: float) -> float:
    if equity <= 0:
        return 0.0
    notional = abs(float(pos.notional or 0))
    lev = max(1, int(pos.leverage or 1))
    margin_frac = notional / max(equity, 1.0) / lev
    return min(budget_pct, margin_frac * 100.0)


def copy_targets_from_leaders(
    leaders: list[CopyLeader],
    snaps: list[WalletSnapshot],
    cfg: Any,
    *,
    now: float,
) -> list[TargetPos]:
    """Aggregate leader positions; resolve conflicts by weighted score."""
    if not leaders:
        return []
    stale_s = float(getattr(cfg, "STALE_SNAPSHOT_S", 480.0) or 480.0)
    by_addr = {s.address.lower(): s for s in snaps}
    budget = _leader_budget_pct(cfg, len(leaders))
    cap_copy = float(getattr(cfg, "COPY_MARGIN_CAP_PCT", 0) or 0)
    if cap_copy <= 0:
        cap_copy = float(getattr(cfg, "COPY_MARGIN_CAP_PCT", 100.0) or 100.0)
    per_coin_cap = float(getattr(cfg, "OUR_GROSS_MARGIN_PCT", 90.0) or 90.0) * (
        float(getattr(cfg, "MAX_MARGIN_PER_COIN_PCT", 33.33) or 33.33) / 100.0
    )

    long_w: dict[str, float] = defaultdict(float)
    short_w: dict[str, float] = defaultdict(float)
    long_margin: dict[str, float] = defaultdict(float)
    short_margin: dict[str, float] = defaultdict(float)
    long_lev: dict[str, list[float]] = defaultdict(list)
    short_lev: dict[str, list[float]] = defaultdict(list)

    for ld in leaders:
        snap = by_addr.get(ld.address.lower())
        if snap is None or (now - snap.fetched_at) > stale_s:
            continue
        positions = [p for p in snap.positions if in_scope(p.coin, cfg)]
        if not positions:
            continue
        per_pos = budget / len(positions)
        weight = max(0.1, ld.score)
        for pos in positions:
            coin = pos.coin
            margin = _pos_margin_pct(pos, snap.account_value, per_pos)
            if cap_copy > 0:
                margin = min(margin, cap_copy)
            if margin <= 0:
                continue
            lev = float(max(1, int(pos.leverage or 1)))
            if pos.side == "long":
                long_w[coin] += weight
                long_margin[coin] += margin * weight
                long_lev[coin].append(lev)
            else:
                short_w[coin] += weight
                short_margin[coin] += margin * weight
                short_lev[coin].append(lev)

    candidates: list[tuple[float, TargetPos]] = []
    coins = set(long_w) | set(short_w)
    for coin in coins:
        lw = long_w.get(coin, 0.0)
        sw = short_w.get(coin, 0.0)
        if lw <= 0 and sw <= 0:
            continue
        if lw >= sw:
            side = "long"
            margin = long_margin[coin] / max(lw, 1e-9)
            levs = long_lev[coin]
        else:
            side = "short"
            margin = short_margin[coin] / max(sw, 1e-9)
            levs = short_lev[coin]
        margin = min(margin, per_coin_cap)
        if margin * (sum(levs) / max(len(levs), 1)) < 0.5:
            continue
        raw_lev = sum(levs) / max(len(levs), 1)
        candidates.append(
            (
                margin * math.sqrt(max(lw, sw)),
                TargetPos(
                    coin=coin,
                    side=side,
                    leverage=_clamp_leverage(cfg, raw_lev),
                    margin_pct=margin,
                    conviction=1.0 if side == "long" else -1.0,
                ),
            )
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    max_pos = max(1, int(getattr(cfg, "COPY_MAX_POSITIONS", 3) or 3))
    return [t for _w, t in candidates[:max_pos]]


def min_fresh_copy_leaders(cfg: Any, n_leaders: int) -> int:
    pct = float(getattr(cfg, "COPY_MIN_FRESH_LEADERS_PCT", 0.67) or 0.67)
    if pct > 0:
        return max(1, int(math.ceil(max(1, n_leaders) * pct)))
    return max(1, n_leaders)
