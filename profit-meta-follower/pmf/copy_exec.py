"""Mirror top copy-leader books into our target positions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .consensus import _clamp_leverage, in_scope
from .copy_score import CopyLeader
from .types import TargetPos, WalletSnapshot


def copy_leader_budget_pct(cfg: Any, n_leaders: int) -> float:
    gross = float(getattr(cfg, "OUR_GROSS_MARGIN_PCT", 90.0) or 90.0)
    return gross / max(1, n_leaders)


def _per_coin_margin_cap(cfg: Any) -> float:
    gross = float(getattr(cfg, "OUR_GROSS_MARGIN_PCT", 90.0) or 90.0)
    cap_frac = float(getattr(cfg, "MAX_MARGIN_PER_COIN_PCT", 33.33) or 33.33) / 100.0
    return gross * cap_frac


def fit_copy_targets_to_budget(
    targets: list[TargetPos],
    cfg: Any,
    *,
    n_leaders: int,
) -> list[TargetPos]:
    """Split the full copy budget evenly across targets so margin sums to ~gross %."""
    if not targets:
        return []
    budget = copy_leader_budget_pct(cfg, n_leaders)
    per_coin_cap = _per_coin_margin_cap(cfg)
    share = budget / len(targets)
    # With 1–2 targets the default per-coin cap (30% of equity) is too tight for copy mode.
    per_target = share if len(targets) <= 2 else min(share, per_coin_cap)
    out: list[TargetPos] = []
    for t in targets:
        out.append(
            TargetPos(
                coin=t.coin,
                side=t.side,
                leverage=t.leverage,
                margin_pct=per_target,
                conviction=1.0 if t.side == "long" else -1.0,
            )
        )
    return out


def redistribute_unfilled_copy_targets(
    targets: list[TargetPos],
    opened_coins: set[str],
    cfg: Any,
    *,
    n_leaders: int,
) -> list[TargetPos]:
    """Move margin from targets we could not open onto positions we did open."""
    if not targets or not opened_coins:
        return targets
    missing = [t for t in targets if t.coin not in opened_coins]
    filled = [t for t in targets if t.coin in opened_coins]
    if not missing or not filled:
        return targets
    budget = copy_leader_budget_pct(cfg, n_leaders)
    per_coin_cap = _per_coin_margin_cap(cfg)
    freed = sum(t.margin_pct for t in missing)
    if freed <= 0:
        return fit_copy_targets_to_budget(filled, cfg, n_leaders=n_leaders)
    extra = freed / len(filled)
    out: list[TargetPos] = []
    for t in filled:
        share = t.margin_pct + extra
        margin = share if len(filled) <= 2 else min(share, per_coin_cap)
        out.append(
            TargetPos(
                coin=t.coin,
                side=t.side,
                leverage=t.leverage,
                margin_pct=margin,
                conviction=1.0 if t.side == "long" else -1.0,
            )
        )
    return out


def _leader_budget_pct(cfg: Any, n_leaders: int) -> float:
    return copy_leader_budget_pct(cfg, n_leaders)


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
    reverse: bool = False,
) -> list[TargetPos]:
    """Aggregate leader positions; resolve conflicts by weighted score.

    reverse=True → opposite side of each mirrored position (copy_reverse mode).
    """
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
            # Size from OUR budget slot — do not shrink to a whale's tiny % of their equity.
            margin = min(per_pos, per_coin_cap)
            if cap_copy > 0:
                margin = min(margin, cap_copy)
            if margin <= 0:
                continue
            lev = float(max(1, int(pos.leverage or 1)))
            side = pos.side
            if reverse:
                side = "short" if side == "long" else "long"
            if side == "long":
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
    max_pos = max(1, int(getattr(cfg, "COPY_MAX_POSITIONS", 5) or 5))
    picked = [t for _w, t in candidates[:max_pos]]
    return fit_copy_targets_to_budget(picked, cfg, n_leaders=len(leaders))


def min_fresh_copy_leaders(cfg: Any, n_leaders: int) -> int:
    pct = float(getattr(cfg, "COPY_MIN_FRESH_LEADERS_PCT", 0.67) or 0.67)
    if pct > 0:
        return max(1, int(math.ceil(max(1, n_leaders) * pct)))
    return max(1, n_leaders)
