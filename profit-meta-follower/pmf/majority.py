"""Majority portfolio: top-N 7d ROI wallets, equal vote on what they hold.

No scalper/holder fill filter. Each snapped wallet casts one vote per
(coin, side) they currently hold. We take the majority side per coin,
keep the most popular coins, and split OUR_GROSS_MARGIN_PCT by hold count.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from .consensus import _clamp_leverage, in_scope, market_blocks_entry
from .types import MarketCtx, TargetPos, WalletSnapshot


def next_open_margin_usd(
    *,
    available: float,
    this_weight: float,
    rest_weight: float,
    planned_usd: float,
    buffer: float = 0.70,
) -> float:
    """Dollar margin for the next open from *current* free collateral.

    `rest_weight` is this coin plus every coin not opened yet. Last coin
    gets buffer × all remaining free so it fills instead of skipping.
    Earlier coins take their share of leftover, never the original equity %.
    """
    avail = max(0.0, float(available))
    planned = max(0.0, float(planned_usd))
    rest = max(1e-9, float(rest_weight))
    this = max(0.0, float(this_weight))
    buf = min(0.90, max(0.50, float(buffer)))
    share = this / rest
    fitted = avail * buf * share
    if share >= 0.999:
        return max(0.0, fitted)
    if planned <= 0:
        return max(0.0, fitted)
    return max(0.0, min(planned, fitted))



@dataclass
class HoldRow:
    coin: str
    side: str
    wallets: int
    hold_pct: float
    agreement: float
    long_n: int
    short_n: int
    median_leverage: int
    mean_leverage: float
    avg_conviction: float
    notional_usd: float
    skip: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _min_notional(cfg: Any) -> float:
    return float(getattr(cfg, "MAJORITY_MIN_NOTIONAL_USD", 50.0) or 0.0)


def tally_holds(
    snaps: list[WalletSnapshot],
    cfg: Any,
    *,
    now: float,
) -> tuple[list[HoldRow], dict[str, Any]]:
    """Count (coin, side) holdings. Denominator = successful snapshots."""
    stale_s = float(getattr(cfg, "STALE_SNAPSHOT_S", 1800.0) or 1800.0)
    min_ntl = _min_notional(cfg)
    ok: list[WalletSnapshot] = []
    empty = 0
    errors = 0
    stale = 0
    for s in snaps:
        if s.error:
            errors += 1
            continue
        if now - float(s.fetched_at or 0) > stale_s:
            stale += 1
            continue
        ok.append(s)
        if not s.positions:
            empty += 1

    n_ok = len(ok)
    long_n: dict[str, int] = defaultdict(int)
    short_n: dict[str, int] = defaultdict(int)
    long_lev: dict[str, list[int]] = defaultdict(list)
    short_lev: dict[str, list[int]] = defaultdict(list)
    long_conv: dict[str, list[float]] = defaultdict(list)
    short_conv: dict[str, list[float]] = defaultdict(list)
    long_ntl: dict[str, float] = defaultdict(float)
    short_ntl: dict[str, float] = defaultdict(float)

    for s in ok:
        seen: set[str] = set()
        for p in s.positions:
            if not in_scope(p.coin, cfg):
                continue
            if float(p.notional or 0) < min_ntl:
                continue
            if p.coin in seen:
                continue
            seen.add(p.coin)
            lev = max(1, int(p.leverage or 1))
            if p.side == "long":
                long_n[p.coin] += 1
                long_lev[p.coin].append(lev)
                long_conv[p.coin].append(abs(float(p.conviction or 0)))
                long_ntl[p.coin] += float(p.notional or 0)
            else:
                short_n[p.coin] += 1
                short_lev[p.coin].append(lev)
                short_conv[p.coin].append(abs(float(p.conviction or 0)))
                short_ntl[p.coin] += float(p.notional or 0)

    coins = set(long_n) | set(short_n)
    rows: list[HoldRow] = []
    for coin in coins:
        ln = int(long_n.get(coin, 0))
        sn = int(short_n.get(coin, 0))
        if ln >= sn and ln > 0:
            side = "long"
            n = ln
            levs = long_lev[coin]
            convs = long_conv[coin]
            ntl = long_ntl[coin]
        else:
            side = "short"
            n = sn
            levs = short_lev[coin]
            convs = short_conv[coin]
            ntl = short_ntl[coin]
        both = ln + sn
        hold_pct = (n / n_ok) if n_ok else 0.0
        agr = (n / both) if both else 0.0
        med = int(round(float(statistics.median(levs)))) if levs else 1
        mean_lev = float(sum(levs) / len(levs)) if levs else 1.0
        avg_c = float(sum(convs) / len(convs)) if convs else 0.0
        rows.append(
            HoldRow(
                coin=coin,
                side=side,
                wallets=n,
                hold_pct=hold_pct,
                agreement=agr,
                long_n=ln,
                short_n=sn,
                median_leverage=med,
                mean_leverage=mean_lev,
                avg_conviction=avg_c,
                notional_usd=ntl,
            )
        )
    rows.sort(key=lambda r: (-r.wallets, -r.hold_pct, r.coin))
    stats = {
        "snapped": len(snaps),
        "ok": n_ok,
        "empty": empty,
        "errors": errors,
        "stale": stale,
        "with_pos": n_ok - empty,
        "coins": len(rows),
    }
    return rows, stats


def _eligible(row: HoldRow, cfg: Any, markets: dict[str, MarketCtx], *, exit_band: bool) -> str:
    min_hold = float(getattr(cfg, "MAJORITY_MIN_HOLD_PCT", 0.05) or 0.0)
    exit_hold = float(getattr(cfg, "MAJORITY_EXIT_HOLD_PCT", 0.03) or 0.0)
    floor = exit_hold if exit_band else min_hold
    min_agr = float(getattr(cfg, "MAJORITY_MIN_SIDE_AGREEMENT", 0.55) or 0.0)
    if row.hold_pct + 1e-12 < floor:
        return "hold_pct"
    if row.agreement + 1e-12 < min_agr:
        return "agreement"
    ctx = markets.get(row.coin)
    if ctx is not None:
        why = market_blocks_entry(row.coin, row.side, ctx, cfg)
        if why:
            return why
    return ""


def majority_max_pairs(cfg: Any) -> int:
    if bool(getattr(cfg, "MAJORITY_SINGLE_PAIR", False)):
        return 1
    return max(1, int(getattr(cfg, "MAX_COINS_IN_BOOK", 4) or 4))


def majority_leverage(cfg: Any, raw: float) -> int:
    """Wallet mean/median leverage, divided by MAJORITY_LEVERAGE_DIV, then clamped.

    Hyperliquid only accepts integer leverage, so 9 / 2 → 4.
    """
    div = float(getattr(cfg, "MAJORITY_LEVERAGE_DIV", 1.0) or 1.0)
    if div < 1.0:
        div = 1.0
    return _clamp_leverage(cfg, float(raw) / div)


def majority_gross_pct(cfg: Any) -> float:
    if bool(getattr(cfg, "MAJORITY_SINGLE_PAIR", False)):
        extra = float(getattr(cfg, "MAJORITY_SINGLE_GROSS_PCT", 0) or 0)
        if extra > 0:
            return extra
    return float(getattr(cfg, "OUR_GROSS_MARGIN_PCT", 95.0) or 95.0)


def hold_key(coin: str, side: str) -> str:
    return f"{str(coin)}|{str(side or '').strip().lower()}"


def board_ranks(rows: list[HoldRow]) -> dict[str, int]:
    return {hold_key(r.coin, r.side): i for i, r in enumerate(rows, start=1)}


def majority_enter_top(cfg: Any) -> int:
    return max(1, int(getattr(cfg, "MAJORITY_ENTER_TOP", 5) or 5))


def majority_rank_watch(cfg: Any) -> int:
    top = majority_enter_top(cfg)
    watch = int(getattr(cfg, "MAJORITY_RANK_WATCH", 0) or 0)
    return max(top + 1, watch if watch > 0 else 16)


def _targets_from_picked(picked: list[HoldRow], cfg: Any) -> list[TargetPos]:
    gross = majority_gross_pct(cfg)
    max_share = float(getattr(cfg, "MAJORITY_MAX_PAIR_SHARE", 0.70) or 0.70)
    votes = [max(1, r.wallets) for r in picked]
    total = float(sum(votes)) or 1.0
    raw_w = [v / total for v in votes]
    if max_share > 0 and raw_w:
        capped = [min(w, max_share) for w in raw_w]
        s = sum(capped) or 1.0
        weights = [w / s for w in capped]
    else:
        weights = raw_w
    targets: list[TargetPos] = []
    for row, w in zip(picked, weights):
        margin = gross * w
        lev = majority_leverage(cfg, row.mean_leverage or float(row.median_leverage))
        if margin * lev < 0.5:
            continue
        targets.append(
            TargetPos(
                coin=row.coin,
                side=row.side,
                leverage=lev,
                margin_pct=margin,
                conviction=row.avg_conviction if row.side == "long" else -row.avg_conviction,
            )
        )
    return targets


def plan_rank_targets(
    annotated: list[HoldRow],
    cfg: Any,
    *,
    prev_ranks: dict[str, int],
    held: dict[str, str],
) -> tuple[list[TargetPos], dict[str, Any]]:
    """Enter only on rank-up into the top N; exit only on rank-down.

    First snapshot with empty prev_ranks seeds ranks and opens nothing.
    """
    enter_top = majority_enter_top(cfg)
    watch_n = majority_rank_watch(cfg)
    current = board_ranks(annotated)
    by_key = {hold_key(r.coin, r.side): r for r in annotated}
    events: list[str] = []
    if not prev_ranks:
        persist = {k: v for k, v in current.items() if v <= watch_n}
        return [], {
            "seed": True,
            "ranks": current,
            "persist_ranks": persist,
            "enter_top": enter_top,
            "watch": watch_n,
            "events": ["seed ranks — no entries until a name moves up"],
            "picked": [],
        }

    keep_rows: list[HoldRow] = []
    for coin, side in (held or {}).items():
        k = hold_key(coin, side)
        row = by_key.get(k)
        cur = current.get(k)
        prev = int(prev_ranks[k]) if k in prev_ranks else None
        if row is None or cur is None:
            events.append(f"rank-drop EXIT {coin} {side} #{prev or '?'}→off")
            continue
        if prev is not None and cur > prev:
            events.append(f"rank-drop EXIT {coin} {side} #{prev}→#{cur}")
            continue
        keep_rows.append(row)
        if prev is None:
            events.append(f"hold {coin} {side} #{cur}")
        elif cur < prev:
            events.append(f"hold {coin} {side} #{prev}→#{cur} (up)")
        else:
            events.append(f"hold {coin} {side} #{cur}")

    keep_keys = {hold_key(r.coin, r.side) for r in keep_rows}
    enters: list[HoldRow] = []
    for row in annotated:
        if row.skip:
            continue
        k = hold_key(row.coin, row.side)
        rank = current.get(k)
        if rank is None or rank > enter_top:
            continue
        if k in keep_keys:
            continue
        old = int(prev_ranks[k]) if k in prev_ranks else watch_n + 1
        if rank < old:
            enters.append(row)
            old_txt = f"#{old}" if k in prev_ranks else "unranked"
            events.append(
                f"rank-up ENTER {row.coin} {row.side} {old_txt}→#{rank}"
            )

    slots = max(0, enter_top - len(keep_rows))
    enters.sort(key=lambda r: current.get(hold_key(r.coin, r.side), 10**9))
    skipped_cap = enters[slots:]
    enters = enters[:slots]
    for row in skipped_cap:
        events.append(
            f"rank-up skip {row.coin} {row.side} — already {enter_top} names"
        )

    picked = keep_rows + enters
    targets = _targets_from_picked(picked, cfg)
    persist = {k: v for k, v in current.items() if v <= watch_n}
    for row in keep_rows:
        persist[hold_key(row.coin, row.side)] = current[hold_key(row.coin, row.side)]
    return targets, {
        "seed": False,
        "ranks": current,
        "persist_ranks": persist,
        "enter_top": enter_top,
        "watch": watch_n,
        "events": events,
        "picked": [
            {
                "coin": t.coin,
                "side": t.side,
                "wallets": next((r.wallets for r in picked if r.coin == t.coin), 0),
                "hold_pct": round(
                    next((r.hold_pct for r in picked if r.coin == t.coin), 0.0) * 100.0,
                    2,
                ),
                "margin_pct": round(t.margin_pct, 3),
                "lev": t.leverage,
            }
            for t in targets
        ],
        "gross": majority_gross_pct(cfg),
        "max_pairs": enter_top,
        "eligible": sum(1 for r in annotated if not r.skip),
        "sticky": False,
        "single": False,
        "rank_entry": True,
    }


def pick_majority_targets(
    rows: list[HoldRow],
    cfg: Any,
    *,
    managed: set[str] | None = None,
    markets: dict[str, MarketCtx] | None = None,
    prev_ranks: dict[str, int] | None = None,
    held: dict[str, str] | None = None,
) -> tuple[list[TargetPos], list[HoldRow], dict[str, Any]]:
    """Top MAX_COINS_IN_BOOK coins, margin ∝ wallet-count, sum ≈ OUR_GROSS_MARGIN_PCT.

    MAJORITY_SINGLE_PAIR=True → only the #1 most-held eligible pair, full gross.
    MAJORITY_RANK_ENTRY=True + prev_ranks dict → enter on rank-up, exit on rank-down.
    """
    markets = markets or {}
    managed = {str(c) for c in (managed or ())}
    single = bool(getattr(cfg, "MAJORITY_SINGLE_PAIR", False))
    max_n = majority_max_pairs(cfg)
    sticky = bool(getattr(cfg, "MAJORITY_STICKY", True)) and not single

    annotated: list[HoldRow] = []
    for row in rows:
        skip = _eligible(row, cfg, markets, exit_band=False)
        annotated.append(
            HoldRow(
                coin=row.coin,
                side=row.side,
                wallets=row.wallets,
                hold_pct=row.hold_pct,
                agreement=row.agreement,
                long_n=row.long_n,
                short_n=row.short_n,
                median_leverage=row.median_leverage,
                mean_leverage=row.mean_leverage,
                avg_conviction=row.avg_conviction,
                notional_usd=row.notional_usd,
                skip=skip,
            )
        )

    rank_on = bool(getattr(cfg, "MAJORITY_RANK_ENTRY", False)) and not single
    if rank_on and prev_ranks is not None:
        targets, rank_meta = plan_rank_targets(
            annotated, cfg, prev_ranks=prev_ranks, held=held or {}
        )
        rank_meta["ranks"] = board_ranks(annotated)
        return targets, annotated, rank_meta

    fresh = [r for r in annotated if not r.skip]
    picked: list[HoldRow] = []
    if sticky and managed:
        kept: list[HoldRow] = []
        for row in annotated:
            if row.coin not in managed:
                continue
            why = _eligible(row, cfg, markets, exit_band=True)
            if why:
                continue
            kept.append(row)
        kept.sort(key=lambda r: (-r.wallets, r.coin))
        picked.extend(kept[:max_n])
    for row in fresh:
        if len(picked) >= max_n:
            break
        if any(p.coin == row.coin for p in picked):
            continue
        picked.append(row)

    targets = _targets_from_picked(picked, cfg)
    meta = {
        "picked": [
            {
                "coin": t.coin,
                "side": t.side,
                "wallets": next((r.wallets for r in picked if r.coin == t.coin), 0),
                "hold_pct": round(
                    next((r.hold_pct for r in picked if r.coin == t.coin), 0.0) * 100.0,
                    2,
                ),
                "margin_pct": round(t.margin_pct, 3),
                "lev": t.leverage,
            }
            for t in targets
        ],
        "gross": majority_gross_pct(cfg),
        "max_pairs": max_n,
        "eligible": len(fresh),
        "sticky": sticky,
        "single": single,
        "ranks": board_ranks(annotated),
    }
    return targets, annotated, meta


def compact_hold_board(rows: list[HoldRow], *, n: int = 16) -> str:
    parts = []
    for i, r in enumerate(rows[: max(1, n)], start=1):
        tag = r.skip or "ok"
        parts.append(
            "#%s %s %s %s/%s (%.1f%%) agr=%.0f%% lev=%s %s"
            % (
                i,
                r.coin,
                r.side,
                r.wallets,
                r.long_n + r.short_n,
                r.hold_pct * 100.0,
                r.agreement * 100.0,
                r.median_leverage,
                tag,
            )
        )
    return " | ".join(parts) if parts else "-"
