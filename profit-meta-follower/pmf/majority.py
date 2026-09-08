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


def pick_majority_targets(
    rows: list[HoldRow],
    cfg: Any,
    *,
    managed: set[str] | None = None,
    markets: dict[str, MarketCtx] | None = None,
) -> tuple[list[TargetPos], list[HoldRow], dict[str, Any]]:
    """Top MAX_COINS_IN_BOOK coins, margin ∝ wallet-count, sum ≈ OUR_GROSS_MARGIN_PCT."""
    markets = markets or {}
    managed = {str(c) for c in (managed or ())}
    max_n = max(1, int(getattr(cfg, "MAX_COINS_IN_BOOK", 4) or 4))
    gross = float(getattr(cfg, "OUR_GROSS_MARGIN_PCT", 95.0) or 95.0)
    max_share = float(getattr(cfg, "MAJORITY_MAX_PAIR_SHARE", 0.70) or 0.70)
    sticky = bool(getattr(cfg, "MAJORITY_STICKY", True))

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
        lev = _clamp_leverage(cfg, row.mean_leverage or float(row.median_leverage))
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
        "gross": gross,
        "max_pairs": max_n,
        "eligible": len(fresh),
        "sticky": sticky,
    }
    return targets, annotated, meta


def compact_hold_board(rows: list[HoldRow], *, n: int = 12) -> str:
    parts = []
    for r in rows[: max(1, n)]:
        tag = r.skip or "ok"
        parts.append(
            "%s %s %s/%s (%.1f%%) agr=%.0f%% lev=%s %s"
            % (
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
