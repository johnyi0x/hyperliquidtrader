"""Score leaderboard wallets for copy-trade mode (activity band + win rate + PnL)."""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from .qualify import Qualifier, _fill_closed_pnl
from .types import QualifiedWallet, fnum


@dataclass
class FillStats:
    n_fills: int = 0
    median_gap_s: float = 0.0
    fills_per_day: float = 0.0
    win_rate: float = 0.0
    closed_pnl: float = 0.0
    fees: float = 0.0
    wins: int = 0
    losses: int = 0
    last_fill_ms: int = 0


@dataclass
class CopyLeader:
    address: str
    score: float
    rank_roi: float
    rank_pnl: float
    account_value: float
    recent: FillStats = field(default_factory=FillStats)
    history: FillStats = field(default_factory=FillStats)
    reasons: list[str] = field(default_factory=list)


def _fills_in_window(fills: list[Any], *, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in fills or []:
        if not isinstance(item, dict):
            continue
        ts = int(item.get("time") or 0)
        if ts <= 0 or ts < start_ms or ts > end_ms:
            continue
        out.append(item)
    return out


def analyze_fills(fills: list[Any], *, now_ms: int, lookback_days: float) -> FillStats:
    """Fill tape stats for a time window ending at now_ms."""
    span_ms = int(max(lookback_days, 0.25) * 86400_000)
    start_ms = now_ms - span_ms
    window = _fills_in_window(fills, start_ms=start_ms, end_ms=now_ms)
    times: list[int] = []
    wins = losses = 0
    closed_pnl = fees = 0.0
    for f in window:
        ts = int(f.get("time") or 0)
        if ts:
            times.append(ts)
        cp = fnum(f.get("closedPnl"))
        closed_pnl += cp
        fees += abs(fnum(f.get("fee")))
        if cp > 0:
            wins += 1
        elif cp < 0:
            losses += 1
    n = len(times)
    times.sort()
    gaps = [(times[i] - times[i - 1]) / 1000.0 for i in range(1, n)]
    median_gap = float(statistics.median(gaps)) if gaps else lookback_days * 86400.0
    per_day = n / max(lookback_days, 0.25)
    closed_n = wins + losses
    win_rate = (wins / closed_n) if closed_n > 0 else 0.5
    return FillStats(
        n_fills=n,
        median_gap_s=median_gap,
        fills_per_day=per_day,
        win_rate=win_rate,
        closed_pnl=closed_pnl,
        fees=fees,
        wins=wins,
        losses=losses,
        last_fill_ms=times[-1] if times else 0,
    )


def passes_copy_filters(recent: FillStats, history: FillStats, cfg: Any) -> tuple[bool, str]:
    min_f = int(getattr(cfg, "COPY_MIN_FILLS", 4) or 4)
    max_f = int(getattr(cfg, "COPY_MAX_FILLS", 35) or 35)
    min_gap = float(getattr(cfg, "COPY_MIN_MEDIAN_GAP_S", 900.0) or 900.0)
    max_gap = float(getattr(cfg, "COPY_MAX_MEDIAN_GAP_S", 28800.0) or 28800.0)
    min_wr = float(getattr(cfg, "COPY_MIN_WIN_RATE", 0.48) or 0.48)
    min_hist_wr = float(getattr(cfg, "COPY_MIN_HIST_WIN_RATE", 0.42) or 0.42)
    min_pnl = float(getattr(cfg, "COPY_MIN_RECENT_PNL", 0.0) or 0.0)
    min_hist_pnl = float(getattr(cfg, "COPY_MIN_HIST_PNL", -50.0) or -50.0)

    if recent.n_fills < min_f:
        return False, f"too_few_fills={recent.n_fills}"
    if recent.n_fills > max_f:
        return False, f"too_many_fills={recent.n_fills}"
    if recent.median_gap_s < min_gap:
        return False, f"scalpy gap={recent.median_gap_s:.0f}s"
    if recent.median_gap_s > max_gap:
        return False, f"dormant gap={recent.median_gap_s:.0f}s"
    if recent.win_rate < min_wr:
        return False, f"low_wr={recent.win_rate:.0%}"
    if history.win_rate < min_hist_wr and history.wins + history.losses >= 6:
        return False, f"hist_wr={history.win_rate:.0%}"
    if recent.closed_pnl < min_pnl:
        return False, f"recent_pnl=${recent.closed_pnl:.0f}"
    if history.closed_pnl < min_hist_pnl:
        return False, f"hist_pnl=${history.closed_pnl:.0f}"
    return True, "ok"


def score_copy_wallet(
    wallet: QualifiedWallet,
    recent: FillStats,
    history: FillStats,
    cfg: Any,
) -> float:
    """Higher = better copy candidate."""
    roi = max(wallet.rank_roi, -0.5) * 100.0
    wr = recent.win_rate * 40.0 + history.win_rate * 20.0
    pnl = math.log10(max(recent.closed_pnl, 1.0) + 10.0) * 8.0
    if recent.closed_pnl < 0:
        pnl -= 15.0
    ideal = float(getattr(cfg, "COPY_IDEAL_GAP_S", 7200.0) or 7200.0)
    gap_pen = abs(math.log(max(recent.median_gap_s, 60.0) / max(ideal, 60.0)))
    activity = max(0.0, 12.0 - gap_pen * 6.0)
    return roi * 0.35 + wr + pnl + activity


def pick_copy_leaders(
    pool: list[QualifiedWallet],
    qualifier: Qualifier,
    cfg: Any,
    *,
    logger: logging.Logger | None = None,
) -> list[CopyLeader]:
    """Scan shortlist, fetch fills once per wallet, return top COPY_TOP_N."""
    log = logger or qualifier.log
    want = max(1, int(getattr(cfg, "COPY_TOP_N", 3) or 3))
    scan = max(want, int(getattr(cfg, "COPY_CANDIDATE_SCAN", 120) or 120))
    recent_d = float(getattr(cfg, "COPY_LOOKBACK_DAYS", 7.0) or 7.0)
    hist_d = float(getattr(cfg, "COPY_HISTORY_DAYS", 30.0) or 30.0)
    now_ms = int(time.time() * 1000)
    start_hist = now_ms - int(hist_d * 86400_000)

    ranked: list[CopyLeader] = []
    rejected = 0
    for i, w in enumerate(pool[:scan]):
        fills = qualifier._recent_fills(w.address, start_hist)
        if fills is None:
            rejected += 1
            continue
        if i > 0 and i % 5 == 0:
            time.sleep(0.2)
        recent = analyze_fills(fills, now_ms=now_ms, lookback_days=recent_d)
        history = analyze_fills(fills, now_ms=now_ms, lookback_days=hist_d)
        ok, why = passes_copy_filters(recent, history, cfg)
        if not ok:
            rejected += 1
            if i < 15:
                log.debug("Copy skip %s roi=%.1f%% — %s", w.address[:10], w.rank_roi * 100, why)
            continue
        sc = score_copy_wallet(w, recent, history, cfg)
        ranked.append(
            CopyLeader(
                address=w.address.lower(),
                score=sc,
                rank_roi=w.rank_roi,
                rank_pnl=w.rank_pnl,
                account_value=w.account_value,
                recent=recent,
                history=history,
                reasons=[why, f"fills={recent.n_fills} gap={recent.median_gap_s:.0f}s wr={recent.win_rate:.0%}"],
            )
        )
    ranked.sort(key=lambda x: x.score, reverse=True)
    leaders = ranked[:want]
    log.info(
        "Copy scan: %s candidates, %s rejected, picked %s leaders",
        min(scan, len(pool)),
        rejected,
        len(leaders),
    )
    for j, ld in enumerate(leaders, 1):
        log.info(
            "  copy #%s %s score=%.1f roi=%.1f%% wr=%.0f%% pnl=$%.0f gap=%.0fh",
            j,
            ld.address[:10],
            ld.score,
            ld.rank_roi * 100,
            ld.recent.win_rate * 100,
            ld.recent.closed_pnl,
            ld.recent.median_gap_s / 3600.0,
        )
    return leaders


def leaders_to_state(leaders: list[CopyLeader]) -> list[dict[str, Any]]:
    return [
        {
            "address": ld.address,
            "score": round(ld.score, 4),
            "rank_roi": ld.rank_roi,
            "rank_pnl": ld.rank_pnl,
            "account_value": ld.account_value,
            "recent_wr": ld.recent.win_rate,
            "recent_pnl": ld.recent.closed_pnl,
            "median_gap_s": ld.recent.median_gap_s,
            "reasons": ld.reasons,
        }
        for ld in leaders
    ]


def leaders_from_state(raw: list[Any]) -> list[CopyLeader]:
    out: list[CopyLeader] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        addr = str(item.get("address") or "").lower()
        if not addr.startswith("0x"):
            continue
        recent = FillStats(
            win_rate=float(item.get("recent_wr") or 0),
            closed_pnl=float(item.get("recent_pnl") or 0),
            median_gap_s=float(item.get("median_gap_s") or 0),
        )
        out.append(
            CopyLeader(
                address=addr,
                score=float(item.get("score") or 0),
                rank_roi=float(item.get("rank_roi") or 0),
                rank_pnl=float(item.get("rank_pnl") or 0),
                account_value=float(item.get("account_value") or 0),
                recent=recent,
                reasons=list(item.get("reasons") or []),
            )
        )
    return out


def leaders_to_basket(leaders: list[CopyLeader]) -> list[QualifiedWallet]:
    return [
        QualifiedWallet(
            address=ld.address,
            account_value=ld.account_value,
            rank_pnl=ld.rank_pnl,
            rank_roi=ld.rank_roi,
            rank_volume=0.0,
            confirm_pnl=0.0,
            score=ld.score,
            reasons=ld.reasons,
        )
        for ld in leaders
    ]


def quick_fill_stats_from_list(fills: list[Any], *, now_ms: int, lookback_days: float) -> FillStats:
    """Public wrapper for backtest."""
    return analyze_fills(fills, now_ms=now_ms, lookback_days=lookback_days)


def closed_pnl_from_fills(fills: list[Any]) -> tuple[float, float, int]:
    return _fill_closed_pnl(fills)
