"""Score leaderboard wallets for copy-trade mode (activity band + consistency)."""

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
    gross_win: float = 0.0
    gross_loss: float = 0.0
    last_fill_ms: int = 0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss <= 1e-9:
            return 5.0 if self.gross_win > 0 else 0.0
        return self.gross_win / self.gross_loss


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
    gross_win = gross_loss = 0.0
    for f in window:
        ts = int(f.get("time") or 0)
        if ts:
            times.append(ts)
        cp = fnum(f.get("closedPnl"))
        closed_pnl += cp
        fees += abs(fnum(f.get("fee")))
        if cp > 0:
            wins += 1
            gross_win += cp
        elif cp < 0:
            losses += 1
            gross_loss += abs(cp)
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
        gross_win=gross_win,
        gross_loss=gross_loss,
        last_fill_ms=times[-1] if times else 0,
    )


def passes_copy_filters(recent: FillStats, history: FillStats, cfg: Any) -> tuple[bool, str]:
    min_f = int(getattr(cfg, "COPY_MIN_FILLS", 6) or 6)
    max_f = int(getattr(cfg, "COPY_MAX_FILLS", 180) or 180)
    min_gap = float(getattr(cfg, "COPY_MIN_MEDIAN_GAP_S", 120.0) or 120.0)
    max_gap = float(getattr(cfg, "COPY_MAX_MEDIAN_GAP_S", 43200.0) or 43200.0)
    min_fpd = float(getattr(cfg, "COPY_MIN_FILLS_PER_DAY", 1.5) or 1.5)
    max_fpd = float(getattr(cfg, "COPY_MAX_FILLS_PER_DAY", 40.0) or 40.0)
    min_wr = float(getattr(cfg, "COPY_MIN_WIN_RATE", 0.52) or 0.52)
    min_hist_wr = float(getattr(cfg, "COPY_MIN_HIST_WIN_RATE", 0.48) or 0.48)
    min_pnl = float(getattr(cfg, "COPY_MIN_RECENT_PNL", 50.0) or 50.0)
    min_hist_pnl = float(getattr(cfg, "COPY_MIN_HIST_PNL", 100.0) or 100.0)
    min_pf = float(getattr(cfg, "COPY_MIN_PROFIT_FACTOR", 1.15) or 1.15)

    if recent.n_fills < min_f:
        return False, f"too_few_fills={recent.n_fills}"
    if recent.n_fills > max_f:
        return False, f"too_many_fills={recent.n_fills}"
    if recent.median_gap_s < min_gap:
        return False, f"scalpy gap={recent.median_gap_s:.0f}s"
    if recent.median_gap_s > max_gap:
        return False, f"dormant gap={recent.median_gap_s:.0f}s"
    if recent.fills_per_day < min_fpd:
        return False, f"slow {recent.fills_per_day:.1f}/d"
    if recent.fills_per_day > max_fpd:
        return False, f"hyper {recent.fills_per_day:.1f}/d"
    if recent.wins + recent.losses < 3:
        return False, f"few_closed={recent.wins + recent.losses}"
    if recent.win_rate < min_wr:
        return False, f"low_wr={recent.win_rate:.0%}"
    if history.win_rate < min_hist_wr and history.wins + history.losses >= 8:
        return False, f"hist_wr={history.win_rate:.0%}"
    if recent.closed_pnl < min_pnl:
        return False, f"recent_pnl=${recent.closed_pnl:.0f}"
    if history.closed_pnl < min_hist_pnl:
        return False, f"hist_pnl=${history.closed_pnl:.0f}"
    if recent.profit_factor < min_pf and recent.losses >= 2:
        return False, f"pf={recent.profit_factor:.2f}"
    return True, "ok"


def score_copy_wallet(
    wallet: QualifiedWallet,
    recent: FillStats,
    history: FillStats,
    cfg: Any,
) -> float:
    """Higher = better copy candidate. Biased to consistent profit + healthy activity."""
    # Consistency: win rates (recent weighted heavier) + profit factor.
    wr = recent.win_rate * 55.0 + history.win_rate * 30.0
    pf = min(recent.profit_factor, 4.0) * 12.0 + min(history.profit_factor, 4.0) * 6.0

    # Realized profit quality (both windows must be strong to score high).
    recent_pnl = math.log10(max(recent.closed_pnl, 1.0) + 10.0) * 10.0
    hist_pnl = math.log10(max(history.closed_pnl, 1.0) + 10.0) * 6.0
    if recent.closed_pnl < 0:
        recent_pnl -= 25.0
    if history.closed_pnl < 0:
        hist_pnl -= 20.0

    # Soft ROI tilt (leaderboard) — secondary to fill consistency.
    roi = max(wallet.rank_roi, -0.2) * 100.0 * 0.20

    # Prefer ~30m median gap / ~5–15 fills/day (active, copyable, not spam).
    ideal_gap = float(getattr(cfg, "COPY_IDEAL_GAP_S", 1800.0) or 1800.0)
    gap_pen = abs(math.log(max(recent.median_gap_s, 30.0) / max(ideal_gap, 30.0)))
    activity = max(0.0, 18.0 - gap_pen * 5.0)
    fpd = recent.fills_per_day
    if 3.0 <= fpd <= 20.0:
        activity += 8.0
    elif 1.5 <= fpd < 3.0 or 20.0 < fpd <= 30.0:
        activity += 3.0
    else:
        activity -= 4.0

    # Prefer wallets that traded recently (still “live”).
    freshness = 0.0
    if recent.last_fill_ms > 0:
        age_h = max(0.0, (time.time() * 1000 - recent.last_fill_ms) / 3_600_000)
        if age_h <= 6:
            freshness = 10.0
        elif age_h <= 24:
            freshness = 5.0
        elif age_h <= 48:
            freshness = 1.0
        else:
            freshness = -8.0

    return wr + pf + recent_pnl + hist_pnl + roi + activity + freshness


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
    scan = max(want, int(getattr(cfg, "COPY_CANDIDATE_SCAN", 200) or 200))
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
            if i < 20:
                log.info("Copy skip %s roi=%.1f%% — %s", w.address[:10], w.rank_roi * 100, why)
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
                reasons=[
                    why,
                    (
                        f"fills={recent.n_fills} {recent.fills_per_day:.1f}/d "
                        f"gap={recent.median_gap_s:.0f}s wr={recent.win_rate:.0%} "
                        f"pf={recent.profit_factor:.2f} pnl=${recent.closed_pnl:.0f}"
                    ),
                ],
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
            "  copy #%s %s score=%.1f roi=%.1f%% wr=%.0f%% pf=%.2f pnl=$%.0f "
            "%.1f/d gap=%.0fm hist_wr=%.0f%% hist_pnl=$%.0f",
            j,
            ld.address[:10],
            ld.score,
            ld.rank_roi * 100,
            ld.recent.win_rate * 100,
            ld.recent.profit_factor,
            ld.recent.closed_pnl,
            ld.recent.fills_per_day,
            ld.recent.median_gap_s / 60.0,
            ld.history.win_rate * 100,
            ld.history.closed_pnl,
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
            "fills_per_day": ld.recent.fills_per_day,
            "profit_factor": ld.recent.profit_factor,
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
            fills_per_day=float(item.get("fills_per_day") or 0),
            gross_win=max(0.0, float(item.get("recent_pnl") or 0)),
            gross_loss=1.0,
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
