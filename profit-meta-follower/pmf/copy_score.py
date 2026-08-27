"""Score leaderboard wallets for copy-trade mode (activity band + consistency)."""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections import defaultdict
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
    fast_flips: int = 0
    round_trips: int = 0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss <= 1e-9:
            return 5.0 if self.gross_win > 0 else 0.0
        return self.gross_win / self.gross_loss

    @property
    def fast_flip_ratio(self) -> float:
        if self.round_trips <= 0:
            return 0.0
        return self.fast_flips / max(1, self.round_trips)


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


def _fill_side_sign(fill: dict[str, Any]) -> int:
    """+1 buy / -1 sell from HL fill fields."""
    side = str(fill.get("side") or "").upper()
    if side in ("B", "BUY"):
        return 1
    if side in ("A", "SELL", "S"):
        return -1
    direction = str(fill.get("dir") or "").lower()
    if "open long" in direction or "close short" in direction:
        return 1
    if "open short" in direction or "close long" in direction:
        return -1
    return 0


def _count_fast_flips(window: list[dict[str, Any]], *, min_hold_s: float) -> tuple[int, int]:
    """Count open→close round trips that finish faster than min_hold_s (bait tape)."""
    by_coin: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for f in window:
        coin = str(f.get("coin") or "").strip()
        if not coin:
            continue
        ts = int(f.get("time") or 0)
        sign = _fill_side_sign(f)
        if ts <= 0 or sign == 0:
            continue
        by_coin[coin].append((ts, sign))
    fast = 0
    trips = 0
    hold_ms = int(max(min_hold_s, 30.0) * 1000.0)
    for events in by_coin.values():
        events.sort(key=lambda x: x[0])
        open_ts: int | None = None
        open_sign = 0
        for ts, sign in events:
            if open_ts is None:
                open_ts = ts
                open_sign = sign
                continue
            if sign == -open_sign:
                trips += 1
                if ts - open_ts < hold_ms:
                    fast += 1
                open_ts = None
                open_sign = 0
            else:
                open_ts = ts
                open_sign = sign
    return fast, trips


def analyze_fills(
    fills: list[Any],
    *,
    now_ms: int,
    lookback_days: float,
    min_hold_s: float = 300.0,
) -> FillStats:
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
    fast_flips, round_trips = _count_fast_flips(window, min_hold_s=min_hold_s)
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
        fast_flips=fast_flips,
        round_trips=round_trips,
    )


def _cfg_float(cfg: Any, name: str, default: float) -> float:
    raw = getattr(cfg, name, default)
    if raw is None:
        return float(default)
    return float(raw)


def _cfg_int(cfg: Any, name: str, default: int) -> int:
    raw = getattr(cfg, name, default)
    if raw is None:
        return int(default)
    return int(raw)


def passes_copy_filters(recent: FillStats, history: FillStats, cfg: Any) -> tuple[bool, str]:
    """Minimum floors only for fills/PnL; max_* of 0 means unlimited."""
    min_f = _cfg_int(cfg, "COPY_MIN_FILLS", 10)
    max_f = _cfg_int(cfg, "COPY_MAX_FILLS", 0)
    min_gap = _cfg_float(cfg, "COPY_MIN_MEDIAN_GAP_S", 90.0)
    max_gap = _cfg_float(cfg, "COPY_MAX_MEDIAN_GAP_S", 0.0)
    min_fpd = _cfg_float(cfg, "COPY_MIN_FILLS_PER_DAY", 6.0)
    max_fpd = _cfg_float(cfg, "COPY_MAX_FILLS_PER_DAY", 0.0)
    min_wr = _cfg_float(cfg, "COPY_MIN_WIN_RATE", 0.40)
    min_hist_wr = _cfg_float(cfg, "COPY_MIN_HIST_WIN_RATE", 0.38)
    min_pnl = _cfg_float(cfg, "COPY_MIN_RECENT_PNL", 0.0)
    min_hist_pnl = _cfg_float(cfg, "COPY_MIN_HIST_PNL", 0.0)
    min_pf = _cfg_float(cfg, "COPY_MIN_PROFIT_FACTOR", 1.0)
    max_flip = _cfg_float(cfg, "COPY_MAX_FAST_FLIP_RATIO", 0.0)

    if recent.n_fills < min_f:
        return False, f"too_few_fills={recent.n_fills}"
    if max_f > 0 and recent.n_fills > max_f:
        return False, f"too_many_fills={recent.n_fills}"
    if recent.median_gap_s < min_gap:
        return False, f"scalpy gap={recent.median_gap_s:.0f}s"
    if max_gap > 0 and recent.median_gap_s > max_gap:
        return False, f"dormant gap={recent.median_gap_s:.0f}s"
    if recent.fills_per_day < min_fpd:
        return False, f"slow {recent.fills_per_day:.1f}/d"
    if max_fpd > 0 and recent.fills_per_day > max_fpd:
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
    if min_pf > 0 and recent.profit_factor < min_pf and recent.losses >= 2:
        return False, f"pf={recent.profit_factor:.2f}"
    if max_flip > 0 and recent.round_trips >= 4 and recent.fast_flip_ratio > max_flip:
        return False, f"bait_flips={recent.fast_flip_ratio:.0%}"
    return True, "ok"


def score_copy_wallet(
    wallet: QualifiedWallet,
    recent: FillStats,
    history: FillStats,
    cfg: Any,
) -> float:
    """Higher = better. Prioritize realized profit + ~3 trades/hour activity."""
    lookback_d = max(0.25, float(getattr(cfg, "COPY_LOOKBACK_DAYS", 7.0) or 7.0))
    ideal_tph = float(getattr(cfg, "COPY_IDEAL_TRADES_PER_HOUR", 3.0) or 3.0)
    # Round-trips/day from tape; ideal ≈ 3/hour = 72/day.
    trips_per_day = recent.round_trips / lookback_d if recent.round_trips > 0 else recent.fills_per_day / 2.0
    ideal_tpd = max(1.0, ideal_tph * 24.0)

    # Soft ROI tilt — board ROI already drives pick order.
    roi = max(min(wallet.rank_roi, 50.0), -0.2) * 8.0
    board_pnl = math.log10(max(wallet.rank_pnl, 1.0) + 10.0) * 8.0
    recent_pnl = math.log10(max(recent.closed_pnl, 1.0) + 10.0) * 18.0
    hist_pnl = math.log10(max(history.closed_pnl, 1.0) + 10.0) * 10.0
    if recent.closed_pnl < 0:
        recent_pnl -= 40.0
    if history.closed_pnl < 0:
        hist_pnl -= 30.0
    if wallet.rank_pnl < 0:
        board_pnl -= 25.0

    wr = recent.win_rate * 35.0 + history.win_rate * 20.0
    pf = min(recent.profit_factor, 4.0) * 10.0 + min(history.profit_factor, 4.0) * 5.0

    # Activity: prefer ~3 complete trades/hour (gap ~10m fills / ~20m RTs).
    ideal_gap = float(getattr(cfg, "COPY_IDEAL_GAP_S", 600.0) or 600.0)
    gap_pen = abs(math.log(max(recent.median_gap_s, 30.0) / max(ideal_gap, 30.0)))
    activity = max(0.0, 20.0 - gap_pen * 6.0)
    tpd_pen = abs(math.log(max(trips_per_day, 0.1) / ideal_tpd))
    activity += max(0.0, 18.0 - tpd_pen * 8.0)
    fpd = recent.fills_per_day
    if 15.0 <= fpd <= 200.0:
        activity += 10.0
    elif 8.0 <= fpd < 15.0 or 200.0 < fpd <= 400.0:
        activity += 4.0
    else:
        activity -= 6.0

    bait_pen = 0.0
    max_flip = float(getattr(cfg, "COPY_MAX_FAST_FLIP_RATIO", 0.0) or 0.0)
    if max_flip > 0:
        bait_pen = recent.fast_flip_ratio * 40.0

    freshness = 0.0
    if recent.last_fill_ms > 0:
        age_h = max(0.0, (time.time() * 1000 - recent.last_fill_ms) / 3_600_000)
        if age_h <= 3:
            freshness = 12.0
        elif age_h <= 12:
            freshness = 6.0
        elif age_h <= 36:
            freshness = 2.0
        else:
            freshness = -10.0

    return wr + pf + recent_pnl + hist_pnl + board_pnl + roi + activity + freshness - bait_pen


def _relaxed_copy_cfg(cfg: Any) -> Any:
    """Softer second-pass thresholds when strict scan finds too few leaders."""
    from types import SimpleNamespace

    keys = [
        "COPY_MIN_FILLS",
        "COPY_MAX_FILLS",
        "COPY_MIN_MEDIAN_GAP_S",
        "COPY_MAX_MEDIAN_GAP_S",
        "COPY_MIN_FILLS_PER_DAY",
        "COPY_MAX_FILLS_PER_DAY",
        "COPY_MIN_WIN_RATE",
        "COPY_MIN_HIST_WIN_RATE",
        "COPY_MIN_RECENT_PNL",
        "COPY_MIN_HIST_PNL",
        "COPY_MIN_PROFIT_FACTOR",
        "COPY_MAX_FAST_FLIP_RATIO",
        "COPY_MIN_HOLD_S",
        "COPY_IDEAL_GAP_S",
        "COPY_IDEAL_TRADES_PER_HOUR",
        "COPY_LOOKBACK_DAYS",
    ]
    vals = {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}
    vals.update(
        {
            "COPY_MIN_FILLS": 6,
            "COPY_MAX_FILLS": 0,
            "COPY_MIN_MEDIAN_GAP_S": 60.0,
            "COPY_MAX_MEDIAN_GAP_S": 0.0,
            "COPY_MIN_FILLS_PER_DAY": 4.0,
            "COPY_MAX_FILLS_PER_DAY": 0.0,
            "COPY_MIN_WIN_RATE": 0.35,
            "COPY_MIN_HIST_WIN_RATE": 0.35,
            "COPY_MIN_RECENT_PNL": 0.0,
            "COPY_MIN_HIST_PNL": 0.0,
            "COPY_MIN_PROFIT_FACTOR": 0.0,
            "COPY_MAX_FAST_FLIP_RATIO": 0.0,
        }
    )
    return SimpleNamespace(**vals)


def pick_copy_leaders(
    pool: list[QualifiedWallet],
    qualifier: Qualifier,
    cfg: Any,
    *,
    logger: logging.Logger | None = None,
) -> list[CopyLeader]:
    """Walk 7d ROI board (highest first), keep first COPY_TOP_N that pass min filters.

    Matches HL leaderboard ROI ordering. COPY_MAX_* of 0 = unlimited.
    """
    log = logger or qualifier.log
    want = max(1, _cfg_int(cfg, "COPY_TOP_N", 2))
    board_n = max(want, _cfg_int(cfg, "COPY_CANDIDATE_SCAN", 1200))
    fetch_max = max(want, _cfg_int(cfg, "COPY_FILL_FETCH_MAX", 100))
    max_roi = _cfg_float(cfg, "COPY_MAX_ROI", 0.0)
    min_eq = _cfg_float(cfg, "COPY_MIN_EQUITY", 1000.0)
    sleep_s = _cfg_float(cfg, "COPY_FILL_SLEEP_S", 0.7)
    recent_d = _cfg_float(cfg, "COPY_LOOKBACK_DAYS", 7.0)
    hist_d = _cfg_float(cfg, "COPY_HISTORY_DAYS", 30.0)
    min_hold = _cfg_float(cfg, "COPY_MIN_HOLD_S", 90.0)
    now_ms = int(time.time() * 1000)
    start_hist = now_ms - int(hist_d * 86400_000)

    skipped_roi = 0
    skipped_eq = 0
    eligible: list[QualifiedWallet] = []
    for w in pool[:board_n]:
        if max_roi > 0 and w.rank_roi > max_roi:
            skipped_roi += 1
            continue
        if min_eq > 0 and w.account_value < min_eq:
            skipped_eq += 1
            continue
        eligible.append(w)
    # Highest ROI first (same as HL UI 7d board).
    eligible.sort(key=lambda w: w.rank_roi, reverse=True)

    log.info(
        "Copy scan start: board=%s eligible=%s fetch_cap=%s skip_roi_cap=%s skip_low_eq=%s "
        "max_roi=%s min_eq=$%.0f order=roi window=%s",
        min(board_n, len(pool)),
        len(eligible),
        fetch_max,
        skipped_roi,
        skipped_eq,
        "off" if max_roi <= 0 else f"{max_roi * 100.0:.0f}%",
        min_eq,
        getattr(cfg, "RANK_WINDOW", "week"),
    )
    if eligible:
        top = eligible[0]
        log.info(
            "Copy board head: %s roi=%.1f%% pnl=$%.0f eq=$%.0f … tail roi=%.1f%%",
            top.address[:10],
            top.rank_roi * 100,
            top.rank_pnl,
            top.account_value,
            eligible[min(49, len(eligible) - 1)].rank_roi * 100,
        )

    def _to_leader(
        w: QualifiedWallet,
        recent: FillStats,
        history: FillStats,
        label: str,
        why: str,
    ) -> CopyLeader:
        return CopyLeader(
            address=w.address.lower(),
            score=score_copy_wallet(w, recent, history, cfg),
            rank_roi=w.rank_roi,
            rank_pnl=w.rank_pnl,
            account_value=w.account_value,
            recent=recent,
            history=history,
            reasons=[
                f"{label}:{why}",
                (
                    f"fills={recent.n_fills} {recent.fills_per_day:.1f}/d "
                    f"gap={recent.median_gap_s:.0f}s wr={recent.win_rate:.0%} "
                    f"pf={recent.profit_factor:.2f} pnl=${recent.closed_pnl:.0f} "
                    f"flip={recent.fast_flip_ratio:.0%}"
                ),
            ],
        )

    analyzed: list[tuple[QualifiedWallet, FillStats, FillStats]] = []
    strict_pass: list[CopyLeader] = []
    fetch_fail = 0
    fetched = 0
    skip_logs = 0
    for w in eligible:
        if fetched >= fetch_max or len(strict_pass) >= want:
            break
        fills = qualifier._recent_fills(w.address, start_hist)
        fetched += 1
        if fetched > 1:
            time.sleep(sleep_s)
        if fills is None:
            fetch_fail += 1
            continue
        recent = analyze_fills(
            fills, now_ms=now_ms, lookback_days=recent_d, min_hold_s=min_hold
        )
        history = analyze_fills(
            fills, now_ms=now_ms, lookback_days=hist_d, min_hold_s=min_hold
        )
        analyzed.append((w, recent, history))
        ok, why = passes_copy_filters(recent, history, cfg)
        if not ok:
            if skip_logs < 25:
                log.info(
                    "Copy skip [strict] %s roi=%.1f%% — %s",
                    w.address[:10],
                    w.rank_roi * 100,
                    why,
                )
                skip_logs += 1
            continue
        strict_pass.append(_to_leader(w, recent, history, "strict", why))
        log.info(
            "Copy keep [strict] %s/%s %s roi=%.1f%% gap=%.0fs %.1f/d pnl=$%.0f",
            len(strict_pass),
            want,
            w.address[:10],
            w.rank_roi * 100,
            recent.median_gap_s,
            recent.fills_per_day,
            recent.closed_pnl,
        )

    log.info(
        "Copy strict filter: fetched=%s analyzed=%s passed=%s",
        fetched,
        len(analyzed),
        len(strict_pass),
    )

    leaders = strict_pass[:want]
    label = "strict"
    if len(leaders) < want:
        log.warning(
            "Copy strict pass only %s/%s — applying relaxed filters on same fills",
            len(leaders),
            want,
        )
        relaxed_cfg = _relaxed_copy_cfg(cfg)
        relaxed: list[CopyLeader] = []
        skip_logs = 0
        for w, recent, history in analyzed:
            ok, why = passes_copy_filters(recent, history, relaxed_cfg)
            if not ok:
                if skip_logs < 20:
                    log.info(
                        "Copy skip [relaxed] %s roi=%.1f%% — %s",
                        w.address[:10],
                        w.rank_roi * 100,
                        why,
                    )
                    skip_logs += 1
                continue
            relaxed.append(_to_leader(w, recent, history, "relaxed", why))
            if len(relaxed) >= want:
                break
        leaders = relaxed[:want]
        label = "relaxed"
        log.info(
            "Copy relaxed filter: analyzed=%s passed=%s",
            len(analyzed),
            len(relaxed),
        )

    # Preserve ROI board order (already highest-first among passers).
    log.info(
        "Copy scan done [%s]: fetched=%s fetch_fail=%s analyzed=%s picked=%s",
        label,
        fetched,
        fetch_fail,
        len(analyzed),
        len(leaders),
    )
    for j, ld in enumerate(leaders, 1):
        log.info(
            "  copy #%s %s score=%.1f roi=%.1f%% wr=%.0f%% pf=%.2f pnl=$%.0f "
            "%.1f/d gap=%.0fm flip=%.0f%% hist_wr=%.0f%% hist_pnl=$%.0f",
            j,
            ld.address[:10],
            ld.score,
            ld.rank_roi * 100,
            ld.recent.win_rate * 100,
            ld.recent.profit_factor,
            ld.recent.closed_pnl,
            ld.recent.fills_per_day,
            ld.recent.median_gap_s / 60.0,
            ld.recent.fast_flip_ratio * 100,
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
            "fast_flip_ratio": ld.recent.fast_flip_ratio,
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
            fast_flips=0,
            round_trips=max(1, int(float(item.get("fast_flip_ratio") or 0) * 10)),
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


def quick_fill_stats_from_list(
    fills: list[Any],
    *,
    now_ms: int,
    lookback_days: float,
    min_hold_s: float = 300.0,
) -> FillStats:
    """Public wrapper for backtest."""
    return analyze_fills(fills, now_ms=now_ms, lookback_days=lookback_days, min_hold_s=min_hold_s)


def closed_pnl_from_fills(fills: list[Any]) -> tuple[float, float, int]:
    return _fill_closed_pnl(fills)
