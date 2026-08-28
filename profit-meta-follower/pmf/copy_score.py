"""Score leaderboard wallets for copy-trade mode (activity band + consistency)."""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .qualify import Qualifier, _fill_closed_pnl, copy_rank_window, is_holder_tape
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


def _board_yield(wallet: QualifiedWallet) -> float:
    vol = max(float(wallet.rank_volume or 0), 1.0)
    return float(wallet.rank_pnl or 0) / vol


def _fmt_board(wallet: QualifiedWallet) -> str:
    vol = float(wallet.rank_volume or 0)
    pnl = float(wallet.rank_pnl or 0)
    y = _board_yield(wallet) if vol > 0 else 0.0
    return (
        f"roi={wallet.rank_roi * 100:.1f}% pnl=${pnl:,.0f} vol=${vol:,.0f} "
        f"yield={y * 100:.4f}% eq=${wallet.account_value:,.0f}"
    )


def _fmt_fills(recent: FillStats) -> str:
    return (
        f"fills={recent.n_fills} {recent.fills_per_day:.1f}/d gap={recent.median_gap_s:.1f}s "
        f"wr={recent.win_rate:.0%} tape_pnl=${recent.closed_pnl:.0f} "
        f"flip={recent.fast_flip_ratio:.0%}"
    )


def _loosen_steps(cfg: Any) -> list[dict[str, float]]:
    """Loosen activity thresholds only — never holders, 0-vol, or sub-minute HFT."""
    base_yield = _cfg_float(cfg, "COPY_MIN_PNL_VOLUME_RATIO", 0.00005)
    base_gap = max(60.0, _cfg_float(cfg, "COPY_MIN_MEDIAN_GAP_S", 90.0))
    max_gap = _cfg_float(cfg, "COPY_MAX_MEDIAN_GAP_S", 2700.0)
    min_f = float(max(1, _cfg_int(cfg, "COPY_MIN_FILLS", 12)))
    min_fpd = max(0.0, _cfg_float(cfg, "COPY_MIN_FILLS_PER_DAY", 10.0))
    min_pnl = _cfg_float(cfg, "COPY_MIN_RECENT_PNL", 1.0)
    return [
        {
            "yield_mult": 1.0,
            "min_gap": base_gap,
            "max_gap": max_gap if max_gap > 0 else 0.0,
            "min_fills": min_f,
            "min_fpd": min_fpd,
            "min_tape_pnl": min_pnl,
        },
        {
            "yield_mult": 0.5,
            "min_gap": max(60.0, base_gap * 0.75),
            "max_gap": (max_gap * 1.5) if max_gap > 0 else 0.0,
            "min_fills": max(8.0, min_f * 0.7),
            "min_fpd": max(4.0, min_fpd * 0.6),
            "min_tape_pnl": 0.0,
        },
        {
            "yield_mult": 0.15,
            "min_gap": 60.0,
            "max_gap": (max_gap * 2.0) if max_gap > 0 else 0.0,
            "min_fills": max(6.0, min_f * 0.5),
            "min_fpd": max(2.0, min_fpd * 0.35),
            "min_tape_pnl": -1e9,
        },
    ]


def passes_copy_board(
    wallet: QualifiedWallet,
    cfg: Any,
    *,
    yield_mult: float = 1.0,
) -> tuple[bool, str]:
    min_vol = _cfg_float(cfg, "COPY_MIN_BOARD_VOLUME", 1.0)
    base_yield = _cfg_float(cfg, "COPY_MIN_PNL_VOLUME_RATIO", 0.00005)
    min_yield = base_yield * max(0.0, yield_mult)
    vol = float(wallet.rank_volume or 0)
    pnl = float(wallet.rank_pnl or 0)
    y = pnl / max(vol, 1.0) if vol > 0 else 0.0
    detail = _fmt_board(wallet)

    if vol <= 0 or (min_vol > 0 and vol < min_vol):
        return False, f"zero_volume | {detail}"
    if pnl <= 0:
        return False, f"no_board_profit pnl=${pnl:,.0f} | {detail}"
    if min_yield > 0 and y < min_yield:
        need = min_yield * 100.0
        return False, (
            f"low_pnl_yield yield={y * 100:.4f}% need>={need:.4f}% "
            f"(pnl/vol churn) | {detail}"
        )
    return True, "ok"


def passes_copy_fills(
    recent: FillStats,
    history: FillStats,
    cfg: Any,
    *,
    min_gap: float,
    hyper_fpd: float = 0.0,
    max_gap: float = 0.0,
    min_fills: float | None = None,
    min_fpd: float | None = None,
    min_tape_pnl: float | None = None,
) -> tuple[bool, str]:
    min_f = int(min_fills) if min_fills is not None else max(0, _cfg_int(cfg, "COPY_MIN_FILLS", 12))
    need_fpd = (
        float(min_fpd)
        if min_fpd is not None
        else _cfg_float(cfg, "COPY_MIN_FILLS_PER_DAY", 10.0)
    )
    max_fpd = _cfg_float(cfg, "COPY_MAX_FILLS_PER_DAY", 300.0)
    gap_max = max_gap if max_gap > 0 else _cfg_float(cfg, "COPY_MAX_MEDIAN_GAP_S", 2700.0)
    pnl_need = (
        float(min_tape_pnl)
        if min_tape_pnl is not None
        else _cfg_float(cfg, "COPY_MIN_RECENT_PNL", 1.0)
    )
    detail = _fmt_fills(recent)

    if recent.n_fills <= 0:
        return False, f"no_trades fills=0 | {detail}"
    if min_f > 0 and recent.n_fills < min_f:
        return False, f"too_few_fills={recent.n_fills} need>={min_f} | {detail}"
    if need_fpd > 0 and recent.fills_per_day < need_fpd:
        return False, (
            f"idle_activity {recent.fills_per_day:.1f}/d need>={need_fpd:.1f}/d | {detail}"
        )
    if max_fpd > 0 and recent.fills_per_day > max_fpd:
        return False, (
            f"hyper_fpd {recent.fills_per_day:.0f}/d cap={max_fpd:.0f}/d | {detail}"
        )

    # Hard floor: never copy sub-minute HFT even if config is loosened elsewhere.
    hard_min_gap = max(60.0, min_gap) if min_gap > 0 else 60.0
    if recent.n_fills >= 2 and recent.median_gap_s < hard_min_gap:
        return False, (
            f"too_fast_scalp gap={recent.median_gap_s:.1f}s need>={hard_min_gap:.0f}s | {detail}"
        )
    if gap_max > 0 and recent.n_fills >= 2 and recent.median_gap_s > gap_max:
        return False, (
            f"holder_gap gap={recent.median_gap_s:.0f}s max={gap_max:.0f}s | {detail}"
        )
    if hyper_fpd > 0 and recent.fills_per_day > hyper_fpd and recent.median_gap_s < 3.0:
        return False, (
            f"hyper_churn {recent.fills_per_day:.0f}/d gap={recent.median_gap_s:.1f}s "
            f"cap={hyper_fpd:.0f}/d | {detail}"
        )

    if pnl_need > -1e8 and recent.closed_pnl < pnl_need:
        return False, f"tape_pnl=${recent.closed_pnl:.0f} need>=${pnl_need:.0f} | {detail}"

    return True, "ok"


def passes_copy_filters(recent: FillStats, history: FillStats, cfg: Any) -> tuple[bool, str]:
    """Strictest fill tier (tests / back-compat). Board filters use passes_copy_board."""
    gap = _cfg_float(cfg, "COPY_MIN_MEDIAN_GAP_S", 90.0)
    return passes_copy_fills(recent, history, cfg, min_gap=gap, hyper_fpd=0.0)


def hard_disqualify_copy(
    wallet: QualifiedWallet,
    recent: FillStats,
    history: FillStats,
    cfg: Any,
    *,
    holder_reason: str = "",
) -> tuple[bool, str]:
    """Hard gates only — soft quality issues are handled via score_copy_wallet."""
    detail_b = _fmt_board(wallet)
    detail_f = _fmt_fills(recent)
    if holder_reason:
        return False, f"holder:{holder_reason} | {detail_b}"

    min_vol = _cfg_float(cfg, "COPY_MIN_BOARD_VOLUME", 1.0)
    vol = float(wallet.rank_volume or 0)
    pnl = float(wallet.rank_pnl or 0)
    if vol <= 0 or (min_vol > 0 and vol < min_vol):
        return False, f"zero_volume | {detail_b}"
    if pnl <= 0:
        return False, f"no_board_profit pnl=${pnl:,.0f} | {detail_b}"
    if recent.n_fills <= 0:
        return False, f"no_trades fills=0 | {detail_f}"

    hard_min_gap = max(60.0, _cfg_float(cfg, "COPY_MIN_MEDIAN_GAP_S", 90.0))
    if recent.n_fills >= 2 and recent.median_gap_s < hard_min_gap:
        return False, (
            f"too_fast_scalp gap={recent.median_gap_s:.1f}s need>={hard_min_gap:.0f}s | {detail_f}"
        )

    max_fpd = _cfg_float(cfg, "COPY_MAX_FILLS_PER_DAY", 300.0)
    if max_fpd > 0 and recent.fills_per_day > max_fpd:
        return False, (
            f"hyper_fpd {recent.fills_per_day:.0f}/d cap={max_fpd:.0f}/d | {detail_f}"
        )

    return True, "ok"


def score_copy_wallet(
    wallet: QualifiedWallet,
    recent: FillStats,
    history: FillStats,
    cfg: Any,
) -> float:
    """Higher = better for stable profitable decent-scalp copy targets."""
    lookback_d = max(0.25, float(getattr(cfg, "COPY_LOOKBACK_DAYS", 7.0) or 7.0))
    ideal_tph = float(getattr(cfg, "COPY_IDEAL_TRADES_PER_HOUR", 4.0) or 4.0)
    trips_per_day = (
        recent.round_trips / lookback_d if recent.round_trips > 0 else recent.fills_per_day / 2.0
    )
    ideal_tpd = max(1.0, ideal_tph * 24.0)

    # Stable profit: realized tape + board PnL (ROI is only a soft tilt).
    roi = max(min(wallet.rank_roi, 5.0), 0.0) * 6.0
    board_pnl = math.log10(max(wallet.rank_pnl, 1.0) + 10.0) * 10.0
    recent_pnl = math.log10(max(recent.closed_pnl, 1.0) + 10.0) * 28.0
    hist_pnl = math.log10(max(history.closed_pnl, 1.0) + 10.0) * 12.0
    if recent.closed_pnl < 0:
        recent_pnl -= 55.0
    elif recent.closed_pnl > 0:
        recent_pnl += 8.0
    if history.closed_pnl < 0:
        hist_pnl -= 25.0
    if wallet.rank_pnl < 0:
        board_pnl -= 30.0

    wr = recent.win_rate * 40.0 + history.win_rate * 15.0
    pf = min(recent.profit_factor, 4.0) * 12.0 + min(history.profit_factor, 4.0) * 4.0

    # Decent scalp band: prefer ~3–10 min gaps, not sub-minute and not multi-hour holders.
    ideal_gap = float(getattr(cfg, "COPY_IDEAL_GAP_S", 300.0) or 300.0)
    gap = max(recent.median_gap_s, 1.0)
    gap_pen = abs(math.log(gap / max(ideal_gap, 30.0)))
    activity = max(0.0, 28.0 - gap_pen * 9.0)
    if 120.0 <= gap <= 1800.0:
        activity += 14.0
    elif 60.0 <= gap < 120.0 or 1800.0 < gap <= 3600.0:
        activity += 4.0
    else:
        activity -= 12.0

    tpd_pen = abs(math.log(max(trips_per_day, 0.1) / ideal_tpd))
    activity += max(0.0, 16.0 - tpd_pen * 7.0)
    fpd = recent.fills_per_day
    if 12.0 <= fpd <= 180.0:
        activity += 12.0
    elif 6.0 <= fpd < 12.0 or 180.0 < fpd <= 280.0:
        activity += 3.0
    else:
        activity -= 10.0

    bait_pen = recent.fast_flip_ratio * 35.0
    max_flip = float(getattr(cfg, "COPY_MAX_FAST_FLIP_RATIO", 0.55) or 0.55)
    if max_flip > 0 and recent.fast_flip_ratio > max_flip:
        bait_pen += 25.0

    freshness = 0.0
    if recent.last_fill_ms > 0:
        age_h = max(0.0, (time.time() * 1000 - recent.last_fill_ms) / 3_600_000)
        if age_h <= 2:
            freshness = 16.0
        elif age_h <= 8:
            freshness = 8.0
        elif age_h <= 24:
            freshness = 2.0
        else:
            freshness = -14.0
    else:
        freshness = -20.0

    soft_pen = 0.0
    min_f = max(0, _cfg_int(cfg, "COPY_MIN_FILLS", 12))
    if min_f > 0 and recent.n_fills < min_f:
        soft_pen += (min_f - recent.n_fills) * 2.5
    need_fpd = _cfg_float(cfg, "COPY_MIN_FILLS_PER_DAY", 10.0)
    if need_fpd > 0 and recent.fills_per_day < need_fpd:
        soft_pen += (need_fpd - recent.fills_per_day) * 3.0
    gap_max = _cfg_float(cfg, "COPY_MAX_MEDIAN_GAP_S", 2700.0)
    if gap_max > 0 and recent.n_fills >= 2 and recent.median_gap_s > gap_max:
        soft_pen += min(40.0, (recent.median_gap_s - gap_max) / 120.0)
    min_yield = _cfg_float(cfg, "COPY_MIN_PNL_VOLUME_RATIO", 0.00005)
    vol = max(float(wallet.rank_volume or 0), 1.0)
    y = float(wallet.rank_pnl or 0) / vol
    if min_yield > 0 and y < min_yield:
        soft_pen += min(30.0, (min_yield - y) / max(min_yield, 1e-9) * 12.0)
    tape_need = _cfg_float(cfg, "COPY_MIN_RECENT_PNL", 1.0)
    if tape_need > 0 and recent.closed_pnl < tape_need:
        soft_pen += min(35.0, (tape_need - recent.closed_pnl) * 0.05)

    return (
        wr + pf + recent_pnl + hist_pnl + board_pnl + roi + activity + freshness - bait_pen - soft_pen
    )


@dataclass
class CopyScanResult:
    leaders: list[CopyLeader]
    rejects: dict[str, str] = field(default_factory=dict)
    scanned: list[str] = field(default_factory=list)
    next_offset: int = 0
    fetched: int = 0
    exhausted: bool = False


def pick_copy_leaders(
    pool: list[QualifiedWallet],
    qualifier: Qualifier,
    cfg: Any,
    *,
    logger: logging.Logger | None = None,
    keep: list[CopyLeader] | None = None,
    skip_addrs: set[str] | None = None,
    start_offset: int = 0,
) -> CopyScanResult:
    """One full pass over the ROI shortlist; pick top-N by copy-trade score."""
    log = logger or qualifier.log
    want = max(1, _cfg_int(cfg, "COPY_TOP_N", 5))
    board_scan = max(1, _cfg_int(cfg, "COPY_BOARD_SCAN", 100))
    candidate_scan = max(board_scan, _cfg_int(cfg, "COPY_CANDIDATE_SCAN", board_scan))
    fetch_max = max(candidate_scan, _cfg_int(cfg, "COPY_FILL_FETCH_MAX", candidate_scan + 10))
    max_roi = _cfg_float(cfg, "COPY_MAX_ROI", 0.0)
    min_eq = _cfg_float(cfg, "COPY_MIN_EQUITY", 1000.0)
    sleep_s = _cfg_float(cfg, "COPY_FILL_SLEEP_S", 0.7)
    recent_d = _cfg_float(cfg, "COPY_LOOKBACK_DAYS", 7.0)
    hist_d = _cfg_float(cfg, "COPY_HISTORY_DAYS", 30.0)
    min_hold = _cfg_float(cfg, "COPY_MIN_HOLD_S", 90.0)
    exclude_holders = bool(getattr(cfg, "COPY_EXCLUDE_HOLDERS", True))
    now_ms = int(time.time() * 1000)
    start_hist = now_ms - int(hist_d * 86400_000)

    eligible: list[QualifiedWallet] = []
    for w in pool:
        if max_roi > 0 and w.rank_roi > max_roi:
            continue
        if min_eq > 0 and w.account_value < min_eq:
            continue
        if w.rank_roi <= 0:
            continue
        eligible.append(w)
    eligible.sort(key=lambda w: (-w.rank_roi, w.address.lower()))

    scan_size = min(len(eligible), candidate_scan, fetch_max)
    wave = eligible[:scan_size]

    log.info(
        "Copy score scan: board=%s eligible=%s want=%s window=%s",
        scan_size,
        len(eligible),
        want,
        copy_rank_window(cfg),
    )
    if wave:
        log.info(
            "Copy board span: #%s %s roi=%.1f%% … #%s %s roi=%.1f%%",
            1,
            wave[0].address[:10],
            wave[0].rank_roi * 100,
            scan_size,
            wave[-1].address[:10],
            wave[-1].rank_roi * 100,
        )

    scored: list[CopyLeader] = []
    scanned: list[str] = []
    rejects: dict[str, str] = {}
    fetch_fail = 0
    fetched = 0
    holder_n = 0
    hard_n = 0
    skip_logs = 0

    for w in wave:
        addr = w.address.lower()
        if fetched >= fetch_max:
            break
        fills = qualifier._recent_fills(addr, start_hist)
        fetched += 1
        scanned.append(addr)
        if fetched > 1:
            time.sleep(sleep_s)
        if fills is None:
            fetch_fail += 1
            rejects[addr] = f"fills_failed | {_fmt_board(w)}"
            log.info("Copy skip [fetch] %s — fills_failed | %s", addr[:10], _fmt_board(w))
            continue

        if exclude_holders:
            is_hold, hold_why = is_holder_tape(fills, now_ms, cfg)
            if is_hold:
                holder_n += 1
                rejects[addr] = f"holder:{hold_why} | {_fmt_board(w)}"
                if skip_logs < 25:
                    log.info(
                        "Copy skip [holder] %s — %s | %s",
                        addr[:10],
                        hold_why,
                        _fmt_board(w),
                    )
                    skip_logs += 1
                continue

        recent = analyze_fills(
            fills, now_ms=now_ms, lookback_days=recent_d, min_hold_s=min_hold
        )
        history = analyze_fills(
            fills, now_ms=now_ms, lookback_days=hist_d, min_hold_s=min_hold
        )
        ok, why = hard_disqualify_copy(w, recent, history, cfg)
        if not ok:
            hard_n += 1
            rejects[addr] = why
            if skip_logs < 25:
                log.info("Copy skip [hard] %s — %s", addr[:10], why)
                skip_logs += 1
            continue

        score = score_copy_wallet(w, recent, history, cfg)
        scored.append(
            CopyLeader(
                address=addr,
                score=score,
                rank_roi=w.rank_roi,
                rank_pnl=w.rank_pnl,
                account_value=w.account_value,
                recent=recent,
                history=history,
                reasons=[
                    f"score={score:.1f}",
                    _fmt_board(w),
                    _fmt_fills(recent),
                ],
            )
        )

    scored.sort(key=lambda x: (-x.score, -x.rank_roi, x.address))
    leaders = scored[:want]

    log.info(
        "Copy scan done [score]: fetched=%s holders=%s hard_skip=%s scored=%s picked=%s/%s",
        fetched,
        holder_n,
        hard_n,
        len(scored),
        len(leaders),
        want,
    )
    for j, ld in enumerate(leaders, 1):
        log.info(
            "  copy #%s %s score=%.1f roi=%.1f%% %s",
            j,
            ld.address[:10],
            ld.score,
            ld.rank_roi * 100,
            ld.reasons[-1] if ld.reasons else "",
        )

    return CopyScanResult(
        leaders=leaders,
        rejects=rejects,
        scanned=scanned,
        next_offset=scan_size,
        fetched=fetched,
        exhausted=True,
    )


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
