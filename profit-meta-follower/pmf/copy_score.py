"""Score leaderboard wallets for copy-trade mode (activity band + consistency)."""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .qualify import Qualifier, _fill_closed_pnl, copy_rank_window
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
    base_yield = _cfg_float(cfg, "COPY_MIN_PNL_VOLUME_RATIO", 0.00005)
    base_gap = _cfg_float(cfg, "COPY_MIN_MEDIAN_GAP_S", 20.0)
    return [
        {"yield_mult": 1.0, "min_gap": base_gap, "hyper_fpd": 800.0},
        {"yield_mult": 0.25, "min_gap": max(10.0, base_gap * 0.5), "hyper_fpd": 1200.0},
        {"yield_mult": 0.05, "min_gap": 5.0, "hyper_fpd": 2000.0},
        {"yield_mult": 0.0, "min_gap": max(5.0, base_gap * 0.25), "hyper_fpd": 2500.0},
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
    hyper_fpd: float,
) -> tuple[bool, str]:
    min_f = max(0, _cfg_int(cfg, "COPY_MIN_FILLS", 1))
    detail = _fmt_fills(recent)

    if min_f > 0 and recent.n_fills < min_f:
        return False, f"too_few_fills={recent.n_fills} need>={min_f} | {detail}"

    if min_gap > 0:
        if recent.n_fills >= 2 and recent.median_gap_s < min_gap:
            return False, (
                f"scalpy gap={recent.median_gap_s:.1f}s need>={min_gap:.0f}s | {detail}"
            )
    elif hyper_fpd > 0 and recent.fills_per_day > hyper_fpd and recent.median_gap_s < 3.0:
        return False, (
            f"hyper_churn {recent.fills_per_day:.0f}/d gap={recent.median_gap_s:.1f}s "
            f"cap={hyper_fpd:.0f}/d | {detail}"
        )

    min_pnl = _cfg_float(cfg, "COPY_MIN_RECENT_PNL", 0.0)
    if min_pnl > 0 and recent.closed_pnl < min_pnl:
        return False, f"tape_pnl=${recent.closed_pnl:.0f} need>=${min_pnl:.0f} | {detail}"

    return True, "ok"


def passes_copy_filters(recent: FillStats, history: FillStats, cfg: Any) -> tuple[bool, str]:
    """Strictest fill tier (tests / back-compat). Board filters use passes_copy_board."""
    gap = _cfg_float(cfg, "COPY_MIN_MEDIAN_GAP_S", 20.0)
    return passes_copy_fills(recent, history, cfg, min_gap=gap, hyper_fpd=800.0)


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
    """Walk ROI board in waves; auto-loosen per wave; pick top COPY_TOP_N."""
    log = logger or qualifier.log
    want = max(1, _cfg_int(cfg, "COPY_TOP_N", 5))
    wave_size = max(1, _cfg_int(cfg, "COPY_BOARD_SCAN", 200))
    fetch_max = max(wave_size, _cfg_int(cfg, "COPY_FILL_FETCH_MAX", 220))
    max_roi = _cfg_float(cfg, "COPY_MAX_ROI", 0.0)
    min_eq = _cfg_float(cfg, "COPY_MIN_EQUITY", 1000.0)
    sleep_s = _cfg_float(cfg, "COPY_FILL_SLEEP_S", 0.7)
    recent_d = _cfg_float(cfg, "COPY_LOOKBACK_DAYS", 7.0)
    hist_d = _cfg_float(cfg, "COPY_HISTORY_DAYS", 30.0)
    min_hold = _cfg_float(cfg, "COPY_MIN_HOLD_S", 90.0)
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

    kept: list[CopyLeader] = []
    seen_keep: set[str] = set()
    for ld in keep or []:
        addr = str(ld.address or "").lower()
        if not addr or addr in seen_keep:
            continue
        kept.append(ld)
        seen_keep.add(addr)
        if len(kept) >= want:
            break

    slots = max(0, want - len(kept))
    min_pass = max(1, slots) if slots > 0 else 0
    offset = max(0, min(int(start_offset), len(eligible)))
    wave_end = min(len(eligible), offset + wave_size)
    wave = eligible[offset:wave_end]

    skip = {a.lower() for a in (skip_addrs or set())} | seen_keep

    log.info(
        "Copy pick: wave=%s..%s eligible=%s need=%s min_pass=%s keep=%s window=%s",
        offset + 1 if wave else offset,
        wave_end,
        len(eligible),
        want,
        min_pass,
        len(kept),
        copy_rank_window(cfg),
    )
    if wave:
        log.info(
            "Copy board span: #%s %s roi=%.1f%% … #%s %s roi=%.1f%%",
            offset + 1,
            wave[0].address[:10],
            wave[0].rank_roi * 100,
            wave_end,
            wave[-1].address[:10],
            wave[-1].rank_roi * 100,
        )

    if len(kept) >= want:
        return CopyScanResult(
            leaders=kept[:want],
            rejects={},
            scanned=[],
            next_offset=offset,
            fetched=0,
            exhausted=wave_end >= len(eligible),
        )

    analyzed: list[tuple[QualifiedWallet, FillStats, FillStats]] = []
    scanned: list[str] = []
    rejects: dict[str, str] = {}
    fetch_fail = 0
    fetched = 0
    cached_hits = 0
    idx = offset

    for w in wave:
        addr = w.address.lower()
        idx += 1
        if addr in skip:
            cached_hits += 1
            continue
        if fetched >= fetch_max:
            break
        fills = qualifier._recent_fills(addr, start_hist)
        fetched += 1
        scanned.append(addr)
        skip.add(addr)
        if fetched > 1:
            time.sleep(sleep_s)
        if fills is None:
            fetch_fail += 1
            rejects[addr] = f"fills_failed | {_fmt_board(w)}"
            log.info("Copy skip [fetch] %s — fills_failed | %s", addr[:10], _fmt_board(w))
            continue
        recent = analyze_fills(
            fills, now_ms=now_ms, lookback_days=recent_d, min_hold_s=min_hold
        )
        history = analyze_fills(
            fills, now_ms=now_ms, lookback_days=hist_d, min_hold_s=min_hold
        )
        analyzed.append((w, recent, history))

    steps = _loosen_steps(cfg)
    passers: list[CopyLeader] = []
    used_label = "strict"
    skip_logs = 0

    for si, step in enumerate(steps):
        label = f"tier{si}"
        passers = []
        for w, recent, history in analyzed:
            addr = w.address.lower()
            if addr in seen_keep:
                continue
            ok_b, why_b = passes_copy_board(w, cfg, yield_mult=step["yield_mult"])
            if not ok_b:
                rejects[addr] = f"[{label}] {why_b}"
                if skip_logs < 25:
                    log.info("Copy skip [%s] %s — %s", label, addr[:10], why_b)
                    skip_logs += 1
                continue
            ok_f, why_f = passes_copy_fills(
                recent,
                history,
                cfg,
                min_gap=step["min_gap"],
                hyper_fpd=step["hyper_fpd"],
            )
            if not ok_f:
                rejects[addr] = f"[{label}] {why_f}"
                if skip_logs < 25:
                    log.info(
                        "Copy skip [%s] %s — %s | %s",
                        label,
                        addr[:10],
                        why_f,
                        _fmt_board(w),
                    )
                    skip_logs += 1
                continue
            passers.append(
                CopyLeader(
                    address=addr,
                    score=score_copy_wallet(w, recent, history, cfg),
                    rank_roi=w.rank_roi,
                    rank_pnl=w.rank_pnl,
                    account_value=w.account_value,
                    recent=recent,
                    history=history,
                    reasons=[f"{label}:ok", _fmt_board(w), _fmt_fills(recent)],
                )
            )
        used_label = label
        log.info(
            "Copy tier %s: yield_mult=%.2f min_gap=%.0fs hyper_fpd=%.0f pass=%s/%s (need>=%s)",
            si,
            step["yield_mult"],
            step["min_gap"],
            step["hyper_fpd"],
            len(passers),
            len(analyzed),
            min_pass,
        )
        if slots <= 0 or len(passers) >= min_pass:
            break

    passers.sort(key=lambda x: (-x.rank_roi, x.address))
    new_pick = passers[:slots]
    leaders = (kept + new_pick)[:want]
    next_offset = idx if fetched > 0 or cached_hits > 0 else wave_end
    exhausted = next_offset >= len(eligible)

    log.info(
        "Copy scan done [%s]: wave_idx=%s→%s fetched=%s skip=%s fail=%s "
        "analyzed=%s pass=%s picked=%s/%s",
        used_label,
        offset,
        next_offset,
        fetched,
        cached_hits,
        fetch_fail,
        len(analyzed),
        len(passers),
        len(leaders),
        want,
    )
    for j, ld in enumerate(leaders, 1):
        log.info(
            "  copy #%s %s roi=%.1f%% %s",
            j,
            ld.address[:10],
            ld.rank_roi * 100,
            ld.reasons[-1] if ld.reasons else "",
        )

    return CopyScanResult(
        leaders=leaders,
        rejects=rejects,
        scanned=scanned,
        next_offset=next_offset,
        fetched=fetched,
        exhausted=exhausted,
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
