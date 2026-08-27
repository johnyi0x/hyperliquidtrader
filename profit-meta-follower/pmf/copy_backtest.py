"""Backtest copy-trade wallet selection + mirror PnL from research books."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .bt_numba import simulate_portfolio
from .copy_exec import copy_targets_from_leaders
from .copy_score import (
    CopyLeader,
    FillStats,
    analyze_fills,
    passes_copy_filters,
    score_copy_wallet,
)
from .research_load import ResearchDataset
from .types import QualifiedWallet, WalletSnapshot


@dataclass
class CopyBacktestResult:
    return_pct: float
    max_dd_pct: float
    round_trips: int
    win_rate_pct: float
    open_legs: int
    leaders_picked: list[str]
    reselects: int
    span_days: float


@dataclass
class _SynthFill:
    time_ms: int
    closed_pnl: float
    fee: float = 0.0


@dataclass
class _WalletTrack:
    address: str
    rank_roi: float = 0.0
    rank_pnl: float = 0.0
    account_value: float = 10_000.0
    fills: list[_SynthFill] = field(default_factory=list)
    snaps_by_tick: dict[int, WalletSnapshot] = field(default_factory=dict)


def _pos_key(pos) -> tuple[str, str]:
    return (str(pos.coin), str(pos.side).lower())


def _infer_synthetic_fills(
    track: _WalletTrack,
    ds: ResearchDataset,
    *,
    fee_rate: float = 0.0005,
) -> None:
    """Build synthetic closed-PnL events from book position changes + marks."""
    prev: dict[str, tuple[str, float]] = {}
    prev_ts = 0.0
    for ti in range(ds.n_ticks):
        snap = track.snaps_by_tick.get(ti)
        if snap is None:
            continue
        ts = float(ds.ts[ti])
        ts_ms = int(ts * 1000)
        cur: dict[str, tuple[str, float]] = {}
        for pos in snap.positions:
            cur[pos.coin] = (pos.side.lower(), float(pos.entry_px or 0.0))
        coins = set(prev) | set(cur)
        for coin in coins:
            j = ds.coin_index.get(coin)
            if j is None:
                continue
            mark = float(ds.marks[ti, j]) if j < ds.marks.shape[1] else 0.0
            if mark <= 0:
                continue
            was = prev.get(coin)
            now_p = cur.get(coin)
            if was and (now_p is None or now_p[0] != was[0]):
                side, entry = was
                signed = 1.0 if side == "long" else -1.0
                pnl = signed * (mark - entry) / max(entry, mark) * track.account_value * 0.05
                track.fills.append(_SynthFill(time_ms=ts_ms, closed_pnl=pnl, fee=mark * fee_rate))
            elif was is None and now_p is not None:
                pass
            elif was and now_p and abs(now_p[1] - was[1]) / max(was[1], 1e-9) > 0.02:
                side, entry = was
                signed = 1.0 if side == "long" else -1.0
                pnl = signed * (mark - entry) / max(entry, mark) * track.account_value * 0.03
                track.fills.append(_SynthFill(time_ms=ts_ms, closed_pnl=pnl, fee=mark * fee_rate))
        prev = cur
        prev_ts = ts
    track.fills.sort(key=lambda f: f.time_ms)


def _build_wallet_tracks(ds: ResearchDataset) -> dict[str, _WalletTrack]:
    tracks: dict[str, _WalletTrack] = {}
    for ti in range(ds.n_ticks):
        book = ds.books[ti]
        if book is None:
            continue
        for snap in book.wallets:
            addr = snap.address.lower()
            tr = tracks.setdefault(addr, _WalletTrack(address=addr, account_value=snap.account_value))
            tr.account_value = max(tr.account_value, snap.account_value)
            tr.snaps_by_tick[ti] = snap
    for addr, tr in tracks.items():
        _infer_synthetic_fills(tr, ds)
    return tracks


def _stats_from_synth(fills: list[_SynthFill], *, now_ms: int, lookback_days: float) -> FillStats:
    raw = [{"time": f.time_ms, "closedPnl": f.closed_pnl, "fee": f.fee} for f in fills]
    return analyze_fills(raw, now_ms=now_ms, lookback_days=lookback_days)


def _score_tracks_at_tick(
    tracks: dict[str, _WalletTrack],
    ti: int,
    ds: ResearchDataset,
    cfg: Any,
) -> list[CopyLeader]:
    now_ms = int(float(ds.ts[ti]) * 1000)
    recent_d = float(getattr(cfg, "COPY_LOOKBACK_DAYS", 7.0) or 7.0)
    hist_d = float(getattr(cfg, "COPY_HISTORY_DAYS", 30.0) or 30.0)
    scan = int(getattr(cfg, "COPY_CANDIDATE_SCAN", 120) or 120)
    ranked: list[CopyLeader] = []

    pool_addrs = sorted(
        tracks.keys(),
        key=lambda a: tracks[a].account_value,
        reverse=True,
    )[:scan]

    for addr in pool_addrs:
        tr = tracks[addr]
        fills = [f for f in tr.fills if f.time_ms <= now_ms]
        if not fills:
            continue
        recent = _stats_from_synth(fills, now_ms=now_ms, lookback_days=recent_d)
        history = _stats_from_synth(fills, now_ms=now_ms, lookback_days=hist_d)
        ok, why = passes_copy_filters(recent, history, cfg)
        if not ok:
            continue
        w = QualifiedWallet(
            address=addr,
            account_value=tr.account_value,
            rank_pnl=tr.rank_pnl,
            rank_roi=tr.rank_roi,
            rank_volume=0.0,
            confirm_pnl=0.0,
            score=0.0,
        )
        sc = score_copy_wallet(w, recent, history, cfg)
        ranked.append(
            CopyLeader(
                address=addr,
                score=sc,
                rank_roi=tr.rank_roi,
                rank_pnl=tr.rank_pnl,
                account_value=tr.account_value,
                recent=recent,
                history=history,
                reasons=[why],
            )
        )
    ranked.sort(key=lambda x: x.score, reverse=True)
    want = max(1, int(getattr(cfg, "COPY_TOP_N", 3) or 3))
    return ranked[:want]


def _targets_to_arrays(
    ds: ResearchDataset,
    targets_by_tick: list[list],
    *,
    max_slots: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    n = ds.n_ticks
    target_coin = np.full((n, max_slots), -1, dtype=np.int32)
    target_side = np.zeros((n, max_slots), dtype=np.int8)
    for ti, targets in enumerate(targets_by_tick):
        for si, t in enumerate(targets[:max_slots]):
            j = ds.coin_index.get(t.coin)
            if j is None:
                continue
            target_coin[ti, si] = j
            target_side[ti, si] = 1 if t.side == "long" else -1
    return target_coin, target_side


def run_copy_backtest(
    ds: ResearchDataset,
    cfg: Any,
    *,
    fee_rate: float = 0.0005,
    margin_frac: float = 0.30,
    leverage: float = 10.0,
    initial_equity: float = 10_000.0,
) -> CopyBacktestResult:
    """Rolling wallet reselect + mirror leaders; Numba PnL sim."""
    if ds.n_ticks < 3:
        return CopyBacktestResult(0, 0, 0, 0, 0, [], 0, ds.span_days)

    tracks = _build_wallet_tracks(ds)
    if not tracks:
        return CopyBacktestResult(0, 0, 0, 0, 0, [], 0, ds.span_days)

    reselect_s = float(getattr(cfg, "COPY_RESELECT_HOURS", 24.0) or 24.0) * 3600.0
    warmup_ticks = max(10, int(len(ds.ts) * 0.15))
    leaders: list[CopyLeader] = []
    last_pick = -1e18
    reselects = 0
    targets_by_tick: list[list] = []

    for ti in range(ds.n_ticks):
        ts = float(ds.ts[ti])
        if ti >= warmup_ticks and (ts - last_pick) >= reselect_s:
            leaders = _score_tracks_at_tick(tracks, ti, ds, cfg)
            last_pick = ts
            reselects += 1
        elif not leaders and ti == warmup_ticks:
            leaders = _score_tracks_at_tick(tracks, ti, ds, cfg)
            last_pick = ts
            reselects += 1

        snaps: list[WalletSnapshot] = []
        for ld in leaders:
            tr = tracks.get(ld.address)
            if tr is None:
                continue
            snap = tr.snaps_by_tick.get(ti)
            if snap is not None:
                snaps.append(snap)
        targets = copy_targets_from_leaders(leaders, snaps, cfg, now=ts) if leaders else []
        targets_by_tick.append(targets)

    max_slots = max(1, int(getattr(cfg, "COPY_MAX_POSITIONS", 3) or 3))
    target_coin, target_side = _targets_to_arrays(ds, targets_by_tick, max_slots=max_slots)
    tick_iv = float(ds.ts[1] - ds.ts[0]) if len(ds.ts) > 1 else 60.0
    cooldown = float(getattr(cfg, "REBALANCE_COOLDOWN_S", 180.0) or 180.0)
    sim = simulate_portfolio(
        ds.marks,
        target_coin,
        target_side,
        cooldown_s=cooldown,
        tick_interval_s=tick_iv,
        fee_rate=fee_rate,
        margin_frac=margin_frac,
        leverage=leverage,
        initial_equity=initial_equity,
        day_ids=ds.day_ids,
        max_slots=max_slots,
    )
    final_leaders = [ld.address for ld in leaders]
    return CopyBacktestResult(
        return_pct=float(sim["return_pct"]),
        max_dd_pct=float(sim["max_dd_pct"]),
        round_trips=int(sim["round_trips"]),
        win_rate_pct=float(sim["win_rate_pct"]),
        open_legs=int(sim["open_legs"]),
        leaders_picked=final_leaders,
        reselects=reselects,
        span_days=ds.span_days,
    )
