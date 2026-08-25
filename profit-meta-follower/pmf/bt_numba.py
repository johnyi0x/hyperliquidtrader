"""Numba portfolio simulator for crowd-following backtests."""

from __future__ import annotations

import numpy as np
from numba import njit

PMF_FEE_PCT = 0.0005  # 0.05% per side


@njit(cache=True)
def _leg_pnl(entry_px: float, exit_px: float, side: int, notional: float) -> float:
    if entry_px <= 0.0 or exit_px <= 0.0 or notional <= 0.0:
        return 0.0
    ret = (exit_px - entry_px) / entry_px
    if side < 0:
        ret = -ret
    return notional * ret


@njit(cache=True)
def _sim_portfolio_core(
    marks: np.ndarray,
    target_coin: np.ndarray,
    target_side: np.ndarray,
    cooldown_ticks: int,
    fee_rate: float,
    margin_frac: float,
    leverage: float,
    initial_equity: float,
    day_ids: np.ndarray,
    max_slots: int,
) -> tuple:
    n_ticks = marks.shape[0]
    n_coins = marks.shape[1]
    cash = initial_equity
    peak = initial_equity
    max_dd = 0.0

    slot_coin = np.full(max_slots, -1, dtype=np.int32)
    slot_side = np.zeros(max_slots, dtype=np.int8)
    slot_entry = np.zeros(max_slots, dtype=np.float64)
    slot_notional = np.zeros(max_slots, dtype=np.float64)

    round_trips = 0
    wins = 0
    total_fees = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    last_rebalance = -cooldown_ticks - 1

    n_days = 0
    if len(day_ids) > 0:
        n_days = int(day_ids[-1]) + 1
    trades_by_day = np.zeros(max(n_days, 1), dtype=np.int32)

    for ti in range(n_ticks):
        # Mark-to-market for drawdown (does not mutate cash).
        mtm = cash
        for s in range(max_slots):
            c = slot_coin[s]
            if c < 0:
                continue
            px = marks[ti, c]
            if px > 0.0 and slot_entry[s] > 0.0:
                mtm += _leg_pnl(slot_entry[s], px, int(slot_side[s]), slot_notional[s])
        if mtm > peak:
            peak = mtm
        if peak > 0.0:
            dd = (peak - mtm) / peak
            if dd > max_dd:
                max_dd = dd

        can_rebalance = (ti - last_rebalance) >= cooldown_ticks
        desired_coins = np.full(max_slots, -1, dtype=np.int32)
        desired_side = np.zeros(max_slots, dtype=np.int8)
        d = 0
        for j in range(max_slots):
            c = int(target_coin[ti, j])
            if c < 0 or c >= n_coins:
                continue
            sd = int(target_side[ti, j])
            if sd == 0:
                continue
            desired_coins[d] = c
            desired_side[d] = 1 if sd > 0 else -1
            d += 1
            if d >= max_slots:
                break

        # Exits (always allowed).
        for s in range(max_slots):
            c = slot_coin[s]
            if c < 0:
                continue
            keep = False
            for j in range(d):
                if desired_coins[j] == c and desired_side[j] == slot_side[s]:
                    keep = True
                    break
            if keep:
                continue
            px = marks[ti, c]
            if px > 0.0 and slot_notional[s] > 0.0:
                fee = slot_notional[s] * fee_rate
                pnl = _leg_pnl(slot_entry[s], px, int(slot_side[s]), slot_notional[s]) - fee
                cash += pnl
                total_fees += fee
                if pnl >= 0.0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss += -pnl
                round_trips += 1
                if len(day_ids) > ti:
                    di = int(day_ids[ti])
                    if 0 <= di < len(trades_by_day):
                        trades_by_day[di] += 1
            slot_coin[s] = -1
            slot_side[s] = 0
            slot_entry[s] = 0.0
            slot_notional[s] = 0.0

        if not can_rebalance:
            continue

        opened = False
        for j in range(d):
            c = desired_coins[j]
            sd = desired_side[j]
            already = False
            for s in range(max_slots):
                if slot_coin[s] == c and slot_side[s] == sd:
                    already = True
                    break
            if already:
                continue
            free = -1
            for s in range(max_slots):
                if slot_coin[s] < 0:
                    free = s
                    break
            if free < 0:
                continue
            px = marks[ti, c]
            if not (px > 0.0):
                continue
            notional = cash * margin_frac * leverage
            if notional <= 0.0:
                continue
            fee = notional * fee_rate
            if cash <= fee:
                continue
            cash -= fee
            total_fees += fee
            slot_coin[free] = c
            slot_side[free] = sd
            slot_entry[free] = px
            slot_notional[free] = notional
            opened = True
        if opened:
            last_rebalance = ti

    # Force-close remaining legs at last mark so every strategy ends flat and
    # open-and-hold still gets a closed-trip PnL for ranking.
    ti = n_ticks - 1 if n_ticks > 0 else 0
    open_legs = 0
    for s in range(max_slots):
        c = slot_coin[s]
        if c < 0:
            continue
        open_legs += 1
        px = marks[ti, c]
        if px > 0.0 and slot_notional[s] > 0.0:
            fee = slot_notional[s] * fee_rate
            pnl = _leg_pnl(slot_entry[s], px, int(slot_side[s]), slot_notional[s]) - fee
            cash += pnl
            total_fees += fee
            if pnl >= 0.0:
                wins += 1
                gross_profit += pnl
            else:
                gross_loss += -pnl
            round_trips += 1
            if len(day_ids) > ti:
                di = int(day_ids[ti])
                if 0 <= di < len(trades_by_day):
                    trades_by_day[di] += 1
        slot_coin[s] = -1

    ret_pct = (cash / initial_equity - 1.0) * 100.0 if initial_equity > 0 else 0.0
    win_rate = (wins / round_trips * 100.0) if round_trips > 0 else 0.0
    return (
        ret_pct,
        max_dd * 100.0,
        round_trips,
        win_rate,
        total_fees,
        gross_profit,
        gross_loss,
        trades_by_day,
        open_legs,
    )


def simulate_portfolio(
    marks: np.ndarray,
    target_coin: np.ndarray,
    target_side: np.ndarray,
    *,
    cooldown_s: float,
    tick_interval_s: float,
    fee_rate: float = PMF_FEE_PCT,
    margin_frac: float = 0.30,
    leverage: float = 10.0,
    initial_equity: float = 10_000.0,
    day_ids: np.ndarray | None = None,
    max_slots: int = 3,
) -> dict[str, float | int | np.ndarray]:
    if marks.ndim != 2:
        raise ValueError("marks must be 2D")
    n_ticks = marks.shape[0]
    if target_coin.shape[0] != n_ticks or target_side.shape[0] != n_ticks:
        raise ValueError("target arrays must match marks rows")
    cooldown_ticks = max(1, int(round(float(cooldown_s) / max(float(tick_interval_s), 1.0))))
    if day_ids is None:
        day_ids = np.zeros(n_ticks, dtype=np.int32)
    out = _sim_portfolio_core(
        np.ascontiguousarray(marks, dtype=np.float64),
        np.ascontiguousarray(target_coin, dtype=np.int32),
        np.ascontiguousarray(target_side, dtype=np.int8),
        cooldown_ticks,
        float(fee_rate),
        float(margin_frac),
        float(leverage),
        float(initial_equity),
        np.ascontiguousarray(day_ids, dtype=np.int32),
        int(max_slots),
    )
    return {
        "return_pct": float(out[0]),
        "max_dd_pct": float(out[1]),
        "round_trips": int(out[2]),
        "win_rate_pct": float(out[3]),
        "total_fees": float(out[4]),
        "gross_profit": float(out[5]),
        "gross_loss": float(out[6]),
        "trades_by_day": out[7],
        "open_legs": int(out[8]),
    }
