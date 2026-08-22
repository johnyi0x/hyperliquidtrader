"""
Numba single-position simulator with optional TP/SL, signal exit, max-hold, DCA.

Priority each bar while in position:
  1) TP/SL (intrabar; SL first if both)
  2) Exit signal / profit snap at close
  3) Max-hold at close
DCA: while open, if adverse move from avg entry >= trigger, add size (up to max).
"""

from __future__ import annotations

import numpy as np
from numba import njit

SIM_INITIAL_EQUITY = 1_000.0
TAKER_FEE_PCT = 0.045


@njit(cache=True)
def _sim_core(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    entry_mask: np.ndarray,
    exit_mask: np.ndarray,
    side: int,
    tp_frac: float,
    sl_frac: float,
    leverage: float,
    bal_frac: float,
    taker: float,
    initial_equity: float,
    use_tpsl: int,
    use_exit_signal: int,
    exit_eid: int,
    exit_snap_frac: float,
    use_max_hold: int,
    max_hold_bars: int,
    dca_enabled: int,
    dca_trigger_frac: float,
    dca_max_adds: int,
    dca_size_mult: float,
) -> tuple:
    n = len(closes)
    equity = initial_equity
    peak = initial_equity
    max_dd = 0.0
    trades = 0
    wins = 0
    total_fees = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    in_pos = False
    entry_px = 0.0
    avg_px = 0.0
    size = 0.0
    base_size = 0.0
    tp_px = 0.0
    sl_px = 0.0
    entry_i = 0
    dca_adds = 0

    for i in range(n):
        c = closes[i]
        h = highs[i]
        lo = lows[i]
        if c <= 0.0 or h <= 0.0 or lo <= 0.0:
            continue

        if in_pos:
            # Optional DCA before exit checks (adverse from average).
            if (
                dca_enabled == 1
                and dca_adds < dca_max_adds
                and avg_px > 0.0
                and base_size > 0.0
            ):
                if side == 1:
                    adverse = (avg_px - c) / avg_px
                else:
                    adverse = (c - avg_px) / avg_px
                if adverse >= dca_trigger_frac:
                    # Equal-size legs: each DCA add matches the base entry size.
                    add_sz = base_size
                    add_fee = c * add_sz * taker
                    if equity > add_fee and add_sz > 0.0:
                        new_notional = avg_px * size + c * add_sz
                        size += add_sz
                        avg_px = new_notional / size
                        equity -= add_fee
                        total_fees += add_fee
                        dca_adds += 1
                        if use_tpsl == 1:
                            if side == 1:
                                tp_px = avg_px * (1.0 + tp_frac)
                                sl_px = avg_px * (1.0 - sl_frac)
                            else:
                                tp_px = avg_px * (1.0 - tp_frac)
                                sl_px = avg_px * (1.0 + sl_frac)

            exit_px = 0.0
            if use_tpsl == 1:
                hit_tp = False
                hit_sl = False
                if side == 1:
                    if lo <= sl_px:
                        hit_sl = True
                    if h >= tp_px:
                        hit_tp = True
                else:
                    if h >= sl_px:
                        hit_sl = True
                    if lo <= tp_px:
                        hit_tp = True
                if hit_sl:
                    exit_px = sl_px
                elif hit_tp:
                    exit_px = tp_px

            if exit_px <= 0.0 and use_exit_signal == 1:
                snap_hit = False
                if exit_eid == 2 and avg_px > 0.0:
                    move = (c - avg_px) / avg_px
                    if side == 1:
                        snap_hit = move >= exit_snap_frac
                    else:
                        snap_hit = move <= -exit_snap_frac
                elif exit_mask[i]:
                    snap_hit = True
                if snap_hit:
                    exit_px = c

            if exit_px <= 0.0 and use_max_hold == 1:
                if max_hold_bars > 0 and (i - entry_i) >= max_hold_bars:
                    exit_px = c

            if exit_px > 0.0:
                gross = (exit_px - avg_px) * size * side
                exit_fee = exit_px * size * taker
                net = gross - exit_fee
                equity += net
                total_fees += exit_fee
                trades += 1
                if net > 0.0:
                    wins += 1
                    gross_profit += net
                else:
                    gross_loss -= net
                if equity > peak:
                    peak = equity
                if peak > 0.0:
                    dd = (peak - equity) / peak * 100.0
                    if dd > max_dd:
                        max_dd = dd
                in_pos = False
            continue

        if equity <= 0.0:
            break
        if not entry_mask[i]:
            continue

        entry_px = c
        avg_px = c
        entry_i = i
        dca_adds = 0
        # balance_pct is TOTAL planned margin for entry + all DCA legs.
        n_legs = 1
        if dca_enabled == 1 and dca_max_adds > 0:
            n_legs = 1 + dca_max_adds
        leg_frac = bal_frac / float(n_legs)
        margin = equity * leg_frac * 0.97  # match live _order_margin_buffer
        notional = margin * leverage
        if notional < 10.0 or entry_px <= 0.0:
            continue
        size = notional / entry_px
        base_size = size
        entry_fee = entry_px * size * taker
        equity -= entry_fee
        total_fees += entry_fee
        if use_tpsl == 1:
            if side == 1:
                tp_px = entry_px * (1.0 + tp_frac)
                sl_px = entry_px * (1.0 - sl_frac)
            else:
                tp_px = entry_px * (1.0 - tp_frac)
                sl_px = entry_px * (1.0 + sl_frac)
        else:
            tp_px = 0.0
            sl_px = 0.0
        in_pos = True

    if in_pos:
        exit_px = closes[n - 1]
        gross = (exit_px - avg_px) * size * side
        exit_fee = exit_px * size * taker
        net = gross - exit_fee
        equity += net
        total_fees += exit_fee
        trades += 1
        if net > 0.0:
            wins += 1
            gross_profit += net
        else:
            gross_loss -= net
        if equity > peak:
            peak = equity
        if peak > 0.0:
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

    ret_pct = (equity - initial_equity) / initial_equity * 100.0
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    # Prefer return, penalize drawdown, small WR bonus; stability bias.
    score = ret_pct - 0.6 * max_dd + 0.12 * win_rate
    return (
        ret_pct,
        max_dd,
        float(trades),
        float(wins),
        total_fees,
        score,
        gross_profit,
        gross_loss,
        win_rate,
    )


def simulate(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    entry_mask: np.ndarray,
    exit_mask: np.ndarray,
    *,
    side: int,
    tp_pct: float,
    sl_pct: float,
    leverage: int,
    balance_pct: float,
    taker_fee_pct: float = TAKER_FEE_PCT,
    use_tpsl: bool = True,
    use_exit_signal: bool = False,
    exit_eid: int = -1,
    exit_snap_pct: float = 0.0,
    use_max_hold: bool = True,
    max_hold_bars: int = 48,
    dca_enabled: bool = False,
    dca_trigger_pct: float = 0.0,
    dca_max_adds: int = 0,
    dca_size_mult: float = 1.0,
) -> dict[str, float]:
    use_tpsl_i = 1 if use_tpsl else 0
    use_exit_i = 1 if use_exit_signal and exit_eid >= 0 else 0
    use_hold_i = 1 if use_max_hold else 0
    dca_i = 1 if dca_enabled and dca_max_adds > 0 else 0
    hold = max(1, int(max_hold_bars))
    taker = float(taker_fee_pct) / 100.0
    snap = float(exit_snap_pct) / 100.0 if exit_eid == 2 else 0.0
    xmask = (
        exit_mask.astype(np.bool_)
        if exit_mask is not None
        else np.zeros(len(closes), dtype=np.bool_)
    )

    # Warm JIT
    _ = _sim_core(
        closes[:2].astype(np.float64),
        highs[:2].astype(np.float64),
        lows[:2].astype(np.float64),
        np.zeros(2, dtype=np.bool_),
        np.zeros(2, dtype=np.bool_),
        1,
        0.01,
        0.01,
        1.0,
        0.9,
        taker,
        SIM_INITIAL_EQUITY,
        use_tpsl_i,
        use_exit_i,
        int(exit_eid),
        snap,
        use_hold_i,
        hold,
        dca_i,
        0.01,
        1,
        1.0,
    )
    core = _sim_core(
        closes.astype(np.float64),
        highs.astype(np.float64),
        lows.astype(np.float64),
        entry_mask.astype(np.bool_),
        xmask,
        int(side),
        float(tp_pct) / 100.0 if use_tpsl else 0.0,
        float(sl_pct) / 100.0 if use_tpsl else 0.0,
        float(max(1, leverage)),
        min(0.95, float(balance_pct) / 100.0),
        taker,
        SIM_INITIAL_EQUITY,
        use_tpsl_i,
        use_exit_i,
        int(exit_eid),
        snap,
        use_hold_i,
        hold,
        dca_i,
        float(dca_trigger_pct) / 100.0,
        int(dca_max_adds),
        float(dca_size_mult) if dca_size_mult > 0 else 1.0,
    )
    trades = int(core[2])
    pf = 999.0
    if core[7] > 1e-12:
        pf = core[6] / core[7]
    elif core[6] <= 0:
        pf = 0.0
    return {
        "return_pct": round(float(core[0]), 4),
        "max_dd_pct": round(float(core[1]), 4),
        "trades": trades,
        "wins": int(core[3]),
        "win_rate_pct": round(float(core[8]), 2),
        "fees_usd": round(float(core[4]), 4),
        "score": round(float(core[5]), 4),
        "profit_factor": round(float(pf), 3),
    }
