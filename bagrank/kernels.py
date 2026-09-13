"""Numba kernels: rank-as-price strategies + portfolio sim on 1h collector bars."""

from __future__ import annotations

import numpy as np
from numba import njit

FEE = 0.0005

SID_RANK_UP_DOWN = 0
SID_TOP_K = 1
SID_RANK_UP_K = 2
SID_PEAK_GIVEBACK = 3
SID_WALLETS_MOM = 4
SID_RANK_EMA = 5
SID_RANK_RSI = 6
SID_RANK_BREAKOUT = 7
SID_AGREE_MOM = 8
SID_WALLETS_EMA = 9

STRATEGY_NAMES = (
    "rank_up_down",
    "top_k",
    "rank_up_k",
    "peak_giveback",
    "wallets_mom",
    "rank_ema",
    "rank_rsi",
    "rank_breakout",
    "agree_mom",
    "wallets_ema",
)

NAME_TO_ID = {n: i for i, n in enumerate(STRATEGY_NAMES)}


@njit(cache=True)
def _ema_update(prev: float, x: float, alpha: float) -> float:
    if prev == 0.0:
        return x
    return alpha * x + (1.0 - alpha) * prev


@njit(cache=True)
def _rsi_update(avg_g: float, avg_l: float, delta: float, n: int) -> tuple:
    gain = delta if delta > 0.0 else 0.0
    loss = -delta if delta < 0.0 else 0.0
    if avg_g == 0.0 and avg_l == 0.0:
        return gain, loss
    ag = (avg_g * (n - 1) + gain) / n
    al = (avg_l * (n - 1) + loss) / n
    return ag, al


@njit(cache=True)
def _rsi_value(avg_g: float, avg_l: float) -> float:
    if avg_l <= 1e-12:
        return 100.0 if avg_g > 0.0 else 50.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


@njit(cache=True)
def _pick_slots(
    rank: np.ndarray,
    side: np.ndarray,
    wallets: np.ndarray,
    want: np.ndarray,
    enter_top: int,
    max_slots: int,
    out_coin: np.ndarray,
    out_side: np.ndarray,
    out_wallets: np.ndarray,
    out_lev: np.ndarray,
    mean_lev: np.ndarray,
) -> None:
    n_coins = rank.shape[0]
    out_coin[:] = -1
    out_side[:] = 0
    out_wallets[:] = 0
    out_lev[:] = 1.0
    # Greedy: lowest rank among wanted names. enter_top<=0 means the whole board.
    cap = enter_top if enter_top > 0 else 1000000
    taken = 0
    while taken < max_slots:
        best_c = -1
        best_r = 10**9
        for c in range(n_coins):
            if not want[c]:
                continue
            r = int(rank[c])
            if r <= 0 or r > cap:
                continue
            already = False
            for k in range(taken):
                if out_coin[k] == c:
                    already = True
                    break
            if already:
                continue
            if r < best_r:
                best_r = r
                best_c = c
        if best_c < 0:
            break
        out_coin[taken] = best_c
        sd = int(side[best_c])
        out_side[taken] = 1 if sd >= 0 else -1
        out_wallets[taken] = int(wallets[best_c]) if wallets[best_c] > 0 else 1
        lev = float(mean_lev[best_c])
        if lev < 1.0:
            lev = 1.0
        out_lev[taken] = lev
        taken += 1


@njit(cache=True)
def build_targets(
    rank: np.ndarray,
    side: np.ndarray,
    wallets: np.ndarray,
    hold_pct: np.ndarray,
    agreement: np.ndarray,
    mean_lev: np.ndarray,
    strategy_id: int,
    enter_top: int,
    max_slots: int,
    param_k: int,
    param_lookback: int,
    param_a: float,
    param_b: float,
) -> tuple:
    """Return target_coin, target_side, target_wallets, target_lev (T, max_slots)."""
    t_n, c_n = rank.shape
    target_coin = np.full((t_n, max_slots), -1, dtype=np.int32)
    target_side = np.zeros((t_n, max_slots), dtype=np.int8)
    target_wallets = np.zeros((t_n, max_slots), dtype=np.int32)
    target_lev = np.ones((t_n, max_slots), dtype=np.float64)
    want = np.zeros(c_n, dtype=np.bool_)
    holding = np.zeros(c_n, dtype=np.bool_)
    best_rank = np.full(c_n, 10**9, dtype=np.int32)
    ema_fast = np.zeros(c_n, dtype=np.float64)
    ema_slow = np.zeros(c_n, dtype=np.float64)
    ema_w = np.zeros(c_n, dtype=np.float64)
    rsi_g = np.zeros(c_n, dtype=np.float64)
    rsi_l = np.zeros(c_n, dtype=np.float64)
    prev_px = np.zeros(c_n, dtype=np.float64)
    rsi_n = param_k if param_k > 1 else 14
    look = param_lookback if param_lookback > 0 else 6
    a_fast = 2.0 / (param_a + 1.0) if param_a > 1.0 else 2.0 / 4.0
    a_slow = 2.0 / (param_b + 1.0) if param_b > 1.0 else 2.0 / 12.0
    k_up = param_k if param_k > 0 else 2
    giveback = param_k if param_k > 0 else 2
    cap = enter_top if enter_top > 0 else 1000000

    for t in range(t_n):
        want[:] = False
        for c in range(c_n):
            r = int(rank[t, c])
            inv = (1.0 / float(r)) if r > 0 else 0.0
            w = float(wallets[t, c])
            agr = float(agreement[t, c])
            prev_r = int(rank[t - 1, c]) if t > 0 else 0
            ref_i = t - look if look > 1 else t - 1
            if look > 1 and ref_i >= 0:
                prev_r = int(rank[ref_i, c])
            old = prev_r if prev_r > 0 else (cap + 8)
            on_board = r > 0 and r <= cap

            if strategy_id == SID_RANK_EMA or strategy_id == SID_RANK_RSI or strategy_id == SID_RANK_BREAKOUT:
                if inv > 0.0:
                    ema_fast[c] = _ema_update(ema_fast[c], inv, a_fast)
                    ema_slow[c] = _ema_update(ema_slow[c], inv, a_slow)
                    if prev_px[c] != 0.0:
                        dg, dl = _rsi_update(rsi_g[c], rsi_l[c], inv - prev_px[c], rsi_n)
                        rsi_g[c] = dg
                        rsi_l[c] = dl
                    prev_px[c] = inv
            if strategy_id == SID_WALLETS_EMA and w > 0.0:
                ema_w[c] = _ema_update(ema_w[c], w, a_fast)

            if t == 0:
                continue

            if strategy_id == SID_RANK_UP_DOWN:
                if holding[c]:
                    if r <= 0 or r > prev_r:
                        holding[c] = False
                    else:
                        want[c] = True
                elif on_board and r < old:
                    holding[c] = True
                    want[c] = True
            elif strategy_id == SID_TOP_K:
                want[c] = on_board
            elif strategy_id == SID_RANK_UP_K:
                improved = old - r
                if holding[c]:
                    if r <= 0 or r > prev_r:
                        holding[c] = False
                    else:
                        want[c] = True
                elif on_board and improved >= k_up:
                    holding[c] = True
                    want[c] = True
            elif strategy_id == SID_PEAK_GIVEBACK:
                if holding[c]:
                    if r > 0 and r < best_rank[c]:
                        best_rank[c] = r
                    if r <= 0 or (r - best_rank[c]) >= giveback:
                        holding[c] = False
                        best_rank[c] = 10**9
                    else:
                        want[c] = True
                elif on_board and r < old:
                    holding[c] = True
                    best_rank[c] = r
                    want[c] = True
            elif strategy_id == SID_WALLETS_MOM:
                prev_w = float(wallets[t - 1, c]) if t > 0 else 0.0
                rising = on_board and w > prev_w
                falling = w < prev_w or r <= 0
                if holding[c]:
                    if falling:
                        holding[c] = False
                    else:
                        want[c] = True
                elif rising:
                    holding[c] = True
                    want[c] = True
            elif strategy_id == SID_RANK_EMA:
                if holding[c]:
                    if r <= 0 or ema_fast[c] < ema_slow[c]:
                        holding[c] = False
                    else:
                        want[c] = True
                elif on_board and ema_fast[c] > ema_slow[c] and inv > 0.0:
                    holding[c] = True
                    want[c] = True
            elif strategy_id == SID_RANK_RSI:
                rsi = _rsi_value(rsi_g[c], rsi_l[c])
                if holding[c]:
                    if r <= 0 or rsi >= param_b:
                        holding[c] = False
                    else:
                        want[c] = True
                elif on_board and rsi <= param_a and rsi > 0.0:
                    holding[c] = True
                    want[c] = True
            elif strategy_id == SID_RANK_BREAKOUT:
                hi = 0.0
                start = t - look
                if start < 0:
                    start = 0
                for u in range(start, t):
                    ru = int(rank[u, c])
                    if ru > 0:
                        px = 1.0 / float(ru)
                        if px > hi:
                            hi = px
                if holding[c]:
                    if r <= 0 or (inv < hi and t > look):
                        holding[c] = False
                    else:
                        want[c] = True
                elif on_board and inv >= hi and hi > 0.0:
                    holding[c] = True
                    want[c] = True
            elif strategy_id == SID_AGREE_MOM:
                prev_a = float(agreement[t - 1, c]) if t > 0 else 0.0
                if holding[c]:
                    if r <= 0 or r > prev_r:
                        holding[c] = False
                    else:
                        want[c] = True
                elif on_board and r < old and agr >= prev_a:
                    holding[c] = True
                    want[c] = True
            elif strategy_id == SID_WALLETS_EMA:
                if holding[c]:
                    if r <= 0 or w < ema_w[c]:
                        holding[c] = False
                    else:
                        want[c] = True
                elif on_board and w > ema_w[c] and ema_w[c] > 0.0:
                    holding[c] = True
                    want[c] = True

        _pick_slots(
            rank[t],
            side[t],
            wallets[t],
            want,
            enter_top,
            max_slots,
            target_coin[t],
            target_side[t],
            target_wallets[t],
            target_lev[t],
            mean_lev[t],
        )
        # unused hold_pct keeps signature stable for future filters
        _ = hold_pct[t, 0] if c_n > 0 else 0.0
    return target_coin, target_side, target_wallets, target_lev


@njit(cache=True)
def _leg_pnl(entry_px: float, exit_px: float, side: int, notional: float) -> float:
    if entry_px <= 0.0 or exit_px <= 0.0 or notional <= 0.0:
        return 0.0
    ret = (exit_px - entry_px) / entry_px
    if side < 0:
        ret = -ret
    return notional * ret


@njit(cache=True)
def simulate_nb(
    marks: np.ndarray,
    target_coin: np.ndarray,
    target_side: np.ndarray,
    target_wallets: np.ndarray,
    target_lev: np.ndarray,
    fee_rate: float,
    gross_frac: float,
    initial_equity: float,
    max_slots: int,
    max_pair_share: float,
    exec_lag: int,
    min_hold: int,
    use_lev: int,
    bar_hours: float,
    exposure_mode: int = 0,
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
    slot_age = np.zeros(max_slots, dtype=np.int32)
    round_trips = 0
    wins = 0
    total_fees = 0.0
    hold_hours_sum = 0.0
    equity_path = np.zeros(n_ticks, dtype=np.float64)
    lag = exec_lag if exec_lag > 0 else 0
    hold_need = min_hold if min_hold > 0 else 0

    for ti in range(n_ticks):
        for s in range(max_slots):
            if slot_coin[s] >= 0:
                slot_age[s] += 1
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
        equity_path[ti] = mtm

        desired_c = np.full(max_slots, -1, dtype=np.int32)
        desired_s = np.zeros(max_slots, dtype=np.int8)
        desired_w = np.zeros(max_slots, dtype=np.float64)
        desired_l = np.ones(max_slots, dtype=np.float64)
        d = 0
        wsum = 0.0
        src = ti - lag
        if src >= 0:
            for j in range(max_slots):
                c = int(target_coin[src, j])
                if c < 0 or c >= n_coins:
                    continue
                sd = int(target_side[src, j])
                if sd == 0:
                    continue
                desired_c[d] = c
                desired_s[d] = 1 if sd > 0 else -1
                ww = float(target_wallets[src, j])
                if ww < 1.0:
                    ww = 1.0
                desired_w[d] = ww
                lv = 1.0
                if use_lev != 0:
                    lv = float(target_lev[src, j])
                    if lv < 1.0:
                        lv = 1.0
                desired_l[d] = lv
                wsum += ww
                d += 1

        for s in range(max_slots):
            c = slot_coin[s]
            if c < 0:
                continue
            keep = False
            for j in range(d):
                if desired_c[j] == c and desired_s[j] == slot_side[s]:
                    keep = True
                    break
            if keep:
                continue
            if hold_need > 0 and slot_age[s] < hold_need:
                continue
            px = marks[ti, c]
            if px > 0.0 and slot_notional[s] > 0.0:
                fee = slot_notional[s] * fee_rate
                pnl = _leg_pnl(slot_entry[s], px, int(slot_side[s]), slot_notional[s]) - fee
                cash += pnl
                total_fees += fee
                hold_hours_sum += float(slot_age[s]) * bar_hours
                if pnl >= 0.0:
                    wins += 1
                round_trips += 1
            slot_coin[s] = -1
            slot_side[s] = 0
            slot_entry[s] = 0.0
            slot_notional[s] = 0.0
            slot_age[s] = 0

        if wsum <= 0.0:
            continue
        n_des = d
        g_use = gross_frac
        if exposure_mode == 1 and max_slots > 0:
            g_use = gross_frac * (float(n_des) / float(max_slots))
        elif exposure_mode == 2 and max_slots > 0:
            g_use = gross_frac * np.sqrt(float(n_des) / float(max_slots))
        if g_use < 0.05:
            g_use = 0.05
        if g_use > 2.0:
            g_use = 2.0
        for j in range(d):
            c = desired_c[j]
            sd = desired_s[j]
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
            weight = desired_w[j] / wsum
            if max_pair_share > 0.0 and weight > max_pair_share:
                weight = max_pair_share
            notional = cash * g_use * weight * desired_l[j]
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
            slot_age[free] = 0

    ti = n_ticks - 1 if n_ticks > 0 else 0
    for s in range(max_slots):
        c = slot_coin[s]
        if c < 0:
            continue
        px = marks[ti, c]
        if px > 0.0 and slot_notional[s] > 0.0:
            fee = slot_notional[s] * fee_rate
            pnl = _leg_pnl(slot_entry[s], px, int(slot_side[s]), slot_notional[s]) - fee
            cash += pnl
            total_fees += fee
            hold_hours_sum += float(slot_age[s]) * bar_hours
            if pnl >= 0.0:
                wins += 1
            round_trips += 1
        slot_coin[s] = -1

    ret_pct = (cash / initial_equity - 1.0) * 100.0 if initial_equity > 0 else 0.0
    win_rate = (wins / round_trips * 100.0) if round_trips > 0 else 0.0
    avg_hold = (hold_hours_sum / float(round_trips)) if round_trips > 0 else 0.0
    return ret_pct, max_dd * 100.0, round_trips, win_rate, total_fees, cash, equity_path, avg_hold


def simulate(
    marks: np.ndarray,
    target_coin: np.ndarray,
    target_side: np.ndarray,
    target_wallets: np.ndarray,
    target_lev: np.ndarray,
    fee_rate: float,
    gross_frac: float,
    initial_equity: float,
    max_slots: int,
    max_pair_share: float,
    exec_lag: int = 0,
    min_hold: int = 0,
    use_lev: int = 1,
    bar_hours: float = 1.0,
    exposure_mode: int = 0,
) -> tuple:
    return simulate_nb(
        marks,
        target_coin,
        target_side,
        target_wallets,
        target_lev,
        fee_rate,
        gross_frac,
        initial_equity,
        max_slots,
        max_pair_share,
        int(exec_lag),
        int(min_hold),
        int(use_lev),
        float(bar_hours),
        int(exposure_mode),
    )


N_FEAT = 22
FEATURE_NAMES = (
    "rank_inv",
    "d_rank",
    "wallets",
    "d_wallets",
    "hold_pct",
    "agreement",
    "d_agree",
    "leverage",
    "conviction",
    "notional",
    "funding",
    "d_oi",
    "premium",
    "ret_1",
    "vs_prev_day",
    "d_volume",
    "ls_imb",
    "rank_accel",
    "w_accel",
    "hold_gap",
    "fund_carry",
    "crowd",
)

MODE_TOP = 0
MODE_THRESH = 1
MODE_FADE = 2
MODE_STREAK = 3
MODE_BREAKOUT = 4
MODE_GIVEBACK = 5
MODE_FLIP = 6
MODE_DIVERGE = 7
MODE_CROWD = 8
MODE_CARRY = 9


@njit(cache=True)
def _pick_by_score(
    score: np.ndarray,
    rank: np.ndarray,
    side: np.ndarray,
    wallets: np.ndarray,
    want: np.ndarray,
    enter_top: int,
    max_slots: int,
    fade: int,
    flip: int,
    mean_lev: np.ndarray,
    out_coin: np.ndarray,
    out_side: np.ndarray,
    out_wallets: np.ndarray,
    out_lev: np.ndarray,
) -> None:
    n_coins = rank.shape[0]
    out_coin[:] = -1
    out_side[:] = 0
    out_wallets[:] = 0
    out_lev[:] = 1.0
    cap = enter_top if enter_top > 0 else 1000000
    taken = 0
    while taken < max_slots:
        best_c = -1
        best = -1.0e300 if fade == 0 else 1.0e300
        for c in range(n_coins):
            if not want[c]:
                continue
            r = int(rank[c])
            if r <= 0 or r > cap:
                continue
            already = False
            for k in range(taken):
                if out_coin[k] == c:
                    already = True
                    break
            if already:
                continue
            sc = float(score[c])
            if fade == 0:
                if sc > best:
                    best = sc
                    best_c = c
            elif sc < best:
                best = sc
                best_c = c
        if best_c < 0:
            break
        out_coin[taken] = best_c
        sd = int(side[best_c])
        signed = 1 if sd >= 0 else -1
        if flip != 0:
            signed = -signed
        out_side[taken] = signed
        out_wallets[taken] = int(wallets[best_c]) if wallets[best_c] > 0 else 1
        lev = float(mean_lev[best_c])
        if lev < 1.0:
            lev = 1.0
        out_lev[taken] = lev
        taken += 1


@njit(cache=True)
def build_score_targets(
    rank: np.ndarray,
    side: np.ndarray,
    wallets: np.ndarray,
    hold_pct: np.ndarray,
    agreement: np.ndarray,
    mean_lev: np.ndarray,
    conviction: np.ndarray,
    notional: np.ndarray,
    long_n: np.ndarray,
    short_n: np.ndarray,
    funding: np.ndarray,
    oi: np.ndarray,
    premium: np.ndarray,
    volume: np.ndarray,
    prev_day: np.ndarray,
    marks: np.ndarray,
    weights: np.ndarray,
    enter_top: int,
    max_slots: int,
    enter_th: float,
    exit_th: float,
    lookback: int,
    mode: int,
    zscore: int,
    min_agree: float,
    min_wallets: float,
    max_abs_funding: float,
    require_improve: int,
) -> tuple:
    t_n, c_n = rank.shape
    target_coin = np.full((t_n, max_slots), -1, dtype=np.int32)
    target_side = np.zeros((t_n, max_slots), dtype=np.int8)
    target_wallets = np.zeros((t_n, max_slots), dtype=np.int32)
    target_lev = np.ones((t_n, max_slots), dtype=np.float64)
    want = np.zeros(c_n, dtype=np.bool_)
    score = np.zeros(c_n, dtype=np.float64)
    feat = np.zeros(N_FEAT, dtype=np.float64)
    holding = np.zeros(c_n, dtype=np.bool_)
    peak_sc = np.zeros(c_n, dtype=np.float64)
    streak = np.zeros(c_n, dtype=np.int32)
    look = lookback if lookback > 1 else 2
    fade = 1 if mode == MODE_FADE else 0
    flip = 1 if mode == MODE_FLIP else 0
    cap = enter_top if enter_top > 0 else 1000000

    for t in range(t_n):
        want[:] = False
        score[:] = 0.0
        n_on = 0
        sum_sc = 0.0
        for c in range(c_n):
            r = int(rank[t, c])
            on = r > 0 and r <= cap
            prev_r = int(rank[t - 1, c]) if t > 0 else 0
            d_rank = float(prev_r - r) if prev_r > 0 and r > 0 else 0.0
            w = float(wallets[t, c])
            prev_w = float(wallets[t - 1, c]) if t > 0 else 0.0
            agr = float(agreement[t, c])
            prev_a = float(agreement[t - 1, c]) if t > 0 else 0.0
            px = float(marks[t, c])
            prev_px = float(marks[t - 1, c]) if t > 0 else 0.0
            oi_now = float(oi[t, c])
            oi_prev = float(oi[t - 1, c]) if t > 0 else 0.0
            vol = float(volume[t, c])
            vol_prev = float(volume[t - 1, c]) if t > 0 else 0.0
            pday = float(prev_day[t, c])
            ntl = float(notional[t, c])
            ln = float(long_n[t, c])
            sn = float(short_n[t, c])
            if d_rank > 0.0:
                streak[c] += 1
            else:
                streak[c] = 0
            feat[0] = (1.0 / float(r)) if r > 0 else 0.0
            feat[1] = d_rank
            feat[2] = w / 100.0
            feat[3] = w - prev_w
            feat[4] = float(hold_pct[t, c])
            feat[5] = agr
            feat[6] = agr - prev_a
            feat[7] = float(mean_lev[t, c]) / 10.0
            feat[8] = float(conviction[t, c])
            feat[9] = np.log1p(ntl) if ntl > 0.0 else 0.0
            feat[10] = float(funding[t, c]) * 100.0
            feat[11] = (oi_now - oi_prev) / (oi_prev + 1.0) if oi_prev > 0.0 else 0.0
            feat[12] = float(premium[t, c])
            feat[13] = (px - prev_px) / prev_px if prev_px > 0.0 and px > 0.0 else 0.0
            feat[14] = (px - pday) / pday if pday > 0.0 and px > 0.0 else 0.0
            feat[15] = (vol - vol_prev) / (vol_prev + 1.0) if vol_prev > 0.0 else 0.0
            feat[16] = (ln - sn) / (ln + sn + 1.0)
            d_rank_prev = 0.0
            if t > 1:
                prev2_r = int(rank[t - 2, c])
                if prev2_r > 0 and prev_r > 0:
                    d_rank_prev = float(prev2_r - prev_r)
            feat[17] = d_rank - d_rank_prev
            prev2_w = float(wallets[t - 2, c]) if t > 1 else 0.0
            feat[18] = (w - prev_w) - (prev_w - prev2_w)
            feat[19] = float(hold_pct[t, c]) - agr
            sd_sign = 1.0 if int(side[t, c]) >= 0 else -1.0
            feat[20] = float(funding[t, c]) * 100.0 * sd_sign
            feat[21] = feat[2] * feat[0]
            sc = 0.0
            for k in range(N_FEAT):
                sc += weights[k] * feat[k]
            if not on:
                sc = 0.0
                holding[c] = False
                peak_sc[c] = 0.0
            score[c] = sc
            if on:
                n_on += 1
                sum_sc += sc

        if zscore != 0 and n_on > 2:
            mean = sum_sc / float(n_on)
            var = 0.0
            for c in range(c_n):
                r = int(rank[t, c])
                if r > 0 and r <= cap:
                    d = score[c] - mean
                    var += d * d
            sd = np.sqrt(var / float(n_on))
            if sd > 1e-12:
                for c in range(c_n):
                    r = int(rank[t, c])
                    if r > 0 and r <= cap:
                        score[c] = (score[c] - mean) / sd

        if t == 0:
            _pick_by_score(
                score, rank[t], side[t], wallets[t], want, enter_top, max_slots,
                fade, flip, mean_lev[t],
                target_coin[t], target_side[t], target_wallets[t], target_lev[t],
            )
            continue

        for c in range(c_n):
            r = int(rank[t, c])
            on = r > 0 and r <= cap
            if not on:
                continue
            if min_agree > 0.0 and float(agreement[t, c]) < min_agree:
                continue
            if min_wallets > 0.0 and float(wallets[t, c]) < min_wallets:
                continue
            if mode != MODE_CARRY and max_abs_funding > 0.0 and abs(float(funding[t, c])) > max_abs_funding:
                continue
            prev_r = int(rank[t - 1, c]) if t > 0 else 0
            improved = prev_r > 0 and r < prev_r
            if require_improve != 0 and not improved:
                if holding[c]:
                    pass
                else:
                    continue
            sc = score[c]
            if mode == MODE_TOP or mode == MODE_FLIP or mode == MODE_FADE:
                want[c] = True
            elif mode == MODE_THRESH:
                if holding[c]:
                    if sc <= exit_th:
                        holding[c] = False
                    else:
                        want[c] = True
                elif sc >= enter_th:
                    holding[c] = True
                    want[c] = True
            elif mode == MODE_STREAK:
                if streak[c] >= look:
                    want[c] = True
                    holding[c] = True
                elif holding[c] and streak[c] > 0:
                    want[c] = True
                else:
                    holding[c] = False
            elif mode == MODE_BREAKOUT:
                hi = -1.0e300
                start = t - look
                if start < 0:
                    start = 0
                for u in range(start, t):
                    ru = int(rank[u, c])
                    if ru > 0:
                        # reuse stored? approximate with rank_inv history via current weights on rank only
                        inv = 1.0 / float(ru)
                        if inv > hi:
                            hi = inv
                inv_now = 1.0 / float(r)
                if holding[c]:
                    if inv_now < hi:
                        holding[c] = False
                    else:
                        want[c] = True
                elif inv_now >= hi and hi > 0.0 and sc >= enter_th:
                    holding[c] = True
                    want[c] = True
            elif mode == MODE_GIVEBACK:
                if holding[c]:
                    if sc > peak_sc[c]:
                        peak_sc[c] = sc
                    if sc <= peak_sc[c] - exit_th:
                        holding[c] = False
                        peak_sc[c] = 0.0
                    else:
                        want[c] = True
                elif sc >= enter_th:
                    holding[c] = True
                    peak_sc[c] = sc
                    want[c] = True
            elif mode == MODE_DIVERGE:
                prev_px = float(marks[t - 1, c]) if t > 0 else 0.0
                px = float(marks[t, c])
                ret1 = (px - prev_px) / prev_px if prev_px > 0.0 and px > 0.0 else 0.0
                prev_r = int(rank[t - 1, c]) if t > 0 else 0
                d_rank = float(prev_r - r) if prev_r > 0 and r > 0 else 0.0
                if (d_rank > 0.0 and ret1 < 0.0) or (d_rank < 0.0 and ret1 > 0.0):
                    want[c] = True
                    holding[c] = True
                else:
                    holding[c] = False
            elif mode == MODE_CROWD:
                wnow = float(wallets[t, c])
                crowd_min = min_wallets if min_wallets > 0.0 else 12.0
                if r <= 3 and wnow >= crowd_min:
                    want[c] = True
                    holding[c] = True
                elif holding[c] and r <= 5:
                    want[c] = True
                else:
                    holding[c] = False
            elif mode == MODE_CARRY:
                fund = abs(float(funding[t, c]))
                need = max_abs_funding if max_abs_funding > 0.0 else 0.00005
                if fund >= need:
                    want[c] = True
                    holding[c] = True
                else:
                    holding[c] = False
            else:
                want[c] = sc >= enter_th

        _pick_by_score(
            score,
            rank[t],
            side[t],
            wallets[t],
            want,
            enter_top,
            max_slots,
            fade,
            flip,
            mean_lev[t],
            target_coin[t],
            target_side[t],
            target_wallets[t],
            target_lev[t],
        )
        _ = long_n[t, 0] if c_n > 0 else 0.0
        _ = short_n[t, 0] if c_n > 0 else 0.0
    return target_coin, target_side, target_wallets, target_lev

