"""Shared signal path: backtest and live use the same kernels + spec."""

from __future__ import annotations

from typing import Any

import numpy as np

from .kernels import NAME_TO_ID, N_FEAT, build_score_targets, build_targets
from .panel import RankPanel, resample_closed


def compute_targets(panel: RankPanel, spec: dict[str, Any]) -> tuple:
    slots = max(1, int(spec.get("slots") or 5))
    top = int(spec.get("enter_top") or 0)
    if spec.get("engine") == "named":
        name = str(spec.get("name") or "top_k")
        sid = int(NAME_TO_ID[name])
        tc, ts, tw, tl = build_targets(
            panel.rank,
            panel.side,
            panel.wallets,
            panel.hold_pct,
            panel.agreement,
            panel.mean_leverage,
            sid,
            top,
            slots,
            int(spec.get("k") or 1),
            int(spec.get("lookback") or 1),
            float(spec.get("a") or 0),
            float(spec.get("b") or 0),
        )
        return tc, ts, apply_size_weights(panel, tc, tw, spec), tl
    w = np.asarray(spec.get("weights") or [0.0] * N_FEAT, dtype=np.float64)
    if w.shape[0] < N_FEAT:
        ww = np.zeros(N_FEAT, dtype=np.float64)
        ww[: w.shape[0]] = w
        w = ww
    tc, ts, tw, tl = build_score_targets(
        panel.rank,
        panel.side,
        panel.wallets,
        panel.hold_pct,
        panel.agreement,
        panel.mean_leverage,
        panel.conviction,
        panel.notional,
        panel.long_n,
        panel.short_n,
        panel.funding,
        panel.oi,
        panel.premium,
        panel.volume,
        panel.prev_day,
        panel.marks,
        w,
        top,
        slots,
        float(spec.get("enter_th") or 0),
        float(spec.get("exit_th") or 0),
        int(spec.get("lookback") or 4),
        int(spec.get("mode") or 0),
        int(spec.get("zscore") or 0),
        float(spec.get("min_agree") or 0),
        float(spec.get("min_wallets") or 0),
        float(spec.get("max_abs_funding") or 0),
        int(spec.get("require_improve") or 0),
    )
    return tc, ts, apply_size_weights(panel, tc, tw, spec), tl


def apply_size_weights(panel: RankPanel, tc: np.ndarray, tw: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Replace slot weights with the searched size rule (wallets / equal / rank / ...)."""
    mode = int(spec.get("size_mode") or 0)
    out = np.asarray(tw, dtype=np.float64).copy()
    if mode <= 0:
        out = np.maximum(out, 1.0)
        out[np.asarray(tc) < 0] = 0.0
        return out
    t_n, slots = tc.shape
    for t in range(t_n):
        for j in range(slots):
            c = int(tc[t, j])
            if c < 0 or c >= panel.n_coins:
                out[t, j] = 0.0
                continue
            if mode == 1:
                out[t, j] = 1.0
            elif mode == 2:
                r = float(panel.rank[t, c])
                out[t, j] = (1.0 / r) if r > 0 else 1.0
            elif mode == 3:
                out[t, j] = max(abs(float(panel.conviction[t, c])), 1e-6)
            elif mode == 4:
                out[t, j] = max(float(panel.agreement[t, c]), 1e-6)
            elif mode == 5:
                ntl = float(panel.notional[t, c])
                out[t, j] = np.log1p(ntl) if ntl > 0 else 1.0
            elif mode == 6:
                out[t, j] = max(float(panel.wallets[t, c]), 1.0) * max(float(panel.agreement[t, c]), 0.05)
            else:
                out[t, j] = max(float(tw[t, j]), 1.0)
    return out


def lagged_holdings(hourly: RankPanel, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Same names the simulator would hold at the last bar (next-bar fill)."""
    step = max(1, int(spec.get("step_h") or 1))
    panel = resample_closed(hourly, step)
    if panel.n_times < 2:
        return []
    tc, ts, tw, tl = compute_targets(panel, spec)
    lag = max(0, int(spec.get("exec_lag") or 1))
    t = panel.n_times - 1
    src = t - lag
    if src < 0:
        return []
    out: list[dict[str, Any]] = []
    slots = tc.shape[1]
    for j in range(slots):
        c = int(tc[src, j])
        if c < 0 or c >= len(panel.coins):
            continue
        sd = int(ts[src, j])
        if sd == 0:
            continue
        px = float(panel.marks[t, c])
        out.append(
            {
                "coin": panel.coins[c],
                "side": "long" if sd > 0 else "short",
                "wallets": float(tw[src, j] if tw[src, j] > 0 else 1.0),
                "lev": float(tl[src, j] if tl[src, j] >= 1.0 else 1.0),
                "px": px,
                "bar_unix": int(panel.cycle_unix[t]),
                "signal_unix": int(panel.cycle_unix[src]),
            }
        )
    return out
