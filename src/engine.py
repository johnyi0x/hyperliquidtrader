"""
Shared live/backtest decision helpers.

Live and paper must call these so entries/exits/DCA match the Numba sim rules
(closed-bar signals, same side convention, same adverse % math).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .indicators import build_features
from .mtf import mtf_entry_signal_now
from .registry import build_entry_mask, build_exit_mask


def candles_to_feats(candles: list[dict]) -> dict[str, np.ndarray] | None:
    if len(candles) < 40:
        return None
    closes = np.array([float(c["c"]) for c in candles], dtype=np.float64)
    highs = np.array([float(c["h"]) for c in candles], dtype=np.float64)
    lows = np.array([float(c["l"]) for c in candles], dtype=np.float64)
    vols = np.array([float(c.get("v", 0) or 0) for c in candles], dtype=np.float64)
    return build_features(closes, highs, lows, vols)


def live_dca_leg_count(*, allow_dca: bool, extra_adds: int) -> int:
    """Fills planned per pair: 1 entry + extra_adds when DCA is on."""
    if not allow_dca:
        return 1
    return 1 + max(0, int(extra_adds or 0))


def position_leg_count(setup: Any) -> int:
    """Total fills planned: 1 entry + dca_max_adds (when DCA on)."""
    if bool(getattr(setup, "dca_enabled", False)) and int(
        getattr(setup, "dca_max_adds", 0) or 0
    ) > 0:
        return 1 + int(setup.dca_max_adds)
    return 1


def total_balance_pct(setup: Any, default: float = 30.0) -> float:
    """Configured TOTAL margin budget for the whole position (all legs)."""
    raw = getattr(setup, "balance_pct", None)
    if raw is None:
        return min(95.0, max(1.0, float(default)))
    return min(95.0, max(1.0, float(raw)))


def leg_balance_pct(setup: Any, default: float = 30.0) -> float:
    """Margin % for a single entry/DCA leg (= total / leg_count)."""
    return total_balance_pct(setup, default) / float(position_leg_count(setup))


def entry_signal(
    setup: Any,
    candles: list[dict],
    *,
    multi_candles: dict[str, list[dict]] | None = None,
) -> int:
    """
    Backtest signal side +1 long / -1 short / 0 none — last closed bar only.

    Never applies REVERSE_STRATEGY. Live/paper must flip the returned side at
    order time so entry timing matches the tuned mask exactly.
    """
    if getattr(setup, "is_mtf", False) or str(getattr(setup, "mode", "")).lower() == "mtf":
        by_iv = dict(multi_candles or {})
        exec_iv = str(setup.interval)
        if exec_iv not in by_iv and candles:
            by_iv[exec_iv] = candles
        return mtf_entry_signal_now(setup, by_iv)

    feats = candles_to_feats(candles)
    if feats is None:
        return 0
    mask = build_entry_mask(
        int(setup.sid), feats, setup.p0, setup.p1, setup.p2, setup.aux
    )
    if not bool(mask[-1]):
        return 0
    return int(setup.side)


def exit_signal(
    setup: Any,
    candles: list[dict],
    *,
    avg_entry_px: float,
    position_side: int | None = None,
) -> bool:
    """
    Closed-bar exit. Timing matches backtest on the side being managed.

    position_side: actual open side (+1/-1). When set (always preferred), exits
    use that side so REVERSE_STRATEGY shorts get short exit rules / DCA-consistent
    profit_snap. Falls back to setup.side only if position_side is unknown.
    """
    if not setup.use_exit_signal or setup.exit_eid < 0:
        return False
    if avg_entry_px <= 0:
        return False
    feats = candles_to_feats(candles)
    if feats is None:
        return False
    close = float(feats["close"][-1])

    # Actual position side wins (critical when orders are reversed vs setup.side).
    side = int(position_side) if position_side is not None else int(setup.side)

    if setup.exit_eid == 2:
        move = (close - avg_entry_px) / avg_entry_px * 100.0
        if side > 0:
            return move >= setup.ex_p0
        return move <= -setup.ex_p0

    mask = build_exit_mask(
        setup.exit_eid, feats, side, setup.ex_p0, setup.ex_aux
    )
    return bool(mask[-1])


def dca_should_add(
    setup: Any,
    *,
    avg_entry_px: float,
    mark_or_close: float,
    position_side: str,
    dca_adds_done: int,
) -> bool:
    """Same adverse-% rule as Numba sim (from average entry)."""
    if not setup.dca_enabled or setup.dca_max_adds <= 0:
        return False
    if dca_adds_done >= setup.dca_max_adds:
        return False
    if avg_entry_px <= 0 or mark_or_close <= 0:
        return False
    if position_side == "long":
        adverse = (avg_entry_px - mark_or_close) / avg_entry_px * 100.0
    else:
        adverse = (mark_or_close - avg_entry_px) / avg_entry_px * 100.0
    return adverse >= float(setup.dca_trigger_pct)


def setup_to_dict(setup: Any) -> dict[str, Any]:
    return {
        "coin": setup.coin,
        "interval": setup.interval,
        "sid": setup.sid,
        "name": setup.name,
        "side": setup.side,
        "p0": setup.p0,
        "p1": setup.p1,
        "p2": setup.p2,
        "aux": setup.aux,
        "tp_pct": setup.tp_pct,
        "sl_pct": setup.sl_pct,
        "balance_pct": setup.balance_pct,
        "use_tpsl": setup.use_tpsl,
        "use_exit_signal": setup.use_exit_signal,
        "use_max_hold": setup.use_max_hold,
        "exit_eid": setup.exit_eid,
        "exit_name": setup.exit_name,
        "ex_p0": setup.ex_p0,
        "ex_aux": setup.ex_aux,
        "dca_enabled": setup.dca_enabled,
        "dca_trigger_pct": setup.dca_trigger_pct,
        "dca_max_adds": setup.dca_max_adds,
        "dca_size_mult": setup.dca_size_mult,
        "score": setup.score,
        "win_rate_pct": setup.win_rate_pct,
        "trades_per_day": setup.trades_per_day,
        "rank_score": setup.rank_score,
        "max_hold_bars": setup.max_hold_bars,
        "mode": getattr(setup, "mode", "legacy"),
        "mtf_intervals": list(getattr(setup, "mtf_intervals", ()) or ()),
        "mtf_ema": getattr(setup, "mtf_ema", 50),
        "mtf_min_agree": getattr(setup, "mtf_min_agree", 3),
        "mtf_min_score": getattr(setup, "mtf_min_score", 0.35),
        "mtf_weight_power": getattr(setup, "mtf_weight_power", 0.5),
    }
