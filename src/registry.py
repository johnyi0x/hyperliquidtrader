"""
Strategy + exit + DCA registries with pruned (non-absurd) parameter grids.

Side: +1 long, -1 short.
Grids are intentionally compact for Numba speed while covering diverse edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .indicators import ema


@dataclass(frozen=True)
class StratDef:
    sid: int
    name: str
    side: int
    desc: str


STRATEGIES: list[StratDef] = [
    StratDef(0, "rsi_long", 1, "RSI oversold → long"),
    StratDef(1, "rsi_short", -1, "RSI overbought → short"),
    StratDef(2, "ema_stretch_long", 1, "price stretched below EMA → long"),
    StratDef(3, "ema_stretch_short", -1, "price stretched above EMA → short"),
    StratDef(4, "ema_cross_long", 1, "fast EMA crosses above slow → long"),
    StratDef(5, "ema_cross_short", -1, "fast EMA crosses below slow → short"),
    StratDef(6, "bb_bounce_long", 1, "close at/below lower BB → long"),
    StratDef(7, "bb_fade_short", -1, "close at/above upper BB → short"),
    StratDef(8, "zscore_long", 1, "z-score deep negative → long"),
    StratDef(9, "zscore_short", -1, "z-score deep positive → short"),
    StratDef(10, "mom_break_long", 1, "strong up ret + volume → long"),
    StratDef(11, "mom_break_short", -1, "strong down ret + volume → short"),
    StratDef(12, "atr_break_long", 1, "close breaks above prior + k*ATR → long"),
    StratDef(13, "atr_break_short", -1, "close breaks below prior - k*ATR → short"),
    StratDef(14, "dump_bounce", 1, "fast dump → long fade"),
    StratDef(15, "pump_fade", -1, "fast pump → short fade"),
    StratDef(16, "dump_rsi_long", 1, "fast dump + RSI oversold → long"),
    StratDef(17, "pump_rsi_short", -1, "fast pump + RSI overbought → short"),
]

STRATEGY_BY_ID = {s.sid: s for s in STRATEGIES}


def _finite(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)


def build_entry_mask(
    sid: int,
    feats: dict[str, np.ndarray],
    p0: float,
    p1: float,
    p2: float,
    aux: float,
) -> np.ndarray:
    close = feats["close"]
    n = len(close)
    mask = np.zeros(n, dtype=np.bool_)

    if sid == 0:
        rsi = feats["rsi14"] if int(aux) != 7 else feats["rsi7"]
        mask = _finite(rsi) & (rsi <= p0)
    elif sid == 1:
        rsi = feats["rsi14"] if int(aux) != 7 else feats["rsi7"]
        mask = _finite(rsi) & (rsi >= p0)
    elif sid in (2, 3):
        period = max(20, int(aux))
        key = {20: "ema20", 50: "ema50", 100: "ema100"}.get(period)
        series = feats[key] if key else ema(close, period)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = (close - series) / np.where(series == 0, np.nan, series) * 100.0
        if sid == 2:
            mask = _finite(dev) & (dev <= -p0)
        else:
            mask = _finite(dev) & (dev >= p0)
    elif sid in (4, 5):
        fast = feats["ema20"]
        slow = feats["ema50"]
        prev_f = np.roll(fast, 1)
        prev_s = np.roll(slow, 1)
        prev_f[0] = np.nan
        prev_s[0] = np.nan
        if sid == 4:
            mask = (
                _finite(fast)
                & _finite(slow)
                & _finite(prev_f)
                & _finite(prev_s)
                & (prev_f <= prev_s)
                & (fast > slow)
            )
        else:
            mask = (
                _finite(fast)
                & _finite(slow)
                & _finite(prev_f)
                & _finite(prev_s)
                & (prev_f >= prev_s)
                & (fast < slow)
            )
    elif sid in (6, 7):
        mid = feats["sma20"]
        atr = feats["atr14"]
        band = p1 * atr
        if sid == 6:
            mask = _finite(mid) & _finite(band) & (close <= mid - band)
        else:
            mask = _finite(mid) & _finite(band) & (close >= mid + band)
    elif sid == 8:
        z = feats["z20"] if int(aux) == 20 else feats["z50"]
        mask = _finite(z) & (z <= -p0)
    elif sid == 9:
        z = feats["z20"] if int(aux) == 20 else feats["z50"]
        mask = _finite(z) & (z >= p0)
    elif sid == 10:
        ret = feats["ret12"]
        vr = feats["vol_ratio"]
        mask = _finite(ret) & _finite(vr) & (ret >= p0) & (vr >= p1)
    elif sid == 11:
        ret = feats["ret12"]
        vr = feats["vol_ratio"]
        mask = _finite(ret) & _finite(vr) & (ret <= -p0) & (vr >= p1)
    elif sid in (12, 13):
        atr = feats["atr14"]
        prev = np.roll(close, 1)
        prev[0] = np.nan
        if sid == 12:
            mask = _finite(atr) & _finite(prev) & (close >= prev + p0 * atr)
        else:
            mask = _finite(atr) & _finite(prev) & (close <= prev - p0 * atr)
    elif sid == 14:
        ret = feats["ret5"]
        mask = _finite(ret) & (ret <= -p0)
    elif sid == 15:
        ret = feats["ret5"]
        mask = _finite(ret) & (ret >= p0)
    elif sid == 16:
        rsi = feats["rsi14"] if int(aux) != 7 else feats["rsi7"]
        ret = feats["ret5"]
        mask = _finite(ret) & _finite(rsi) & (ret <= -p0) & (rsi <= p1)
    elif sid == 17:
        rsi = feats["rsi14"] if int(aux) != 7 else feats["rsi7"]
        ret = feats["ret5"]
        mask = _finite(ret) & _finite(rsi) & (ret >= p0) & (rsi >= p1)

    warmup = 40
    if n > warmup:
        mask[:warmup] = False
    else:
        mask[:] = False
    return mask


# (p0, p1, p2, aux) — compact grids
ENTRY_GRIDS: dict[int, list[tuple[float, float, float, float]]] = {
    0: [(r, 0, 0, p) for p in (14, 7) for r in (25, 30, 35)],
    1: [(r, 0, 0, p) for p in (14, 7) for r in (65, 70, 75)],
    2: [(d, 0, 0, p) for p in (50, 100) for d in (1.0, 1.5, 2.5)],
    3: [(d, 0, 0, p) for p in (50, 100) for d in (1.0, 1.5, 2.5)],
    4: [(0, 0, 0, 0)],
    5: [(0, 0, 0, 0)],
    6: [(0, k, 0, 0) for k in (1.5, 2.0, 2.5)],
    7: [(0, k, 0, 0) for k in (1.5, 2.0, 2.5)],
    8: [(z, 0, 0, w) for w in (20, 50) for z in (1.5, 2.0, 2.5)],
    9: [(z, 0, 0, w) for w in (20, 50) for z in (1.5, 2.0, 2.5)],
    10: [(r, v, 0, 0) for r in (0.8, 1.5, 2.5) for v in (1.5, 2.0)],
    11: [(r, v, 0, 0) for r in (0.8, 1.5, 2.5) for v in (1.5, 2.0)],
    12: [(k, 0, 0, 0) for k in (0.8, 1.2, 1.8)],
    13: [(k, 0, 0, 0) for k in (0.8, 1.2, 1.8)],
    14: [(d, 0, 0, 0) for d in (0.8, 1.2, 2.0, 3.0)],
    15: [(d, 0, 0, 0) for d in (0.8, 1.2, 2.0, 3.0)],
    16: [
        (d, r, 0, p)
        for p in (14, 7)
        for d in (0.8, 1.2, 2.0)
        for r in (30.0, 35.0)
    ],
    17: [
        (d, r, 0, p)
        for p in (14, 7)
        for d in (0.8, 1.2, 2.0)
        for r in (65.0, 70.0)
    ],
}

# Exit signal ids for closed-bar exits
EXIT_SIGNAL_GRIDS: list[dict[str, Any]] = [
    {"exit_eid": -1, "exit_name": "none", "ex_p0": 0.0, "ex_aux": 0.0},
    {"exit_eid": 0, "exit_name": "rsi_target", "ex_p0": 50.0, "ex_aux": 14.0},
    {"exit_eid": 0, "exit_name": "rsi_target", "ex_p0": 55.0, "ex_aux": 14.0},
    {"exit_eid": 0, "exit_name": "rsi_target", "ex_p0": 60.0, "ex_aux": 14.0},
    {"exit_eid": 1, "exit_name": "ema_revert", "ex_p0": 0.3, "ex_aux": 50.0},
    {"exit_eid": 1, "exit_name": "ema_revert", "ex_p0": 0.6, "ex_aux": 50.0},
    {"exit_eid": 2, "exit_name": "profit_snap", "ex_p0": 0.5, "ex_aux": 0.0},
    {"exit_eid": 2, "exit_name": "profit_snap", "ex_p0": 1.0, "ex_aux": 0.0},
    {"exit_eid": 2, "exit_name": "profit_snap", "ex_p0": 1.5, "ex_aux": 0.0},
]

TP_GRID = (0.5, 0.8, 1.0, 1.5, 2.0, 3.0)
TP_GRID_FAST = (0.8, 1.5, 2.0, 3.0)

EXIT_SIGNAL_GRIDS_FAST: list[dict[str, Any]] = [
    {"exit_eid": -1, "exit_name": "none", "ex_p0": 0.0, "ex_aux": 0.0},
    {"exit_eid": 0, "exit_name": "rsi_target", "ex_p0": 55.0, "ex_aux": 14.0},
    {"exit_eid": 1, "exit_name": "ema_revert", "ex_p0": 0.5, "ex_aux": 50.0},
    {"exit_eid": 2, "exit_name": "profit_snap", "ex_p0": 1.0, "ex_aux": 0.0},
]

def dca_refine_grid(*, fast: bool, max_adds: int) -> list[dict[str, Any]]:
    """Equal-size extra fills after entry. max_adds=1 → two legs total."""
    n = max(1, int(max_adds))
    triggers = (0.8, 1.5, 2.5) if fast else (0.8, 1.2, 1.5, 2.0, 2.5)
    return [
        {
            "dca_enabled": True,
            "dca_trigger_pct": float(t),
            "dca_max_adds": n,
            "dca_size_mult": 1.0,
        }
        for t in triggers
    ]


DCA_GRID_FAST: list[dict[str, Any]] = dca_refine_grid(fast=True, max_adds=1)

# DCA: always-on variants when ALLOW_DCA (equal-size legs; size_mult kept at 1).
# balance_pct is TOTAL margin budget for entry + all DCA adds combined.
DCA_GRID: list[dict[str, Any]] = dca_refine_grid(fast=False, max_adds=1)

DCA_OFF: dict[str, Any] = {
    "dca_enabled": False,
    "dca_trigger_pct": 0.0,
    "dca_max_adds": 0,
    "dca_size_mult": 0.0,
}


def build_exit_mask(
    exit_eid: int,
    feats: dict[str, np.ndarray],
    side: int,
    ex_p0: float,
    ex_aux: float,
) -> np.ndarray:
    close = feats["close"]
    n = len(close)
    mask = np.zeros(n, dtype=np.bool_)
    if exit_eid < 0:
        return mask
    if exit_eid == 0:
        rsi = feats["rsi14"]
        if side > 0:
            mask = _finite(rsi) & (rsi >= ex_p0)
        else:
            # Mirror: long exits at 60 → short exits at 40
            thr = (100.0 - ex_p0) if ex_p0 >= 50.0 else ex_p0
            mask = _finite(rsi) & (rsi <= thr)
    elif exit_eid == 1:
        period = max(20, int(ex_aux))
        key = {20: "ema20", 50: "ema50", 100: "ema100"}.get(period)
        series = feats[key] if key else ema(close, period)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = (close - series) / np.where(series == 0, np.nan, series) * 100.0
        prev = np.roll(dev, 1)
        prev[0] = np.nan
        # Require recovery from a real stretch (avoids instant exits on EMA-cross entries).
        if side > 0:
            mask = (
                _finite(dev)
                & _finite(prev)
                & (prev <= -ex_p0)
                & (dev >= -ex_p0)
            )
        else:
            mask = (
                _finite(dev)
                & _finite(prev)
                & (prev >= ex_p0)
                & (dev <= ex_p0)
            )
    # exit_eid 2 profit_snap handled in sim via entry price
    return mask


def iter_entry_combos() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sid, grid in ENTRY_GRIDS.items():
        sdef = STRATEGY_BY_ID[sid]
        for p0, p1, p2, aux in grid:
            out.append(
                {
                    "sid": sid,
                    "name": sdef.name,
                    "side": sdef.side,
                    "p0": float(p0),
                    "p1": float(p1),
                    "p2": float(p2),
                    "aux": float(aux),
                }
            )
    return out


def screen_exit_bundle(*, use_tpsl: bool, use_max_hold: bool) -> dict[str, Any]:
    """Cheap baseline exit for stage-1 screening."""
    return {
        "use_tpsl": bool(use_tpsl),
        "tp_pct": 1.0 if use_tpsl else 0.0,
        "sl_pct": 1.0 if use_tpsl else 0.0,
        "use_exit_signal": False,
        "exit_eid": -1,
        "exit_name": "none",
        "ex_p0": 0.0,
        "ex_aux": 0.0,
        "use_max_hold": bool(use_max_hold),
        "dca_enabled": False,
        "dca_trigger_pct": 0.0,
        "dca_max_adds": 0,
        "dca_size_mult": 0.0,
        "balance_pct": 30.0,
    }


def iter_refine_combos(
    *,
    use_tpsl: bool,
    use_exit_signal: bool,
    use_max_hold: bool,
    allow_dca: bool,
    balance_grid: tuple[float, ...],
    profile: str = "full",
    dca_max_adds: int = 1,
) -> list[dict[str, Any]]:
    """Stage-2 exit/DCA/balance variants (applied to top screened entries)."""
    fast = str(profile or "full").strip().lower() == "fast"
    tp_vals = (TP_GRID_FAST if fast else TP_GRID) if use_tpsl else (0.0,)
    if use_exit_signal:
        exits = EXIT_SIGNAL_GRIDS_FAST if fast else EXIT_SIGNAL_GRIDS
    else:
        exits = [
            {"exit_eid": -1, "exit_name": "none", "ex_p0": 0.0, "ex_aux": 0.0}
        ]
    # DCA is mandatory when allowed (+ TP/SL for live protect re-center).
    if allow_dca and use_tpsl:
        dcas = dca_refine_grid(fast=fast, max_adds=dca_max_adds)
    else:
        dcas = [DCA_OFF]
    bals = balance_grid or (30.0,)
    out: list[dict[str, Any]] = []
    for tp in tp_vals:
        for ex in exits:
            for dca in dcas:
                for bal in bals:
                    # Soft exchange headroom: never plan >95% total margin.
                    total_bal = min(95.0, max(1.0, float(bal)))
                    out.append(
                        {
                            "use_tpsl": bool(use_tpsl),
                            "tp_pct": float(tp) if use_tpsl else 0.0,
                            "sl_pct": float(tp) if use_tpsl else 0.0,
                            "use_exit_signal": bool(
                                use_exit_signal and int(ex["exit_eid"]) >= 0
                            ),
                            "exit_eid": int(ex["exit_eid"]),
                            "exit_name": str(ex["exit_name"]),
                            "ex_p0": float(ex["ex_p0"]),
                            "ex_aux": float(ex["ex_aux"]),
                            "use_max_hold": bool(use_max_hold),
                            "dca_enabled": bool(dca["dca_enabled"]),
                            "dca_trigger_pct": float(dca["dca_trigger_pct"]),
                            "dca_max_adds": int(dca["dca_max_adds"]),
                            "dca_size_mult": 1.0,
                            "balance_pct": total_bal,
                        }
                    )
    return out
