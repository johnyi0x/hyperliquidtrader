"""Grid / random search over crowd strategies with Numba PnL + distribution scoring."""

from __future__ import annotations

import itertools
import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .bt_numba import PMF_FEE_PCT, simulate_portfolio
from .bt_replay import STRATEGY_REPLAYERS
from .research_load import ResearchDataset, build_dataset

log = logging.getLogger("pmf-bt-tune")

# Process-pool worker state (set by _pool_init).
_POOL_DS: ResearchDataset | None = None
_POOL_CFG: Any = None
_POOL_TICK: float = 60.0
_POOL_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] | None = None


def _pool_init(ds: ResearchDataset, base_cfg: Any, tick_iv: float) -> None:
    global _POOL_DS, _POOL_CFG, _POOL_TICK, _POOL_CACHE
    _POOL_DS = ds
    _POOL_CFG = base_cfg
    _POOL_TICK = float(tick_iv)
    _POOL_CACHE = {}


def _pool_run_job(job: tuple[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Worker entry: run one combo; return TuneResult as a plain dict."""
    strategy, params = job
    assert _POOL_DS is not None and _POOL_CFG is not None
    try:
        r = run_one(
            _POOL_DS,
            _POOL_CFG,
            strategy=strategy,
            params=params,
            tick_interval=_POOL_TICK,
            progress_cb=None,
            replay_cache=_POOL_CACHE,
        )
        return {
            "strategy": r.strategy,
            "params": r.params,
            "return_pct": r.return_pct,
            "max_dd_pct": r.max_dd_pct,
            "round_trips": r.round_trips,
            "win_rate_pct": r.win_rate_pct,
            "score": r.score,
            "trades_per_day": r.trades_per_day,
            "active_day_ratio": r.active_day_ratio,
            "cluster_share": r.cluster_share,
            "total_fees": r.total_fees,
            "open_legs": r.open_legs,
            "meta": r.meta,
        }
    except Exception as exc:
        log.debug("Tune skip %s %s: %s", strategy, params, exc)
        return None


def _job_locality_key(job: tuple[str, dict[str, Any]]) -> tuple[Any, ...]:
    """Cluster combos for process-local caches (MTF entry first — dominant cost)."""
    strat, params = job
    # MTF entry cache key uses preset/ema/agree/score/interval — group those first.
    mtf = (
        params.get("MTF_PRESET"),
        params.get("MTF_EMA"),
        params.get("MTF_MIN_AGREE"),
        params.get("MTF_MIN_SCORE"),
        params.get("MTF_WEIGHT_POWER"),
    )
    swing = (
        params.get("SWING_ENTRY"),
        params.get("SWING_TF"),
        params.get("SWING_RSI_BUY"),
        params.get("SWING_RSI_SELL"),
        params.get("SWING_LOOKBACK_S"),
    )
    crowd = tuple(
        sorted(
            (k, params[k])
            for k in params
            if k != "REBALANCE_COOLDOWN_S" and not str(k).startswith(("MTF_", "SWING_"))
        )
    )
    rest_mtf = tuple(sorted((k, params[k]) for k in params if str(k).startswith("MTF_")))
    rest_swing = tuple(sorted((k, params[k]) for k in params if str(k).startswith("SWING_")))
    return (strat, mtf, swing, crowd, rest_mtf, rest_swing)


def _chunk_jobs(
    jobs: list[tuple[str, dict[str, Any]]],
    workers: int,
) -> list[list[tuple[str, dict[str, Any]]]]:
    """Contiguous slices of locality-sorted jobs → each worker warms one niche."""
    ordered = sorted(jobs, key=_job_locality_key)
    workers = max(1, workers)
    n = len(ordered)
    if n == 0:
        return []
    size = (n + workers - 1) // workers
    return [ordered[i : i + size] for i in range(0, n, size)]


def _pool_run_chunk(chunk: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any] | None]:
    # Keep process-local raw-vote + MTF entry caches warm across chunks on this worker.
    return [_pool_run_job(job) for job in chunk]


def _result_from_dict(d: dict[str, Any]) -> TuneResult:
    return TuneResult(
        strategy=str(d["strategy"]),
        params=dict(d["params"]),
        return_pct=float(d["return_pct"]),
        max_dd_pct=float(d["max_dd_pct"]),
        round_trips=int(d["round_trips"]),
        win_rate_pct=float(d["win_rate_pct"]),
        score=float(d["score"]),
        trades_per_day=float(d["trades_per_day"]),
        active_day_ratio=float(d["active_day_ratio"]),
        cluster_share=float(d["cluster_share"]),
        total_fees=float(d.get("total_fees") or 0.0),
        open_legs=int(d.get("open_legs") or 0),
        meta=dict(d.get("meta") or {}),
    )


def _freeze_cfg(base_cfg: Any) -> SimpleNamespace:
    """Picklable snapshot of UPPERCASE cfg attrs (config modules are not picklable)."""
    ns = SimpleNamespace()
    for name in dir(base_cfg):
        if not name.isupper():
            continue
        try:
            setattr(ns, name, getattr(base_cfg, name))
        except Exception:
            continue
    return ns


def _worker_count(total_jobs: int) -> int:
    """Cap workers: avoid oversubscribing on small jobs / tiny datasets."""
    override = os.environ.get("PMF_BT_WORKERS", "").strip()
    if override.isdigit():
        return max(1, min(int(override), total_jobs))
    cpu = os.cpu_count() or 2
    want = max(1, min(cpu, 8, total_jobs))
    # Leave one core free on big machines.
    if cpu >= 6 and want == cpu:
        want = cpu - 1
    return max(1, want)

# Default: filter on/off + dump-shield variants (price gates on top of crowd).
# Extra replay styles: --all-strategies.
LIVE_STRATEGIES = (
    "cloud_holders",
    "cloud_all",
    "crowd_dump_holders",
    "crowd_dump_all",
    "crowd_btcdump_holders",
    "mtf_meta_holders",
    "mtf_meta_all",
    "swing_meta_holders",
    "swing_meta_all",
)

# Meta + own entry/exit timing (multi-candle MTF + indicator swing), holders and all-pool.
# Used by --meta-timing so these get a deep search without crowding from gate/refine styles.
META_TIMING_STRATEGIES = (
    "mtf_meta_holders",
    "mtf_meta_all",
    "swing_meta_holders",
    "swing_meta_all",
)

TUNABLE_KEYS = (
    "FLOW_EMA_ALPHA",
    "EXIT_FLOW",
    "EXIT_RAW_FLOW",
    "EXIT_AGREEMENT_GIVEBACK",
    "CONV_GIVEBACK",
    "OPEN_CONFIRM_S",
    "MIN_SIDE_AGREEMENT",
    "EXIT_SIDE_AGREEMENT",
    "MIN_AVG_CONVICTION",
    "EXIT_AVG_CONVICTION",
    "MIN_ENTRY_FLOW",
    "REBALANCE_COOLDOWN_S",
    "MAX_COINS_IN_BOOK",
)

# Extra knobs for crowd+price strategies (live + backtest).
INDICATOR_KEYS = (
    "DUMP_RET_PCT",
    "DUMP_LOOKBACK_S",
    "DUMP_RANGE_PCT",
    "TREND_BIAS_MIN",
    "MAX_ATR_PCT",
    "RSI_MAX",
    "RSI_MIN",
)

# Only affects Numba sim — crowd replay can be reused across these.
SIM_ONLY_KEYS = frozenset({"REBALANCE_COOLDOWN_S"})

# Tight sensible bands near live defaults (no junk extremes).
DEFAULT_GRID: dict[str, list[Any]] = {
    "FLOW_EMA_ALPHA": [0.20, 0.24],
    "EXIT_FLOW": [-0.011, -0.015],
    "EXIT_RAW_FLOW": [-0.020, -0.028],
    "EXIT_AGREEMENT_GIVEBACK": [0.28, 0.34],
    "CONV_GIVEBACK": [0.26, 0.34],
    "OPEN_CONFIRM_S": [150.0, 210.0],
    "MIN_SIDE_AGREEMENT": [0.08, 0.12],
    "EXIT_SIDE_AGREEMENT": [0.04, 0.06],
    "MIN_AVG_CONVICTION": [0.018, 0.026],
    "EXIT_AVG_CONVICTION": [0.016, 0.024],
    "MIN_ENTRY_FLOW": [0.0, 0.002],
    "REBALANCE_COOLDOWN_S": [180.0, 300.0],
}

INDICATOR_GRID: dict[str, list[Any]] = {
    "DUMP_RET_PCT": [-0.02, -0.03],
    "DUMP_LOOKBACK_S": [900.0, 1800.0],
    "DUMP_RANGE_PCT": [-0.025, -0.035],
    "TREND_BIAS_MIN": [0.0, 0.005],
    "MAX_ATR_PCT": [0.04, 0.06],
    "RSI_MAX": [72.0, 78.0],
    "RSI_MIN": [22.0, 28.0],
}

# Original multi-candle (no DCA) knobs — only attached to mtf_meta_* strategies.
MTF_KEYS = (
    "MTF_PRESET",
    "MTF_EMA",
    "MTF_MIN_AGREE",
    "MTF_MIN_SCORE",
    "MTF_EXIT",
    "MTF_META_MODE",
    "MTF_MAX_HOLD_S",
)

MTF_GRID: dict[str, list[Any]] = {
    "MTF_PRESET": [
        "rsi_long_30",
        "rsi_long_35",
        "rsi_short_70",
        "ema_x_long",
        "ema_x_short",
        "bb_bounce_long",
        "zscore_long",
        "dump_bounce",
        "pump_fade",
    ],
    "MTF_EMA": [20, 50],
    "MTF_MIN_AGREE": [2, 3],
    "MTF_MIN_SCORE": [0.30, 0.45],
    "MTF_EXIT": ["none", "rsi_55", "ema_0.5", "profit_1.0"],
    "MTF_META_MODE": ["follow", "reverse"],
    "MTF_MAX_HOLD_S": [3600.0, 14400.0, 86400.0],
}

# Holder-meta + indicator swing timing (own entries/exits) — swing_meta_* only.
SWING_KEYS = (
    "SWING_META_MODE",
    "SWING_ENTRY",
    "SWING_TF",
    "SWING_RSI_BUY",
    "SWING_RSI_SELL",
    "SWING_BAND_PCT",
    "SWING_BREAK_PCT",
    "SWING_LOOKBACK_S",
    "SWING_TP_PCT",
    "SWING_SL_PCT",
    "SWING_MAX_HOLD_S",
    "SWING_EXIT_RSI",
    "SWING_REENTRY_S",
)

SWING_GRID: dict[str, list[Any]] = {
    "SWING_META_MODE": ["follow", "reverse"],
    "SWING_ENTRY": ["rsi_dip", "ema_pullback", "breakout", "range_dip"],
    "SWING_TF": ["1m", "15m", "1h"],
    "SWING_RSI_BUY": [25.0, 30.0, 35.0, 40.0, 45.0],
    "SWING_RSI_SELL": [55.0, 60.0, 65.0, 70.0, 75.0],
    "SWING_BAND_PCT": [0.003, 0.005, 0.008, 0.012, 0.020],
    "SWING_BREAK_PCT": [0.004, 0.008, 0.012, 0.020, 0.030],
    "SWING_LOOKBACK_S": [600.0, 1800.0, 3600.0, 7200.0],
    "SWING_TP_PCT": [0.4, 0.8, 1.2, 2.0, 3.0, 5.0],
    "SWING_SL_PCT": [0.6, 1.0, 1.8, 3.0, 5.0],
    "SWING_MAX_HOLD_S": [1800.0, 7200.0, 21600.0, 86400.0],
    "SWING_EXIT_RSI": [0.0, 55.0, 60.0, 70.0],
    "SWING_REENTRY_S": [0.0, 300.0, 900.0, 3600.0],
}

# All searchable knobs, in stable order.
ALL_PARAM_KEYS = tuple(TUNABLE_KEYS) + tuple(INDICATOR_KEYS) + tuple(MTF_KEYS) + tuple(SWING_KEYS)

# Wide sweep ranges for stage 1 (coarse). Endpoints stay inside sane bands but
# span far more than the deep grid so a strategy is never judged on one corner.
WIDE_GRID: dict[str, list[Any]] = {
    "FLOW_EMA_ALPHA": [0.12, 0.20, 0.28, 0.36],
    "EXIT_FLOW": [-0.005, -0.011, -0.018, -0.026],
    "EXIT_RAW_FLOW": [-0.010, -0.020, -0.032, -0.045],
    "EXIT_AGREEMENT_GIVEBACK": [0.18, 0.28, 0.38, 0.48],
    "CONV_GIVEBACK": [0.18, 0.28, 0.38, 0.48],
    "OPEN_CONFIRM_S": [60.0, 150.0, 240.0, 330.0],
    "MIN_SIDE_AGREEMENT": [0.04, 0.10, 0.16, 0.22],
    "EXIT_SIDE_AGREEMENT": [0.02, 0.05, 0.08, 0.11],
    "MIN_AVG_CONVICTION": [0.010, 0.020, 0.030, 0.040],
    "EXIT_AVG_CONVICTION": [0.008, 0.018, 0.028, 0.038],
    "MIN_ENTRY_FLOW": [0.0, 0.003, 0.006],
    "REBALANCE_COOLDOWN_S": [120.0, 300.0, 480.0, 660.0],
    "DUMP_RET_PCT": [-0.010, -0.025, -0.040, -0.060],
    "DUMP_LOOKBACK_S": [300.0, 1800.0, 3600.0, 7200.0],
    "DUMP_RANGE_PCT": [-0.010, -0.025, -0.040, -0.060],
    "TREND_BIAS_MIN": [0.0, 0.004, 0.008, 0.012],
    "MAX_ATR_PCT": [0.015, 0.040, 0.070, 0.100],
    "RSI_MAX": [60.0, 70.0, 78.0, 86.0],
    "RSI_MIN": [14.0, 22.0, 30.0, 40.0],
}


def _wide_values(key: str, deep_values: list[Any], *, points: int = 4) -> list[Any]:
    """Stage-1 values: explicit wide band if defined, else spread across deep list."""
    wide = WIDE_GRID.get(key)
    if wide:
        return list(wide)
    vals = list(deep_values)
    if len(vals) <= points:
        return vals
    step = (len(vals) - 1) / float(points - 1)
    idx = sorted({int(round(i * step)) for i in range(points)})
    return [vals[i] for i in idx]


def coarse_grid(grid: dict[str, list[Any]], *, points: int = 4) -> dict[str, list[Any]]:
    """Wide, shallow sweep of every knob (fast cross-strategy screen)."""
    return {k: _wide_values(k, v, points=points) for k, v in grid.items() if v}


def deep_grid_around(
    grid: dict[str, list[Any]],
    best: dict[str, Any],
    *,
    width: int = 2,
) -> dict[str, list[Any]]:
    """Stage-2 grid: dense values near the stage-1 winner, full list for categoricals."""
    out: dict[str, list[Any]] = {}
    for key, values in grid.items():
        vals = list(values)
        if not vals:
            continue
        pick = best.get(key)
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
        if pick is None or not numeric or not isinstance(pick, (int, float)):
            out[key] = vals
            continue
        merged = sorted({*(float(v) for v in vals), float(pick)})
        i = min(range(len(merged)), key=lambda j: abs(merged[j] - float(pick)))
        lo = max(0, i - width)
        hi = min(len(merged), i + width + 1)
        window = merged[lo:hi]
        # Add midpoints between neighbours for finer resolution.
        dense: list[float] = []
        for a, b in zip(window, window[1:]):
            dense.extend([a, (a + b) / 2.0])
        dense.append(window[-1])
        if all(isinstance(v, int) for v in vals):
            out[key] = sorted({int(round(v)) for v in dense})
        else:
            out[key] = sorted({round(v, 6) for v in dense})
    return out


@dataclass
class TuneResult:
    strategy: str
    params: dict[str, Any]
    return_pct: float
    max_dd_pct: float
    round_trips: int
    win_rate_pct: float
    score: float
    trades_per_day: float
    active_day_ratio: float
    cluster_share: float
    total_fees: float = 0.0
    open_legs: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def sim_size_from_cfg(cfg: Any) -> tuple[float, float]:
    """Match live fixed-size: 90% gross × 33.33%/coin = 30% equity, ~10x mean lev."""
    gross = float(getattr(cfg, "OUR_GROSS_MARGIN_PCT", 90.0) or 90.0)
    per_coin = float(getattr(cfg, "MAX_MARGIN_PER_COIN_PCT", 33.33) or 33.33)
    margin_frac = (gross * (per_coin / 100.0)) / 100.0
    lo = float(getattr(cfg, "OUR_MIN_LEVERAGE", 2) or 2)
    hi = float(getattr(cfg, "OUR_MAX_LEVERAGE", 20) or 20)
    leverage = min(hi, max(lo, 10.0))
    return margin_frac, leverage


def _fmt_dur(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _progress(done: int, total: int, t0: float, *, label: str = "", frac_extra: float = 0.0) -> None:
    """Single-line bar + ETA on stderr. frac_extra in [0,1) for in-combo progress."""
    import sys

    total = max(1, int(total))
    done = min(max(0, int(done)), total)
    elapsed = time.time() - t0
    # Include partial combo so the bar moves during long replays.
    effective = min(float(total), float(done) + max(0.0, min(0.999, float(frac_extra))))
    frac = effective / total
    if effective > 0.05:
        eta = elapsed * (total - effective) / effective
        eta_s = _fmt_dur(eta)
    else:
        eta_s = "?"
    width = 28
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    tag = f" {label}" if label else ""
    line = (
        f"\r[{bar}] {done}/{total} {frac:5.1%}"
        f"  elapsed {_fmt_dur(elapsed)}  eta {eta_s}{tag}   "
    )
    sys.stderr.write(line)
    sys.stderr.flush()
    if done == 0 and frac_extra <= 0:
        sys.stderr.write("\n")
        sys.stderr.flush()
    elif done >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _replay_cache_key(strategy: str, params: dict[str, Any]) -> tuple[Any, ...]:
    items = tuple(sorted((k, params[k]) for k in params if k not in SIM_ONLY_KEYS))
    return (strategy, items)


def _tick_interval_s(ds: ResearchDataset) -> float:
    if ds.n_ticks < 2:
        return 60.0
    gaps = np.diff(ds.ts)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 60.0
    return float(np.median(gaps))


def score_backtest(
    *,
    return_pct: float,
    max_dd_pct: float,
    round_trips: int,
    trades_by_day: np.ndarray,
    span_days: float,
    n_day_bins: int,
    min_trips: int | None = None,
) -> tuple[float, float, float, float]:
    """Profit-first. Open-and-hold (0 closed trips + open MTM) can win."""
    _ = min_trips  # kept for call-site compat; no longer used as a gate
    tpd = round_trips / max(span_days, 0.25) if round_trips > 0 else 0.0
    active_days = int(np.sum(trades_by_day > 0)) if len(trades_by_day) else 0
    active_ratio = active_days / max(n_day_bins, 1)
    total = int(np.sum(trades_by_day)) if len(trades_by_day) else round_trips
    cluster = float(np.max(trades_by_day) / max(total, 1)) if total > 0 else 0.0

    dd_pen = max_dd_pct * 0.15
    cluster_pen = max(0.0, cluster - 0.70) * 8.0 if n_day_bins > 1 and total > 0 else 0.0
    scalp_pen = max(0.0, tpd - 4.0) * 3.0
    score = return_pct - dd_pen - cluster_pen - scalp_pen
    return score, tpd, active_ratio, cluster


def rank_results(results: list[TuneResult], *, span_days: float = 0.0) -> list[TuneResult]:
    """Highest profit first (holds included)."""
    _ = span_days
    return sorted(
        results,
        key=lambda r: (r.return_pct, r.score, r.open_legs, -r.round_trips),
        reverse=True,
    )


def _expand_grid(grid: dict[str, list[Any]], *, max_combos: int) -> list[dict[str, Any]]:
    """Sample up to max_combos without materializing the full cartesian product.

    Full product of DEFAULT_GRID alone is ~1.5M; with dump/rsi knobs it hits
    tens/hundreds of millions — list(product(...)) hangs the process after load.
    """
    keys = [k for k in ALL_PARAM_KEYS if k in grid and grid[k]]
    if not keys:
        return [{}]
    values = [list(grid[k]) for k in keys]
    sizes = [len(v) for v in values]
    total = 1
    for s in sizes:
        total *= s
        if total > 10**12:  # overflow guard
            total = 10**12
            break
    if total <= max_combos:
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # Uniform random sample of distinct combos (no full product in RAM).
    want = min(int(max_combos), int(total))
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    # Cap attempts so we never spin forever on tiny unique spaces.
    max_attempts = max(want * 40, want + 1000)
    attempts = 0
    while len(out) < want and attempts < max_attempts and len(seen) < total:
        attempts += 1
        combo = tuple(random.choice(v) for v in values)
        if combo in seen:
            continue
        seen.add(combo)
        out.append(dict(zip(keys, combo)))
    return out


def _grid_for_strategy(strategy: str, base_grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    g = dict(base_grid)
    name = str(strategy or "").lower()
    if any(tag in name for tag in ("dump", "btcdump")):
        g["DUMP_RET_PCT"] = INDICATOR_GRID["DUMP_RET_PCT"]
        g["DUMP_LOOKBACK_S"] = INDICATOR_GRID["DUMP_LOOKBACK_S"]
        g["DUMP_RANGE_PCT"] = INDICATOR_GRID["DUMP_RANGE_PCT"]
    if "trend" in name:
        g["TREND_BIAS_MIN"] = INDICATOR_GRID["TREND_BIAS_MIN"]
    if "vol" in name:
        g["MAX_ATR_PCT"] = INDICATOR_GRID["MAX_ATR_PCT"]
    if "rsi" in name and "mtf" not in name:
        g["RSI_MAX"] = INDICATOR_GRID["RSI_MAX"]
        g["RSI_MIN"] = INDICATOR_GRID["RSI_MIN"]
        g["DUMP_RET_PCT"] = INDICATOR_GRID["DUMP_RET_PCT"]
        g["DUMP_LOOKBACK_S"] = INDICATOR_GRID["DUMP_LOOKBACK_S"]
    if "mtf" in name:
        for k, vals in MTF_GRID.items():
            g[k] = list(vals)
    if "swing" in name:
        for k, vals in SWING_GRID.items():
            g[k] = list(vals)
    return g


def run_one(
    ds: ResearchDataset,
    base_cfg: Any,
    *,
    strategy: str,
    params: dict[str, Any],
    tick_interval: float,
    progress_cb: Callable[[str, int, int], None] | None = None,
    replay_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] | None = None,
) -> TuneResult:
    replay_fn = STRATEGY_REPLAYERS[strategy]
    cache_key = _replay_cache_key(strategy, params)
    cached = replay_cache.get(cache_key) if replay_cache is not None else None

    if cached is not None:
        target_coin, target_side = cached
        if progress_cb is not None:
            progress_cb("replay_cached", 1, 1)
    else:

        def _replay_progress(done: int, total: int) -> None:
            if progress_cb is not None:
                progress_cb("replay", done, total)

        target_coin, target_side = replay_fn(ds, base_cfg, params, progress_cb=_replay_progress)
        if replay_cache is not None:
            replay_cache[cache_key] = (target_coin, target_side)

    if progress_cb is not None:
        progress_cb("numba_sim", 0, 1)
    margin_frac, leverage = sim_size_from_cfg(base_cfg)
    sim = simulate_portfolio(
        ds.marks,
        target_coin,
        target_side,
        cooldown_s=float(params.get("REBALANCE_COOLDOWN_S", getattr(base_cfg, "REBALANCE_COOLDOWN_S", 240))),
        tick_interval_s=tick_interval,
        fee_rate=PMF_FEE_PCT,
        margin_frac=margin_frac,
        leverage=leverage,
        day_ids=ds.day_ids,
        max_slots=int(params.get("MAX_COINS_IN_BOOK", getattr(base_cfg, "MAX_COINS_IN_BOOK", 3)) or 3),
    )
    if progress_cb is not None:
        progress_cb("numba_sim", 1, 1)
    sc, tpd, active_ratio, cluster = score_backtest(
        return_pct=float(sim["return_pct"]),
        max_dd_pct=float(sim["max_dd_pct"]),
        round_trips=int(sim["round_trips"]),
        trades_by_day=sim["trades_by_day"],
        span_days=ds.span_days,
        n_day_bins=len(ds.day_labels),
    )
    return TuneResult(
        strategy=strategy,
        params=dict(params),
        return_pct=float(sim["return_pct"]),
        max_dd_pct=float(sim["max_dd_pct"]),
        round_trips=int(sim["round_trips"]),
        win_rate_pct=float(sim["win_rate_pct"]),
        score=sc,
        trades_per_day=tpd,
        active_day_ratio=active_ratio,
        cluster_share=cluster,
        total_fees=float(sim["total_fees"]),
        open_legs=int(sim.get("open_legs") or 0),
        meta={
            "margin_frac": margin_frac,
            "leverage": leverage,
            "live_listed": int(getattr(ds, "live_listed", 0) or 0),
            "live_holders": len(getattr(ds, "live_holder_addrs", set()) or []),
            "replay_cached": cached is not None,
        },
    )


def _build_combos(
    strategies: list[str],
    grid_for: Callable[[str], dict[str, list[Any]]],
    *,
    max_combos: int,
    label: str,
) -> dict[str, list[dict[str, Any]]]:
    print(f"{label}: building param grids for {len(strategies)} strategies...", flush=True)
    out: dict[str, list[dict[str, Any]]] = {}
    g0 = time.time()
    for i, strat in enumerate(strategies):
        combos = _expand_grid(grid_for(strat), max_combos=max_combos)
        # Cluster sim-only variants so replay cache hits consecutively.
        combos.sort(key=lambda p: _replay_cache_key(strat, p))
        out[strat] = combos
        _progress(i + 1, len(strategies), g0, label=f"grid {strat}")
    return out


def _run_search(
    ds: ResearchDataset,
    base_cfg: Any,
    strat_combos: dict[str, list[dict[str, Any]]],
    *,
    tick_iv: float,
    label: str,
) -> list[TuneResult]:
    jobs: list[tuple[str, dict[str, Any]]] = []
    for strat, combos in strat_combos.items():
        for params in combos:
            jobs.append((strat, params))
    jobs = sorted(jobs, key=_job_locality_key)
    total = max(1, len(jobs))
    workers = _worker_count(total)
    t0 = time.time()
    results: list[TuneResult] = []
    cache_hits = 0
    cfg_snap = _freeze_cfg(base_cfg)

    from .bt_replay import clear_replay_caches

    clear_replay_caches()

    print(
        f"{label}: {len(strat_combos)} strategies, {total} combos, ticks={ds.n_ticks}, "
        f"workers={workers} (panel+MTF cache+Numba PnL; parallel when ≥12 jobs)",
        flush=True,
    )
    _progress(0, total, t0, label="starting")

    # Single-worker path: fine-grained progress + no Windows pickle spawn cost.
    # Process pool only when enough jobs to amortize dataset pickle (~tens of seconds).
    # Windows spawn is costly; still worth it once jobs cover pickle+init (~10–15s).
    use_pool = workers > 1 and total >= 12
    if not use_pool:
        if workers > 1 and total < 12:
            print(
                f"  (using 1 worker for {total} combos — pool spawn overhead not worth it; "
                f"set PMF_BT_WORKERS or raise --max-combos for parallel)",
                flush=True,
            )
        replay_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
        done = 0
        for strat, params in jobs:

            def _inner(phase: str, cur: int, tot: int, _s=strat) -> None:
                tot = max(1, tot)
                frac_extra = (cur / tot) if phase.startswith("replay") else (0.95 if phase == "numba_sim" else 0.0)
                _progress(
                    done,
                    total,
                    t0,
                    label=f"{_s} {phase} {cur}/{tot}",
                    frac_extra=frac_extra,
                )

            try:
                r = run_one(
                    ds,
                    cfg_snap,
                    strategy=strat,
                    params=params,
                    tick_interval=tick_iv,
                    progress_cb=_inner,
                    replay_cache=replay_cache,
                )
                results.append(r)
                if r.meta.get("replay_cached"):
                    cache_hits += 1
            except Exception as exc:
                log.debug("Tune skip %s %s: %s", strat, params, exc)
            done += 1
            _progress(done, total, t0, label=strat)
        workers_used = 1
    else:
        done = 0
        workers_used = workers
        chunks = _chunk_jobs(jobs, workers)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_pool_init,
            initargs=(ds, cfg_snap, tick_iv),
        ) as pool:
            futs = {pool.submit(_pool_run_chunk, ch): len(ch) for ch in chunks}
            for fut in as_completed(futs):
                n_jobs = futs[fut]
                try:
                    payloads = fut.result() or []
                except Exception as exc:
                    log.debug("Tune worker chunk fail: %s", exc)
                    payloads = [None] * n_jobs
                for payload in payloads:
                    if payload is not None:
                        r = _result_from_dict(payload)
                        results.append(r)
                        if r.meta.get("replay_cached"):
                            cache_hits += 1
                    done += 1
                    if done == 1 or done % max(1, total // 100) == 0 or done >= total:
                        _progress(done, total, t0, label="pool")

    elapsed = time.time() - t0
    _progress(total, total, t0, label="done")
    print(
        f"{label} finished in {_fmt_dur(elapsed)} ({elapsed:.1f}s)  {done} combos  "
        f"workers={workers_used}  replay_cache_hits={cache_hits}",
        flush=True,
    )
    return results


def top_strategies(results: list[TuneResult], *, k: int) -> list[tuple[str, TuneResult]]:
    """Best result per strategy, best k strategies first.

    A strategy that never opened a position scores a flat 0.00% and would beat a
    slightly negative one — deep-diving that wastes the stage-2 budget, so active
    strategies are ranked ahead of inert ones.
    """
    best_by: dict[str, TuneResult] = {}
    for r in rank_results(results):
        if r.strategy not in best_by:
            best_by[r.strategy] = r
    ordered = rank_results(list(best_by.values()))
    active = [r for r in ordered if r.round_trips > 0 or r.open_legs > 0]
    inert = [r for r in ordered if r.round_trips <= 0 and r.open_legs <= 0]
    return [(r.strategy, r) for r in (active + inert)[:k]]


def tune_all(
    base_cfg: Any,
    data_dir: Path,
    *,
    max_days: int = 7,
    strategies: list[str] | None = None,
    grid: dict[str, list[Any]] | None = None,
    max_combos_per_strategy: int = 120,
    state_path: Path | None = None,
    two_stage: bool = True,
    top_k: int = 2,
    coarse_combos_per_strategy: int | None = None,
) -> tuple[ResearchDataset | None, list[TuneResult]]:
    """Two-stage search: wide+shallow across all strategies, then deep on top_k.

    Stage 1 sweeps each knob across its full sane range with few points, so every
    strategy is screened fairly and fast. Stage 2 spends the whole combo budget on
    the best `top_k` strategies with a dense grid centred on their stage-1 winner.
    """
    ds = build_dataset(data_dir, max_days=max_days, state_path=state_path, progress=True)
    if ds is None or ds.n_ticks < 3:
        return ds, []

    from .bt_replay import clear_replay_caches
    from .mtf_exec import clear_mtf_entry_cache

    clear_replay_caches()
    clear_mtf_entry_cache()

    strategies = [s for s in (strategies or list(LIVE_STRATEGIES)) if s in STRATEGY_REPLAYERS]
    grid = grid or DEFAULT_GRID
    tick_iv = _tick_interval_s(ds)

    # MTF entry signals are memoized in mtf_exec._ENTRY_CACHE. Full MTF_GRID product
    # is huge — skip bulk warm and let the global cache fill lazily across combos.
    if any("mtf" in str(s).lower() for s in strategies):
        mtf_product = 1
        for vals in MTF_GRID.values():
            mtf_product *= max(1, len(vals))
        if mtf_product > 64:
            print(
                f"MTF entry cache: lazy (skip bulk warm; grid product={mtf_product})",
                flush=True,
            )
        else:
            print("MTF entry cache: lazy (small grid; warm on first combo)", flush=True)

    if not two_stage or len(strategies) <= max(1, top_k):
        combos = _build_combos(
            strategies,
            lambda s: _grid_for_strategy(s, grid),
            max_combos=max_combos_per_strategy,
            label="Search",
        )
        results = _run_search(ds, base_cfg, combos, tick_iv=tick_iv, label="Search")
        results = rank_results(results, span_days=ds.span_days)
        _log_best(results, len(results))
        return ds, results

    coarse_n = int(
        coarse_combos_per_strategy
        if coarse_combos_per_strategy is not None
        else max(8, round(max_combos_per_strategy * 0.25))
    )
    stage1_combos = _build_combos(
        strategies,
        lambda s: coarse_grid(_grid_for_strategy(s, grid)),
        max_combos=coarse_n,
        label="Stage 1 (wide screen)",
    )
    stage1 = _run_search(ds, base_cfg, stage1_combos, tick_iv=tick_iv, label="Stage 1 (wide screen)")
    if not stage1:
        return ds, []

    finalists = top_strategies(stage1, k=max(1, top_k))
    print(
        "Stage 1 ranking: "
        + ", ".join(f"{name} {r.return_pct:+.2f}% (trips={r.round_trips})" for name, r in finalists),
        flush=True,
    )

    best_params = {name: dict(r.params) for name, r in finalists}
    deep_budget = max(max_combos_per_strategy, coarse_n * 4)
    stage2_combos = _build_combos(
        [name for name, _ in finalists],
        lambda s: deep_grid_around(_grid_for_strategy(s, grid), best_params[s]),
        max_combos=deep_budget,
        label="Stage 2 (deep dive)",
    )
    # Re-run the stage-1 winners in stage 2 so a deep grid can never lose to it.
    for name, r in finalists:
        if r.params not in stage2_combos[name]:
            stage2_combos[name].insert(0, dict(r.params))
    stage2 = _run_search(ds, base_cfg, stage2_combos, tick_iv=tick_iv, label="Stage 2 (deep dive)")

    results = rank_results(stage1 + stage2, span_days=ds.span_days)
    _log_best(results, len(stage1) + len(stage2))
    return ds, results


def _log_best(results: list[TuneResult], n_combos: int) -> None:
    best = results[0] if results else None
    log.info(
        "Tuned %s combos | best %s ret=%.2f%% trips=%s open=%s",
        n_combos,
        best.strategy if best else "-",
        best.return_pct if best else 0.0,
        best.round_trips if best else 0,
        best.open_legs if best else 0,
    )


def save_tuned(path: Path, best: TuneResult, *, dataset: ResearchDataset | None = None) -> None:
    payload = {
        "saved_at": time.time(),
        "fee_pct_per_side": PMF_FEE_PCT,
        "strategy": best.strategy,
        "params": best.params,
        "metrics": {
            "return_pct": best.return_pct,
            "max_dd_pct": best.max_dd_pct,
            "round_trips": best.round_trips,
            "win_rate_pct": best.win_rate_pct,
            "score": best.score,
            "trades_per_day": best.trades_per_day,
            "active_day_ratio": best.active_day_ratio,
            "cluster_share": best.cluster_share,
            "open_legs": best.open_legs,
        },
    }
    if dataset is not None:
        payload["dataset"] = {
            "ticks": dataset.n_ticks,
            "coins": dataset.n_coins,
            "span_days": dataset.span_days,
            "days": dataset.day_labels,
            "source": dataset.source,
            "live_holders": len(getattr(dataset, "live_holder_addrs", set()) or []),
            "live_listed": int(getattr(dataset, "live_listed", 0) or 0),
            "live_basket_target": int(getattr(dataset, "live_basket_target", 0) or 0),
            "holders_labeled": len(getattr(dataset, "holder_addrs", set()) or []),
        }
    if best.meta:
        payload["sim"] = {
            "margin_frac": best.meta.get("margin_frac"),
            "leverage": best.meta.get("leverage"),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_tuned(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
