"""
Multi-interval staged Numba tuner.

Stage 1: screen all entry combos with a cheap baseline exit (fast).
Stage 2: refine top-N entries with TP/SL × exit-signal × DCA × balance grids.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .candles import INTERVAL_MS, fetch_closed_candles, probe_max_candles
from .data_files import append_jsonl
from .indicators import build_features
from .mtf import (
    arrays_from_candles,
    combined_entry_mask,
    default_min_agree,
    mtf_param_grid,
    prepare_interval_biases,
    strategy_label,
)
from .registry import (
    STRATEGY_BY_ID,
    build_entry_mask,
    build_exit_mask,
    iter_entry_combos,
    iter_refine_combos,
    screen_exit_bundle,
)
from .sim_numba import TAKER_FEE_PCT, simulate
from .tune_profile import resolve_tune_profile


def _arrays(candles: list[dict]) -> tuple[np.ndarray, ...]:
    closes = np.array([float(c["c"]) for c in candles], dtype=np.float64)
    highs = np.array([float(c["h"]) for c in candles], dtype=np.float64)
    lows = np.array([float(c["l"]) for c in candles], dtype=np.float64)
    vols = np.array([float(c.get("v", 0) or 0) for c in candles], dtype=np.float64)
    return closes, highs, lows, vols


def _sample_days(n_bars: int, interval: str) -> float:
    step = INTERVAL_MS.get(interval, 60_000) / 1000.0
    bars_per_day = max(1.0, 86400.0 / step)
    return max(1.0, n_bars / bars_per_day)


def _rank(
    score: float,
    tpd: float,
    target_tpd: float,
    *,
    return_pct: float = 0.0,
    max_dd_pct: float = 0.0,
    win_rate_pct: float = 0.0,
    balance_pct: float = 30.0,
) -> float:
    """
    Rank setups size-neutrally so 95% balance does not auto-win vs 20%.
    Uses unitized return/DD (÷ balance fraction) + frequency preference.
    """
    scale = max(0.05, float(balance_pct) / 100.0)
    unit = (float(return_pct) / scale) - 0.6 * (float(max_dd_pct) / scale) + 0.12 * float(
        win_rate_pct
    )
    # Blend raw score lightly in case return fields missing
    blended = 0.85 * unit + 0.15 * float(score)
    target = max(0.5, float(target_tpd))
    freq = min(tpd, target * 2.0) / target
    return float(blended) * (0.35 + 0.65 * freq)


def _rank_from_stats(
    stats: dict[str, Any],
    tpd: float,
    target_tpd: float,
    balance_pct: float,
) -> float:
    return round(
        _rank(
            float(stats.get("score", 0.0)),
            tpd,
            target_tpd,
            return_pct=float(stats.get("return_pct", 0.0)),
            max_dd_pct=float(stats.get("max_dd_pct", 0.0)),
            win_rate_pct=float(stats.get("win_rate_pct", 0.0)),
            balance_pct=float(balance_pct),
        ),
        4,
    )


def _max_hold_bars(interval: str, hours: float) -> int:
    step = INTERVAL_MS.get(interval, 60_000) / 1000.0
    return max(1, int(round(hours * 3600.0 / step)))


def tune_coin_interval(
    info: Any,
    coin: str,
    interval: str,
    *,
    leverage: int,
    data_dir: Path,
    requested_candles: int,
    max_candles: int,
    taker_fee_pct: float,
    min_win_rate: float,
    target_trades_per_day: float,
    min_trades_abs: int,
    balance_grid: tuple[float, ...],
    use_tpsl: bool,
    use_exit_signal: bool,
    use_max_hold: bool,
    max_position_hours: float,
    allow_dca: bool,
    screen_top_n: int,
    refine_profile: str = "full",
    dca_max_adds: int = 1,
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    log = logger or logging.getLogger("hl-multi")
    candles = fetch_closed_candles(
        info,
        coin,
        interval,
        requested_candles,
        max_candles=max_candles,
        data_dir=data_dir,
        logger=log,
    )
    if len(candles) < 80:
        log.warning("%s %s: only %s candles — skip", coin, interval, len(candles))
        return None

    closes, highs, lows, vols = _arrays(candles)
    feats = build_features(closes, highs, lows, vols)
    days = _sample_days(len(closes), interval)
    hold_bars = _max_hold_bars(interval, max_position_hours)
    min_trades = max(min_trades_abs, int(target_trades_per_day * days * 0.3))
    min_tpd = max(0.4, target_trades_per_day * 0.45)

    entries = iter_entry_combos()
    baseline = screen_exit_bundle(use_tpsl=use_tpsl, use_max_hold=use_max_hold)
    log.info(
        "Tune %s @ %s: %s bars (%.1fd) | screen %s entries | hold=%sbars | "
        "min_trades=%s min_wr=%.0f target_tpd=%.1f",
        coin,
        interval,
        len(closes),
        days,
        len(entries),
        hold_bars,
        min_trades,
        min_win_rate,
        target_trades_per_day,
    )

    entry_cache: dict[tuple, np.ndarray] = {}
    screened: list[dict[str, Any]] = []
    t0 = time.time()

    for ent in entries:
        key = (ent["sid"], ent["p0"], ent["p1"], ent["p2"], ent["aux"])
        if key not in entry_cache:
            entry_cache[key] = build_entry_mask(
                ent["sid"], feats, ent["p0"], ent["p1"], ent["p2"], ent["aux"]
            )
        mask = entry_cache[key]
        if not mask.any():
            continue
        stats = simulate(
            closes,
            highs,
            lows,
            mask,
            np.zeros(len(closes), dtype=np.bool_),
            side=ent["side"],
            tp_pct=baseline["tp_pct"],
            sl_pct=baseline["sl_pct"],
            leverage=leverage,
            balance_pct=baseline["balance_pct"],
            taker_fee_pct=taker_fee_pct,
            use_tpsl=baseline["use_tpsl"],
            use_exit_signal=False,
            use_max_hold=baseline["use_max_hold"],
            max_hold_bars=hold_bars,
        )
        if stats["trades"] < min_trades or stats["win_rate_pct"] < min_win_rate:
            continue
        if stats["score"] <= 0:
            continue
        tpd = stats["trades"] / days
        screened.append(
            {
                **ent,
                **stats,
                "trades_per_day": round(tpd, 3),
                "rank_score": _rank_from_stats(
                    stats, tpd, target_trades_per_day, baseline["balance_pct"]
                ),
            }
        )

    if not screened:
        log.warning("%s %s: no entries passed screen (%.1fs)", coin, interval, time.time() - t0)
        return None

    screened.sort(key=lambda r: r["rank_score"], reverse=True)
    top = screened[: max(1, screen_top_n)]
    refine = iter_refine_combos(
        use_tpsl=use_tpsl,
        use_exit_signal=use_exit_signal,
        use_max_hold=use_max_hold,
        allow_dca=allow_dca,
        balance_grid=balance_grid,
        profile=refine_profile,
        dca_max_adds=dca_max_adds,
    )
    log.info(
        "%s %s: refining top %s entries × %s exit/DCA/bal combos [%s]",
        coin,
        interval,
        len(top),
        len(refine),
        refine_profile,
    )

    exit_cache: dict[tuple, np.ndarray] = {}
    best: dict[str, Any] | None = None
    tested = 0
    for ent in top:
        ekey = (ent["sid"], ent["p0"], ent["p1"], ent["p2"], ent["aux"])
        mask = entry_cache[ekey]
        for ref in refine:
            xkey = (ref["exit_eid"], ref["ex_p0"], ref["ex_aux"], ent["side"])
            if xkey not in exit_cache:
                exit_cache[xkey] = build_exit_mask(
                    ref["exit_eid"],
                    feats,
                    ent["side"],
                    ref["ex_p0"],
                    ref["ex_aux"],
                )
            xmask = exit_cache[xkey]
            stats = simulate(
                closes,
                highs,
                lows,
                mask,
                xmask,
                side=ent["side"],
                tp_pct=ref["tp_pct"],
                sl_pct=ref["sl_pct"],
                leverage=leverage,
                balance_pct=ref["balance_pct"],
                taker_fee_pct=taker_fee_pct,
                use_tpsl=ref["use_tpsl"],
                use_exit_signal=ref["use_exit_signal"],
                exit_eid=ref["exit_eid"],
                exit_snap_pct=ref["ex_p0"],
                use_max_hold=ref["use_max_hold"],
                max_hold_bars=hold_bars,
                dca_enabled=ref["dca_enabled"],
                dca_trigger_pct=ref["dca_trigger_pct"],
                dca_max_adds=ref["dca_max_adds"],
                dca_size_mult=ref["dca_size_mult"],
            )
            tested += 1
            if stats["trades"] < min_trades or stats["win_rate_pct"] < min_win_rate:
                continue
            if stats["score"] <= 0:
                continue
            tpd = stats["trades"] / days
            row = {
                **{k: ent[k] for k in ("sid", "name", "side", "p0", "p1", "p2", "aux")},
                **ref,
                **stats,
                "coin": coin,
                "interval": interval,
                "bars": len(closes),
                "days": round(days, 3),
                "max_hold_bars": hold_bars,
                "trades_per_day": round(tpd, 3),
                "rank_score": _rank_from_stats(
                    stats, tpd, target_trades_per_day, ref["balance_pct"]
                ),
            }
            # Prefer setups that clear the frequency floor, then highest rank.
            def _better(a: dict[str, Any], b: dict[str, Any]) -> bool:
                a_ok = a["trades_per_day"] >= min_tpd
                b_ok = b["trades_per_day"] >= min_tpd
                if a_ok != b_ok:
                    return a_ok
                return a["rank_score"] > b["rank_score"]

            if best is None or _better(row, best):
                best = row

    elapsed = time.time() - t0
    if best is None:
        # Fall back to best screened entry + baseline
        s = top[0]
        best = {
            **{k: s[k] for k in ("sid", "name", "side", "p0", "p1", "p2", "aux")},
            **baseline,
            "return_pct": s["return_pct"],
            "max_dd_pct": s["max_dd_pct"],
            "trades": s["trades"],
            "wins": s.get("wins", 0),
            "win_rate_pct": s["win_rate_pct"],
            "fees_usd": s.get("fees_usd", 0),
            "score": s["score"],
            "profit_factor": s.get("profit_factor", 0),
            "coin": coin,
            "interval": interval,
            "bars": len(closes),
            "days": round(days, 3),
            "max_hold_bars": hold_bars,
            "trades_per_day": s["trades_per_day"],
            "rank_score": s["rank_score"],
        }
        log.warning("%s %s: refine empty — using screen winner", coin, interval)

    sdef = STRATEGY_BY_ID[best["sid"]]
    log.info(
        "BEST %s @ %s: %s %s p0=%.2f | tp/sl=%.2f exit=%s dca=%s bal=%.0f%% | "
        "rank=%.2f ret=%.1f%% wr=%.0f%% trades=%s (%.1f/d) dd=%.1f%% [%.1fs %s sims]",
        coin,
        interval,
        sdef.name,
        "LONG" if best["side"] > 0 else "SHORT",
        best["p0"],
        best.get("tp_pct", 0),
        best.get("exit_name", "none"),
        "ON" if best.get("dca_enabled") else "OFF",
        best.get("balance_pct", 100),
        best["rank_score"],
        best["return_pct"],
        best["win_rate_pct"],
        best["trades"],
        best["trades_per_day"],
        best["max_dd_pct"],
        elapsed,
        tested,
    )
    return best


def tune_coin_mtf(
    info: Any,
    coin: str,
    intervals: list[str],
    *,
    exec_interval: str,
    leverage: int,
    data_dir: Path,
    requested_candles: int,
    max_candles: int,
    taker_fee_pct: float,
    min_win_rate: float,
    target_trades_per_day: float,
    min_trades_abs: int,
    balance_grid: tuple[float, ...],
    use_tpsl: bool,
    use_exit_signal: bool,
    use_max_hold: bool,
    max_position_hours: float,
    allow_dca: bool,
    screen_top_n: int,
    refine_profile: str = "full",
    mtf_profile: str = "full",
    dca_max_adds: int = 1,
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    """
    Tune one coin using ALL intervals jointly:
    multi-TF consensus permission AND LTF entry trigger on exec_interval.
    """
    log = logger or logging.getLogger("hl-multi")
    ivs = [i for i in intervals if i in INTERVAL_MS]
    if exec_interval not in ivs:
        ivs = [exec_interval] + ivs
    # Stable unique order by period ascending
    ivs = sorted(dict.fromkeys(ivs), key=lambda x: INTERVAL_MS.get(x, 0))

    candles_by: dict[str, list[dict[str, Any]]] = {}
    for iv in ivs:
        candles_by[iv] = fetch_closed_candles(
            info,
            coin,
            iv,
            requested_candles,
            max_candles=max_candles,
            data_dir=data_dir,
            logger=log,
        )
    exec_candles = candles_by.get(exec_interval) or []
    if len(exec_candles) < 80:
        log.warning(
            "%s MTF: exec %s only %s bars — skip",
            coin,
            exec_interval,
            len(exec_candles),
        )
        return None
    usable = sum(1 for v in candles_by.values() if len(v) >= 60)
    if usable < 2:
        log.warning("%s MTF: need ≥2 intervals with data — got %s", coin, usable)
        return None

    closes, highs, lows, _vols = arrays_from_candles(exec_candles)
    feats = build_features(closes, highs, lows, _vols)
    days = _sample_days(len(closes), exec_interval)
    hold_bars = _max_hold_bars(exec_interval, max_position_hours)
    # Confluence is rarer — soften absolute floor slightly vs single-TF
    min_trades = max(min_trades_abs, int(target_trades_per_day * days * 0.15))
    min_tpd = max(0.2, target_trades_per_day * 0.25)
    screen_agree = max(2, default_min_agree(usable) - 1)
    screen_mtf = {
        "mtf_ema": 50,
        "mtf_min_agree": screen_agree,
        "mtf_min_score": 0.25,
        "mtf_weight_power": 0.5,
    }
    # Stage-1 screen uses a slightly softer WR so refine can still harden later.
    screen_wr = max(50.0, float(min_win_rate) - 2.0)

    entries = iter_entry_combos()
    baseline = screen_exit_bundle(use_tpsl=use_tpsl, use_max_hold=use_max_hold)
    log.info(
        "Tune %s MTF exec=%s using %s | %s bars (%.1fd) | screen %s triggers | "
        "consensus≥%s score≥%.2f | hold=%sbars min_trades=%s | profile=%s/%s",
        coin,
        exec_interval,
        ",".join(ivs),
        len(closes),
        days,
        len(entries),
        screen_agree,
        screen_mtf["mtf_min_score"],
        hold_bars,
        min_trades,
        refine_profile,
        mtf_profile,
    )

    entry_cache: dict[tuple, np.ndarray] = {}
    bias_cache: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    screened: list[dict[str, Any]] = []
    t0 = time.time()

    def _biases_for(ema_period: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        key = int(ema_period)
        cached = bias_cache.get(key)
        if cached is not None:
            return cached
        by_iv = dict(candles_by)
        by_iv[exec_interval] = exec_candles
        built = prepare_interval_biases(by_iv, ema_period=key)
        bias_cache[key] = built
        return built

    def _mask_for(ent: dict[str, Any], mtf: dict[str, Any]) -> np.ndarray | None:
        key = (
            ent["sid"],
            ent["p0"],
            ent["p1"],
            ent["p2"],
            ent["aux"],
            mtf["mtf_ema"],
            mtf["mtf_min_agree"],
            mtf["mtf_min_score"],
            mtf["mtf_weight_power"],
        )
        if key in entry_cache:
            return entry_cache[key]
        packed = combined_entry_mask(
            exec_interval=exec_interval,
            exec_candles=exec_candles,
            candles_by_interval=candles_by,
            sid=ent["sid"],
            side=ent["side"],
            p0=ent["p0"],
            p1=ent["p1"],
            p2=ent["p2"],
            aux=ent["aux"],
            ema_period=int(mtf["mtf_ema"]),
            min_agree=int(mtf["mtf_min_agree"]),
            min_score=float(mtf["mtf_min_score"]),
            weight_power=float(mtf["mtf_weight_power"]),
            biases=_biases_for(int(mtf["mtf_ema"])),
        )
        if packed is None:
            entry_cache[key] = np.zeros(len(closes), dtype=np.bool_)
            return entry_cache[key]
        mask, _, _ = packed
        entry_cache[key] = mask
        return mask

    for ent in entries:
        mask = _mask_for(ent, screen_mtf)
        if mask is None or not mask.any():
            continue
        stats = simulate(
            closes,
            highs,
            lows,
            mask,
            np.zeros(len(closes), dtype=np.bool_),
            side=ent["side"],
            tp_pct=baseline["tp_pct"],
            sl_pct=baseline["sl_pct"],
            leverage=leverage,
            balance_pct=baseline["balance_pct"],
            taker_fee_pct=taker_fee_pct,
            use_tpsl=baseline["use_tpsl"],
            use_exit_signal=False,
            use_max_hold=baseline["use_max_hold"],
            max_hold_bars=hold_bars,
        )
        if stats["trades"] < min_trades or stats["win_rate_pct"] < screen_wr:
            continue
        if stats["score"] <= 0:
            continue
        tpd = stats["trades"] / days
        screened.append(
            {
                **ent,
                **stats,
                **screen_mtf,
                "trades_per_day": round(tpd, 3),
                "rank_score": _rank_from_stats(
                    stats, tpd, target_trades_per_day, baseline["balance_pct"]
                ),
            }
        )

    if not screened:
        log.warning("%s MTF: no triggers passed consensus screen (%.1fs)", coin, time.time() - t0)
        return None

    screened.sort(key=lambda r: r["rank_score"], reverse=True)
    top = screened[: max(1, screen_top_n)]
    refine = iter_refine_combos(
        use_tpsl=use_tpsl,
        use_exit_signal=use_exit_signal,
        use_max_hold=use_max_hold,
        allow_dca=allow_dca,
        balance_grid=balance_grid,
        profile=refine_profile,
        dca_max_adds=dca_max_adds,
    )
    mtf_grid = mtf_param_grid(usable, profile=mtf_profile)
    log.info(
        "%s MTF: refining top %s triggers × %s consensus × %s exit/DCA/bal [%s]",
        coin,
        len(top),
        len(mtf_grid),
        len(refine),
        refine_profile,
    )

    exit_cache: dict[tuple, np.ndarray] = {}
    best: dict[str, Any] | None = None
    tested = 0

    def _better(a: dict[str, Any], b: dict[str, Any]) -> bool:
        a_ok = a["trades_per_day"] >= min_tpd
        b_ok = b["trades_per_day"] >= min_tpd
        if a_ok != b_ok:
            return a_ok
        return a["rank_score"] > b["rank_score"]

    for ent in top:
        for mtf in mtf_grid:
            mask = _mask_for(ent, mtf)
            if mask is None or not mask.any():
                continue
            for ref in refine:
                xkey = (ref["exit_eid"], ref["ex_p0"], ref["ex_aux"], ent["side"])
                if xkey not in exit_cache:
                    exit_cache[xkey] = build_exit_mask(
                        ref["exit_eid"],
                        feats,
                        ent["side"],
                        ref["ex_p0"],
                        ref["ex_aux"],
                    )
                xmask = exit_cache[xkey]
                stats = simulate(
                    closes,
                    highs,
                    lows,
                    mask,
                    xmask,
                    side=ent["side"],
                    tp_pct=ref["tp_pct"],
                    sl_pct=ref["sl_pct"],
                    leverage=leverage,
                    balance_pct=ref["balance_pct"],
                    taker_fee_pct=taker_fee_pct,
                    use_tpsl=ref["use_tpsl"],
                    use_exit_signal=ref["use_exit_signal"],
                    exit_eid=ref["exit_eid"],
                    exit_snap_pct=ref["ex_p0"],
                    use_max_hold=ref["use_max_hold"],
                    max_hold_bars=hold_bars,
                    dca_enabled=ref["dca_enabled"],
                    dca_trigger_pct=ref["dca_trigger_pct"],
                    dca_max_adds=ref["dca_max_adds"],
                    dca_size_mult=ref["dca_size_mult"],
                )
                tested += 1
                if stats["trades"] < min_trades or stats["win_rate_pct"] < min_win_rate:
                    continue
                if stats["score"] <= 0:
                    continue
                tpd = stats["trades"] / days
                row = {
                    **{k: ent[k] for k in ("sid", "side", "p0", "p1", "p2", "aux")},
                    "name": strategy_label(ent["sid"], ent["side"]),
                    **ref,
                    **stats,
                    **mtf,
                    "mode": "mtf",
                    "coin": coin,
                    "interval": exec_interval,
                    "mtf_intervals": list(ivs),
                    "bars": len(closes),
                    "days": round(days, 3),
                    "max_hold_bars": hold_bars,
                    "trades_per_day": round(tpd, 3),
                    "rank_score": _rank_from_stats(
                        stats, tpd, target_trades_per_day, ref["balance_pct"]
                    ),
                }
                if best is None or _better(row, best):
                    best = row

    elapsed = time.time() - t0
    if best is None:
        s = top[0]
        best = {
            **{k: s[k] for k in ("sid", "side", "p0", "p1", "p2", "aux")},
            "name": strategy_label(s["sid"], s["side"]),
            **baseline,
            **{k: s[k] for k in screen_mtf},
            "return_pct": s["return_pct"],
            "max_dd_pct": s["max_dd_pct"],
            "trades": s["trades"],
            "wins": s.get("wins", 0),
            "win_rate_pct": s["win_rate_pct"],
            "fees_usd": s.get("fees_usd", 0),
            "score": s["score"],
            "profit_factor": s.get("profit_factor", 0),
            "mode": "mtf",
            "coin": coin,
            "interval": exec_interval,
            "mtf_intervals": list(ivs),
            "bars": len(closes),
            "days": round(days, 3),
            "max_hold_bars": hold_bars,
            "trades_per_day": s["trades_per_day"],
            "rank_score": s["rank_score"],
        }
        log.warning("%s MTF: refine empty — using screen winner", coin)

    log.info(
        "BEST %s MTF@%s: %s %s | agree≥%s score≥%.2f ema=%s | tp/sl=%.2f exit=%s "
        "dca=%s bal=%.0f%% | rank=%.2f ret=%.1f%% wr=%.0f%% trades=%s (%.1f/d) "
        "dd=%.1f%% [%.1fs %s sims]",
        coin,
        exec_interval,
        best["name"],
        "LONG" if best["side"] > 0 else "SHORT",
        best.get("mtf_min_agree"),
        best.get("mtf_min_score"),
        best.get("mtf_ema"),
        best.get("tp_pct", 0),
        best.get("exit_name", "none"),
        "ON" if best.get("dca_enabled") else "OFF",
        best.get("balance_pct", 100),
        best["rank_score"],
        best["return_pct"],
        best["win_rate_pct"],
        best["trades"],
        best["trades_per_day"],
        best["max_dd_pct"],
        elapsed,
        tested,
    )
    return best


def run_full_tune(
    info: Any,
    coins: list[str],
    intervals: list[str],
    *,
    leverage: int,
    data_dir: Path,
    requested_candles: int = 5000,
    taker_fee_pct: float = TAKER_FEE_PCT,
    min_win_rate: float = 52.0,
    target_trades_per_day: float = 4.0,
    min_trades_abs: int = 5,
    balance_grid: tuple[float, ...] = (50.0, 75.0, 100.0),
    use_tpsl: bool = True,
    use_exit_signal: bool = True,
    use_max_hold: bool = True,
    max_position_hours: float = 4.0,
    allow_dca: bool = True,
    dca_max_adds: int = 1,
    screen_top_n: int = 12,
    keep_best_per_interval: bool = False,
    strategy_mode: str = "mtf",
    mtf_exec_interval: str = "1m",
    leverage_by_coin: dict[str, int] | None = None,
    tune_profile: str = "fast",
    logger: logging.Logger | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Returns {coin: [winning setup(s)]}.

    strategy_mode:
      - "mtf": one joint multi-TF consensus setup per coin (uses ALL intervals)
      - "legacy": independent per-interval winners (old behavior)

    tune_profile:
      - "fast": smaller refine/consensus grids (default)
      - "full": original thorough grids

    leverage_by_coin: optional map api_coin/symbol → leverage; falls back to `leverage`.
    """
    log = logger or logging.getLogger("hl-multi")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    mode = (strategy_mode or "mtf").strip().lower()
    prof = resolve_tune_profile(
        tune_profile,
        balance_grid=balance_grid,
        screen_top_n=screen_top_n,
    )
    balance_grid = tuple(prof["balance_grid"])
    screen_top_n = int(prof["screen_top_n"])
    refine_profile = str(prof["refine_profile"])
    mtf_profile = str(prof["mtf_profile"])
    log.info(
        "Tune profile=%s | screen_top_n=%s | balance=%s | refine=%s | mtf=%s",
        prof["profile"],
        screen_top_n,
        balance_grid,
        refine_profile,
        mtf_profile,
    )
    lev_map = {str(k): max(1, int(v)) for k, v in (leverage_by_coin or {}).items()}

    def _lev_for(coin: str) -> int:
        if coin in lev_map:
            return lev_map[coin]
        bare = coin.split(":", 1)[-1] if ":" in coin else coin
        if bare in lev_map:
            return lev_map[bare]
        for k, v in lev_map.items():
            if k.upper() == coin.upper() or k.upper() == bare.upper():
                return v
        return max(1, int(leverage))

    probe_coin = coins[0] if coins else "BTC"
    probe_iv = mtf_exec_interval if mode == "mtf" else (intervals[0] if intervals else "1m")
    max_candles = probe_max_candles(
        info, probe_coin, probe_iv, data_dir=data_dir, logger=log
    )

    results: dict[str, list[dict[str, Any]]] = {}
    detail = data_dir / "tuning.jsonl"

    def _append_detail(winner: dict[str, Any]) -> None:
        try:
            append_jsonl(
                detail,
                {"ts": datetime.now(timezone.utc).isoformat(), **winner},
                logger=log,
            )
        except OSError:
            pass

    if mode == "mtf":
        exec_iv = mtf_exec_interval if mtf_exec_interval in INTERVAL_MS else "1m"
        log.info(
            "MTF mode — joint consensus across %s | exec=%s",
            ",".join(intervals),
            exec_iv,
        )
        for coin in coins:
            coin_lev = _lev_for(coin)
            try:
                winner = tune_coin_mtf(
                    info,
                    coin,
                    list(intervals),
                    exec_interval=exec_iv,
                    leverage=coin_lev,
                    data_dir=data_dir,
                    requested_candles=requested_candles,
                    max_candles=max_candles,
                    taker_fee_pct=taker_fee_pct,
                    min_win_rate=min_win_rate,
                    target_trades_per_day=target_trades_per_day,
                    min_trades_abs=min_trades_abs,
                    balance_grid=balance_grid,
                    use_tpsl=use_tpsl,
                    use_exit_signal=use_exit_signal,
                    use_max_hold=use_max_hold,
                    max_position_hours=max_position_hours,
                    allow_dca=allow_dca,
                    dca_max_adds=dca_max_adds,
                    screen_top_n=screen_top_n,
                    refine_profile=refine_profile,
                    mtf_profile=mtf_profile,
                    logger=log,
                )
            except Exception as exc:
                log.exception("MTF tune failed %s: %s", coin, exc)
                continue
            if winner:
                winner["leverage"] = coin_lev
                results[coin] = [winner]
                _append_detail(winner)
        return results

    # ----- legacy: independent per-interval -----
    for coin in coins:
        coin_lev = _lev_for(coin)
        per_iv: list[dict[str, Any]] = []
        for interval in intervals:
            try:
                winner = tune_coin_interval(
                    info,
                    coin,
                    interval,
                    leverage=coin_lev,
                    data_dir=data_dir,
                    requested_candles=requested_candles,
                    max_candles=max_candles,
                    taker_fee_pct=taker_fee_pct,
                    min_win_rate=min_win_rate,
                    target_trades_per_day=target_trades_per_day,
                    min_trades_abs=min_trades_abs,
                    balance_grid=balance_grid,
                    use_tpsl=use_tpsl,
                    use_exit_signal=use_exit_signal,
                    use_max_hold=use_max_hold,
                    max_position_hours=max_position_hours,
                    allow_dca=allow_dca,
                    dca_max_adds=dca_max_adds,
                    screen_top_n=screen_top_n,
                    refine_profile=refine_profile,
                    logger=log,
                )
            except Exception as exc:
                log.exception("Tune failed %s %s: %s", coin, interval, exc)
                continue
            if winner:
                winner.setdefault("mode", "legacy")
                winner["leverage"] = coin_lev
                per_iv.append(winner)
                _append_detail(winner)

        if not per_iv:
            continue
        if keep_best_per_interval:
            results[coin] = per_iv
        else:
            results[coin] = [max(per_iv, key=lambda r: r["rank_score"])]

    return results


def select_top_live_pairs(
    results: dict[str, list[dict[str, Any]]],
    max_pairs: int,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    If more than max_pairs coins have winners, keep the top max_pairs by
    best rank_score. Otherwise return results unchanged.
    """
    log = logger or logging.getLogger("hl-multi")
    if max_pairs <= 0 or len(results) <= max_pairs:
        return results

    ranked: list[tuple[float, str, list[dict[str, Any]]]] = []
    for coin, rows in results.items():
        if not rows:
            continue
        best = max(rows, key=lambda r: float(r.get("rank_score", 0.0) or 0.0))
        ranked.append((float(best.get("rank_score", 0.0) or 0.0), coin, rows))
    ranked.sort(key=lambda t: t[0], reverse=True)

    kept = ranked[:max_pairs]
    dropped = ranked[max_pairs:]
    out = {coin: rows for _, coin, rows in kept}
    log.info(
        "Pair cap: kept top %s/%s by rank → %s",
        max_pairs,
        len(results),
        ", ".join(f"{c}({r:.1f})" for r, c, _ in kept),
    )
    if dropped:
        log.info(
            "Pair cap: dropped %s",
            ", ".join(f"{c}({r:.1f})" for r, c, _ in dropped),
        )
    return out
