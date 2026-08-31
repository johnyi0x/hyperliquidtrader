#!/usr/bin/env python3
"""One-shot multi-interval backtest/tune (no live orders)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from hyperliquid.utils import constants

import config as cfg
from src.data_files import housekeep_data_dir
from src.hl_rate_limit import RequestGuard, ThrottledInfo, default_shared_budget
from src.logger import setup_logger
from src.market_resolver import parse_coin_input
from src.pair_universe import resolve_pair_universe
from src.store import SetupStore
from src.tuner import run_full_tune, select_top_live_pairs

PROJECT = Path(__file__).resolve().parent


def main() -> None:
    load_dotenv(PROJECT / ".env")
    sibling = PROJECT.parent / "hyperliquid-rsi-bot" / ".env"
    if sibling.exists():
        load_dotenv(sibling, override=False)

    if not cfg.USE_TP_SL and not cfg.USE_EXIT_SIGNAL and not cfg.USE_MAX_HOLD:
        raise ValueError("Enable at least one exit layer in config.py")
    from src.candles import INTERVAL_MS

    bad = [i for i in cfg.INTERVALS if i not in INTERVAL_MS]
    if bad:
        raise ValueError(f"Unknown INTERVALS {bad}")

    pair_mode = str(getattr(cfg, "PAIR_SELECTION_MODE", "manual") or "manual").strip().lower()
    if pair_mode not in ("manual", "top_volume", "volume", "auto_volume"):
        raise ValueError(
            f"PAIR_SELECTION_MODE must be 'manual' or 'top_volume' (got {pair_mode!r})"
        )

    logger = setup_logger("hl-multi-backtest", PROJECT / "logs")
    data_dir = PROJECT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    housekeep_data_dir(data_dir, logger=logger)
    store = SetupStore(data_dir, logger, refresh_hours=cfg.BACKTEST_REFRESH_HOURS)

    base_url = (
        constants.TESTNET_API_URL if cfg.USE_TESTNET else constants.MAINNET_API_URL
    )
    info = ThrottledInfo(
        base_url,
        skip_ws=True,
        guard=RequestGuard(logger=logger, shared_budget=default_shared_budget()),
    )

    universe = resolve_pair_universe(
        info,
        mode=pair_mode,
        manual_pairs=tuple(cfg.PAIRS) if not isinstance(cfg.PAIRS, str) else (cfg.PAIRS,),
        top_volume_count=int(getattr(cfg, "TOP_VOLUME_COUNT", 50) or 50),
        include_xyz=bool(getattr(cfg, "INCLUDE_XYZ_PAIRS", False)),
        use_max_leverage=bool(getattr(cfg, "USE_MAX_LEVERAGE", True)),
        default_leverage=int(cfg.LEVERAGE),
        leverage_overrides=getattr(cfg, "PAIR_LEVERAGE", None),
        requested_leverage_for=cfg.requested_leverage_for,
        min_max_leverage=int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0),
        max_max_leverage=int(getattr(cfg, "MAX_MAX_LEVERAGE", 0) or 0),
        xyz_mode=cfg.xyz_pair_mode() if hasattr(cfg, "xyz_pair_mode") else None,
        logger=logger,
    )
    coins = []
    leverage_by_coin: dict[str, int] = {}
    for raw, lev in universe:
        sym, dex = parse_coin_input(raw)
        api = f"{dex}:{sym}" if dex else sym
        coins.append(api)
        leverage_by_coin[api] = int(lev)
        leverage_by_coin[sym] = int(lev)
        leverage_by_coin[raw] = int(lev)

    logger.info(
        "Backtest mode=%s pair_mode=%s pairs=%s intervals=%s exec=%s candles<=%s "
        "target≈%.1f/d max_live=%s lev_default=%sx",
        getattr(cfg, "STRATEGY_MODE", "mtf"),
        pair_mode,
        coins,
        list(cfg.INTERVALS),
        getattr(cfg, "MTF_EXEC_INTERVAL", "1m"),
        cfg.REQUESTED_CANDLES,
        cfg.TARGET_TRADES_PER_DAY,
        int(getattr(cfg, "MAX_LIVE_PAIRS", 5) or 5),
        cfg.LEVERAGE,
    )
    if pair_mode in ("top_volume", "volume", "auto_volume"):
        logger.info(
            "Top-volume: count=%s xyz_mode=%s use_max_lev=%s min_maxLev≥%s max_maxLev≤%s",
            int(getattr(cfg, "TOP_VOLUME_COUNT", 50) or 50),
            cfg.xyz_pair_mode() if hasattr(cfg, "xyz_pair_mode") else getattr(cfg, "INCLUDE_XYZ_PAIRS", False),
            bool(getattr(cfg, "USE_MAX_LEVERAGE", True)),
            int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0),
            int(getattr(cfg, "MAX_MAX_LEVERAGE", 0) or 0) or "off",
        )
    if getattr(cfg, "PAIR_LEVERAGE", None):
        logger.info("Per-pair leverage overrides: %s", cfg.PAIR_LEVERAGE)
    results = run_full_tune(
        info,
        coins,
        list(cfg.INTERVALS),
        leverage=cfg.LEVERAGE,
        leverage_by_coin=leverage_by_coin,
        data_dir=data_dir,
        requested_candles=cfg.REQUESTED_CANDLES,
        taker_fee_pct=cfg.TAKER_FEE_PCT,
        min_win_rate=cfg.MIN_WIN_RATE_PCT,
        target_trades_per_day=cfg.TARGET_TRADES_PER_DAY,
        min_trades_abs=cfg.MIN_TRADES_ABS,
        balance_grid=cfg.BALANCE_PCT_GRID,
        use_tpsl=cfg.USE_TP_SL,
        use_exit_signal=cfg.USE_EXIT_SIGNAL,
        use_max_hold=cfg.USE_MAX_HOLD,
        max_position_hours=cfg.MAX_POSITION_HOURS,
        allow_dca=cfg.ALLOW_DCA,
        dca_max_adds=int(getattr(cfg, "DCA_MAX_ADDS", 1) or 1),
        screen_top_n=cfg.SCREEN_TOP_N,
        keep_best_per_interval=cfg.KEEP_BEST_PER_INTERVAL,
        strategy_mode=getattr(cfg, "STRATEGY_MODE", "mtf"),
        mtf_exec_interval=getattr(cfg, "MTF_EXEC_INTERVAL", "1m"),
        tune_profile=getattr(cfg, "TUNE_PROFILE", "fast"),
        logger=logger,
    )
    if results:
        max_live = int(getattr(cfg, "MAX_LIVE_PAIRS", 5) or 5)
        results = select_top_live_pairs(results, max_live, logger=logger)
        store.save_results(results, leverage=cfg.LEVERAGE)
        logger.info("Done: %s", store.describe())
    else:
        logger.warning("No winners found")


if __name__ == "__main__":
    main()
