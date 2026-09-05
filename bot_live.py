#!/usr/bin/env python3
"""
Hyperliquid multi-interval strategy bot (paper or live).

First run: auto-tunes if no saved params (run_backtest.py optional).
While running + flat: retunes every BACKTEST_REFRESH_HOURS.

Paper and live use the same OrderExecutor path — only the client backend differs
(PaperHyperliquidClient vs HyperliquidClient). Entry/exit/DCA rules come from
src.engine (same masks as the Numba backtest).

Run:  python bot_live.py
Tune: python run_backtest.py
Paper balance: python check_paper.py
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from hyperliquid.utils import constants

import config as cfg
from src.candles import INTERVAL_MS
from src.data_files import housekeep_data_dir
from src.candle_book import snapshot_weight_budget
from src.ema_dev import (
    EmaDevStore,
    EmaDevTrade,
    adverse_pct,
    fill_pcts as ema_dev_fill_pcts,
    pick_farthest,
    protect_pcts as ema_dev_protect_pcts,
    signed_dev_pct,
    snap_from_candles,
    should_tp as ema_dev_should_tp,
    tp_price_from_dev as ema_dev_tp_price_from_dev,
    tp_through_pct as ema_dev_tp_through_pct,
)
from src.engine import (
    dca_should_add,
    entry_signal,
    exit_signal,
    live_dca_leg_count,
    setup_to_dict,
)
from src.exchange_client import HyperliquidClient
from src.hl_rate_limit import (
    RequestGuard,
    ThrottledInfo,
    _is_transient_network_error,
    default_shared_budget,
)
from src.logger import setup_logger
from src.hft_pingpong import (
    HftStore,
    book_from_l2,
    chop_from_candles,
    chop_reject_reason,
    decide as hft_decide,
    filter_hft_entries,
    quote_px_ok,
    rank_chop,
    target_clip_notional,
)
from src.pricing import (
    ceil_size,
    maker_limit_price,
    mid_post_only_price,
    round_size,
)
from src.market_resolver import parse_coin_input, sdk_perp_dexs_for_dexes
from src.mtf import mtf_consensus_snapshot
from src.order_executor import OrderExecutor
from src.pair_universe import (
    MOVER_TUNE_LOCK,
    allowed_sides_for_movers,
    is_auto_pair_mode,
    is_mover_mode,
    is_volume_mode,
    mover_tune_side,
    resolve_pair_universe,
)
from src.paper_broker import PaperHyperliquidClient
from src.position_guard import (
    cleanup_closed_coin,
    cleanup_when_flat,
    ensure_protected_position,
    wait_until_flat,
)
from src.store import LiveSetup, SetupStore, _row_to_setup
from src.trade_state import TradeStateStore
from src.tuner import run_full_tune, select_top_live_pairs
from src.watchlist import PairSetup, activate_pair, activate_pair_for_trade, resolve_watchlist

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = PROJECT_DIR / "logs"
ACTIVE_SETUP_PATH = DATA_DIR / "active_setup.json"


def load_secrets() -> tuple[str, str]:
    load_dotenv(PROJECT_DIR / ".env")
    sibling = PROJECT_DIR.parent / "hyperliquid-rsi-bot" / ".env"
    if sibling.exists():
        load_dotenv(sibling, override=False)
    wallet = os.environ.get("HYPE_WALLET_ADDRESS", "").strip()
    key = os.environ.get("HYPE_PRIVATE_KEY", "").strip()
    if not wallet or not key:
        raise RuntimeError("Set HYPE_WALLET_ADDRESS and HYPE_PRIVATE_KEY in .env")
    if not wallet.startswith("0x") or len(wallet) != 42:
        raise RuntimeError("HYPE_WALLET_ADDRESS must be 42-char hex")
    if not key.startswith("0x"):
        key = "0x" + key
    return wallet, key


def effective_leverage(requested: int, max_lev: int | None) -> int:
    if max_lev is None or max_lev < 1:
        return max(1, requested)
    return max(1, min(requested, int(max_lev)))


def init_pairs(
    client: HyperliquidClient,
    coin_leverage: list[tuple[str, int]],
    logger,
    *,
    arm_leverage: bool = True,
) -> list[PairSetup]:
    """Build watchlist from [(coin_input, requested_leverage), ...]."""
    setups: list[PairSetup] = []
    for raw, requested in coin_leverage:
        symbol, dex = parse_coin_input(raw)
        client.configure_coin(f"{dex}:{symbol}" if dex else symbol, perp_dex=dex)
        cross = cfg.USE_CROSS_MARGIN
        if client.only_isolated and cfg.USE_CROSS_MARGIN:
            cross = False
            logger.warning("%s requires isolated — using isolated", client.coin)
        lev = effective_leverage(int(requested), client.max_leverage)
        if lev < int(requested):
            logger.info(
                "%s leverage capped %sx → %sx (exchange max)",
                raw,
                requested,
                lev,
            )
        if arm_leverage:
            client.set_leverage(lev, is_cross=cross)
        tp = float(cfg.TAKE_PROFIT_PCT)
        setups.append(
            PairSetup(
                coin_input=raw,
                market=client.market,
                leverage=lev,
                use_cross_margin=cross,
                tp_pct=tp,
                sl_pct=tp,
            )
        )
        logger.info(
            "Watch %s | api=%s szDecimals=%s maxLev=%s using %sx%s",
            raw,
            client.coin,
            client.market.sz_decimals,
            client.max_leverage,
            lev,
            "" if arm_leverage else " (arm on quote)",
        )
    return setups


def build_leverage_map(watch: list[PairSetup]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in watch:
        out[e.api_coin] = e.leverage
        out[e.symbol] = e.leverage
        out[e.coin_input] = e.leverage
    return out


def seconds_until_next_candle(interval: str) -> float:
    step = INTERVAL_MS[interval] / 1000.0
    now = time.time()
    return max(1.0, step - (now % step) + 0.4)


def save_active_setups(memory: dict[str, LiveSetup]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {coin: setup_to_dict(setup) for coin, setup in memory.items()}
    ACTIVE_SETUP_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_active_setups() -> dict[str, LiveSetup]:
    if not ACTIVE_SETUP_PATH.exists():
        return {}
    try:
        raw = json.loads(ACTIVE_SETUP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    out: dict[str, LiveSetup] = {}
    if not isinstance(raw, dict):
        return {}
    # Legacy single-setup file: {"coin": "...", "sid": ...}
    if "sid" in raw:
        coin = str(raw.get("coin", "") or "")
        setup = _row_to_setup(coin, raw)
        if setup is not None and coin:
            out[coin] = setup
        return out
    for coin, row in raw.items():
        if not isinstance(row, dict):
            continue
        setup = _row_to_setup(str(coin), row)
        if setup is not None:
            out[str(coin)] = setup
    return out


def load_active_setup() -> LiveSetup | None:
    setups = load_active_setups()
    if not setups:
        return None
    return next(iter(setups.values()))


def clear_active_setup(coin: str | None = None) -> None:
    if coin is None:
        if ACTIVE_SETUP_PATH.exists():
            try:
                ACTIVE_SETUP_PATH.unlink()
            except OSError:
                pass
        return
    setups = load_active_setups()
    setups.pop(coin, None)
    if setups:
        save_active_setups(setups)
    elif ACTIVE_SETUP_PATH.exists():
        try:
            ACTIVE_SETUP_PATH.unlink()
        except OSError:
            pass


def resolve_setup_for_position(
    coin: str,
    store: SetupStore,
    memory: dict[str, LiveSetup],
) -> LiveSetup | None:
    if coin in memory:
        return memory[coin]
    disk = load_active_setups()
    if coin in disk:
        return disk[coin]
    return store.setup_for(coin)


def find_watch_entry(watch: list, coin: str):
    return next((e for e in watch if coin in e.position_coin_names()), None)


def live_margin_pcts(_setup: object | None = None) -> tuple[float, float, int, int]:
    """Live size: (total_budget_pct, per_fill_pct, split_count, legs).

    Per pair uses TOTAL_BALANCE_PCT / BALANCE_SPLIT_POSITIONS of *equity*.
    That pair slice is split equally across entry + DCA_MAX_ADDS extra fills.
    """
    total = min(95.0, max(1.0, float(getattr(cfg, "TOTAL_BALANCE_PCT", 95.0) or 95.0)))
    n = max(1, int(getattr(cfg, "BALANCE_SPLIT_POSITIONS", 3) or 3))
    pair_pct = total / float(n)
    extra = int(getattr(cfg, "DCA_MAX_ADDS", 1) or 0)
    legs = live_dca_leg_count(
        allow_dca=bool(getattr(cfg, "ALLOW_DCA", False)),
        extra_adds=extra,
    )
    return total, pair_pct / float(legs), n, legs


def live_dca_policy(setup: object | None) -> object | None:
    """Live DCA overlay: config leg count, trigger from tune or 1.2% default."""
    extra = int(getattr(cfg, "DCA_MAX_ADDS", 1) or 0)
    if not bool(getattr(cfg, "ALLOW_DCA", False)) or extra <= 0:
        return None
    trigger = 1.2
    if setup is not None:
        tuned = float(getattr(setup, "dca_trigger_pct", 0) or 0)
        if tuned > 0:
            trigger = tuned
    return type("DcaPolicy", (), {
        "dca_enabled": True,
        "dca_max_adds": extra,
        "dca_trigger_pct": trigger,
    })()


def pos_side_int(side: str) -> int:
    return 1 if side == "long" else -1


def main() -> None:
    hft_on = bool(getattr(cfg, "hft_pingpong_enabled", lambda: False)())
    ema_dev_on = bool(cfg.ema_dev_strategy_enabled())
    if hft_on:
        ema_dev_on = False
    if not ema_dev_on and not hft_on:
        if not cfg.USE_TP_SL and not cfg.USE_EXIT_SIGNAL and not cfg.USE_MAX_HOLD:
            raise ValueError("Enable at least one of USE_TP_SL, USE_EXIT_SIGNAL, USE_MAX_HOLD")
    elif ema_dev_on:
        ema_iv = str(getattr(cfg, "EMA_DEV_INTERVAL", "1m") or "1m")
        if ema_iv not in INTERVAL_MS:
            raise ValueError(
                f"Unknown EMA_DEV_INTERVAL {ema_iv!r}; known={list(INTERVAL_MS)}"
            )
        if int(getattr(cfg, "EMA_DEV_PERIOD", 0) or 0) < 2:
            raise ValueError("EMA_DEV_PERIOD must be >= 2")
        entry_pct = float(getattr(cfg, "EMA_DEV_ENTRY_PCT", 0) or 0)
        total_pct = float(getattr(cfg, "EMA_DEV_TOTAL_PCT", 0) or 0)
        if entry_pct <= 0 or entry_pct > 99:
            raise ValueError("EMA_DEV_ENTRY_PCT must be in (0, 99]")
        if total_pct <= 0 or total_pct > 99:
            raise ValueError("EMA_DEV_TOTAL_PCT must be in (0, 99]")
    pair_mode = str(getattr(cfg, "PAIR_SELECTION_MODE", "manual") or "manual").strip().lower()
    if pair_mode != "manual" and not is_auto_pair_mode(pair_mode):
        raise ValueError(
            "PAIR_SELECTION_MODE must be 'manual', 'top_volume', or 'top_movers' "
            f"(got {pair_mode!r})"
        )
    if pair_mode == "manual":
        pairs = tuple(cfg.PAIRS) if not isinstance(cfg.PAIRS, str) else (cfg.PAIRS,)
        if not pairs:
            raise ValueError("PAIRS must list at least one pair")
    else:
        if is_volume_mode(pair_mode) and int(getattr(cfg, "TOP_VOLUME_COUNT", 0) or 0) < 1:
            raise ValueError("TOP_VOLUME_COUNT must be >= 1 for top_volume mode")
        mover_n = int(
            getattr(cfg, "TOP_MOVER_COUNT", 0)
            or getattr(cfg, "TOP_VOLUME_COUNT", 0)
            or 0
        )
        if is_mover_mode(pair_mode) and mover_n < 1:
            raise ValueError("TOP_MOVER_COUNT must be >= 1 for top_movers mode")
        if int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0) < 0:
            raise ValueError("MIN_MAX_LEVERAGE must be >= 0")
        if int(getattr(cfg, "MAX_MAX_LEVERAGE", 0) or 0) < 0:
            raise ValueError("MAX_MAX_LEVERAGE must be >= 0")
        if float(getattr(cfg, "MIN_DAY_NOTIONAL_USD", 0) or 0) < 0:
            raise ValueError("MIN_DAY_NOTIONAL_USD must be >= 0")
        pairs = ()
    vol_n = int(getattr(cfg, "TOP_VOLUME_COUNT", 50) or 50)
    mov_n = int(
        getattr(cfg, "TOP_MOVER_COUNT", 0) or getattr(cfg, "TOP_VOLUME_COUNT", 14) or 14
    )
    universe_max_lev = int(getattr(cfg, "MAX_MAX_LEVERAGE", 0) or 0)
    if hft_on:
        scan_n = max(1, int(getattr(cfg, "HFT_SCAN_COUNT", 16) or 16))
        vol_n = max(vol_n, scan_n)
        mov_n = max(mov_n, scan_n)
        universe_max_lev = int(getattr(cfg, "HFT_MAX_MAX_LEVERAGE", 20) or 20)
    bad_iv = [i for i in cfg.INTERVALS if i not in INTERVAL_MS]
    if bad_iv:
        raise ValueError(f"Unknown INTERVALS {bad_iv}; known={list(INTERVAL_MS)}")
    if not cfg.INTERVALS:
        raise ValueError("INTERVALS must list at least one candle interval")
    if cfg.LEVERAGE < 1:
        raise ValueError("LEVERAGE must be >= 1")
    for k, v in (getattr(cfg, "PAIR_LEVERAGE", None) or {}).items():
        if int(v) < 1:
            raise ValueError(f"PAIR_LEVERAGE[{k!r}] must be >= 1")
    if cfg.TARGET_TRADES_PER_DAY <= 0:
        raise ValueError("TARGET_TRADES_PER_DAY must be > 0")
    # Reverse only when REVERSE_STRATEGY is on — never implied by top_movers.
    flip_live = bool(cfg.reverse_orders_enabled())

    logger = setup_logger("hl-multi-bot", LOG_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    housekeep_data_dir(DATA_DIR, logger=logger)

    def apply_hft_watch(entries):
        """HFT-only: keep markets whose exchange maxLev is ≤ HFT_MAX_MAX_LEVERAGE."""
        if not hft_on:
            return list(entries)
        cap = max(1, int(getattr(cfg, "HFT_MAX_MAX_LEVERAGE", 20) or 20))
        kept, dropped = filter_hft_entries(entries, max_max_leverage=cap)
        if dropped:
            extra = "" if len(dropped) <= 12 else f" +{len(dropped) - 12} more"
            logger.info(
                "HFT maxLev≤%sx: dropped %s (%s%s)",
                cap,
                len(dropped),
                ", ".join(dropped[:12]),
                extra,
            )
        if not kept:
            logger.warning("HFT maxLev≤%sx left 0 pairs in the scan", cap)
        else:
            logger.info(
                "HFT watch %s names with maxLev≤%sx: %s",
                len(kept),
                cap,
                ", ".join(
                    f"{e.api_coin}({int(e.market.max_leverage)}x)" for e in kept[:12]
                ),
            )
        return kept

    trade_store = TradeStateStore(DATA_DIR / "trade_state.json")
    ema_store = EmaDevStore(DATA_DIR / "ema_dev_state.json")
    hft_store = HftStore(DATA_DIR / "hft_pingpong_state.json")
    store = SetupStore(DATA_DIR, logger, refresh_hours=cfg.BACKTEST_REFRESH_HOURS)

    wallet, private_key = load_secrets()
    data_use_testnet = cfg.USE_TESTNET and not cfg.PAPER_TRADING
    base_url = (
        constants.TESTNET_API_URL if data_use_testnet else constants.MAINNET_API_URL
    )
    ip_reserve = int(getattr(cfg, "IP_WEIGHT_RESERVE", 50) or 0)
    os.environ["HL_IP_WEIGHT_RESERVE"] = str(ip_reserve)
    request_guard = RequestGuard(
        logger=logger,
        shared_budget=default_shared_budget(reserve=ip_reserve),
        ip_reserve=ip_reserve,
    )
    bootstrap = ThrottledInfo(
        base_url,
        skip_ws=True,
        guard=request_guard,
    )

    universe = resolve_pair_universe(
        bootstrap,
        mode=pair_mode,
        manual_pairs=tuple(cfg.PAIRS) if not isinstance(cfg.PAIRS, str) else (cfg.PAIRS,),
        top_volume_count=vol_n,
        top_mover_count=mov_n,
        include_xyz=bool(getattr(cfg, "INCLUDE_XYZ_PAIRS", False)),
        use_max_leverage=bool(getattr(cfg, "USE_MAX_LEVERAGE", True)),
        default_leverage=int(cfg.LEVERAGE),
        leverage_overrides=getattr(cfg, "PAIR_LEVERAGE", None),
        requested_leverage_for=cfg.requested_leverage_for,
        min_max_leverage=int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0),
        max_max_leverage=universe_max_lev,
        min_day_notional=float(getattr(cfg, "MIN_DAY_NOTIONAL_USD", 0) or 0),
        xyz_mode=cfg.xyz_pair_mode() if hasattr(cfg, "xyz_pair_mode") else None,
        logger=logger,
    )
    pair_inputs = [c for c, _ in universe.pairs]
    markets = resolve_watchlist(bootstrap, pair_inputs, None)
    sdk_dexes = sdk_perp_dexs_for_dexes({m.perp_dex for m in markets})

    sym, dex = parse_coin_input(pair_inputs[0])
    client_kwargs = dict(
        wallet_address=wallet,
        private_key=private_key,
        coin=f"{dex}:{sym}" if dex else sym,
        logger=logger,
        use_testnet=data_use_testnet,
        perp_dex=dex,
        market=markets[0],
        sdk_perp_dexs=sdk_dexes,
        request_guard=request_guard,
        ip_weight_reserve=ip_reserve,
    )
    if cfg.PAPER_TRADING:
        client = PaperHyperliquidClient(
            **client_kwargs,
            paper_data_dir=DATA_DIR,
            paper_start_balance=cfg.PAPER_START_BALANCE,
            paper_taker_fee_pct=cfg.TAKER_FEE_PCT,
            account_filename="paper_account.json",
            trades_filename="paper_trades.jsonl",
        )
    else:
        client = HyperliquidClient(**client_kwargs)

    watch = init_pairs(client, universe.pairs, logger, arm_leverage=not hft_on)
    watch = apply_hft_watch(watch)
    if hft_on and not watch:
        raise ValueError(
            "HFT watch is empty after maxLev filter. Need pairs with exchange "
            "max leverage ≤ HFT_MAX_MAX_LEVERAGE (and min ≥ MIN_MAX_LEVERAGE)."
        )
    cleanup_coins = [e.api_coin for e in watch]
    # Default tune leverage; per-coin overrides come from each PairSetup.
    tune_leverage = watch[0].leverage if watch else cfg.LEVERAGE
    leverage_by_coin = build_leverage_map(watch)
    universe_built_at = time.time()
    mover_buckets: dict[str, str] = dict(universe.buckets)

    executor = OrderExecutor(
        client,
        wait_seconds=30,
        max_attempts=5,
        logger=logger,
        use_market_orders=cfg.USE_MARKET_ORDERS,
        market_slippage=cfg.MARKET_ORDER_SLIPPAGE,
        mid_limit_then_market=ema_dev_on
        and bool(getattr(cfg, "EMA_DEV_LIMIT_ORDERS", False)),
        mid_limit_wait_seconds=float(
            getattr(cfg, "EMA_DEV_LIMIT_WAIT_SECONDS", 10.0) or 10.0
        ),
        mid_limit_attempts=int(getattr(cfg, "EMA_DEV_LIMIT_ATTEMPTS", 3) or 3),
    )

    def refresh_universe_if_needed(*, force: bool = False) -> None:
        """Re-rank volume/mover leaders before each tune in auto pair modes."""
        nonlocal watch, cleanup_coins, tune_leverage, leverage_by_coin
        nonlocal universe_built_at, mover_buckets
        if not is_auto_pair_mode(pair_mode):
            return
        # Startup already ranked — skip duplicate refresh on first tune.
        if not force and time.time() - universe_built_at < 90.0:
            return
        if is_mover_mode(pair_mode):
            logger.info(
                "Refreshing top-mover universe (n=%s xyz_mode=%s min_maxLev≥%s max_maxLev≤%s minVol≥$%s)",
                mov_n,
                cfg.xyz_pair_mode() if hasattr(cfg, "xyz_pair_mode") else getattr(cfg, "INCLUDE_XYZ_PAIRS", False),
                int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0),
                universe_max_lev or "off",
                int(getattr(cfg, "MIN_DAY_NOTIONAL_USD", 0) or 0) or "off",
            )
        else:
            logger.info(
                "Refreshing top-volume universe (n=%s xyz_mode=%s min_maxLev≥%s max_maxLev≤%s minVol≥$%s)",
                vol_n,
                cfg.xyz_pair_mode() if hasattr(cfg, "xyz_pair_mode") else getattr(cfg, "INCLUDE_XYZ_PAIRS", False),
                int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0),
                universe_max_lev or "off",
                int(getattr(cfg, "MIN_DAY_NOTIONAL_USD", 0) or 0) or "off",
            )
        fresh = resolve_pair_universe(
            client.info,
            mode=pair_mode,
            manual_pairs=tuple(cfg.PAIRS)
            if not isinstance(cfg.PAIRS, str)
            else (cfg.PAIRS,),
            top_volume_count=vol_n,
            top_mover_count=mov_n,
            include_xyz=bool(getattr(cfg, "INCLUDE_XYZ_PAIRS", False)),
            use_max_leverage=bool(getattr(cfg, "USE_MAX_LEVERAGE", True)),
            default_leverage=int(cfg.LEVERAGE),
            leverage_overrides=getattr(cfg, "PAIR_LEVERAGE", None),
            requested_leverage_for=cfg.requested_leverage_for,
            min_max_leverage=int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0),
            max_max_leverage=universe_max_lev,
            min_day_notional=float(getattr(cfg, "MIN_DAY_NOTIONAL_USD", 0) or 0),
            xyz_mode=cfg.xyz_pair_mode() if hasattr(cfg, "xyz_pair_mode") else None,
            logger=logger,
        )
        watch = init_pairs(client, fresh.pairs, logger, arm_leverage=not hft_on)
        filtered = apply_hft_watch(watch)
        if hft_on and not filtered:
            logger.warning(
                "HFT refresh produced 0 names with maxLev≤%sx — keeping previous watch",
                int(getattr(cfg, "HFT_MAX_MAX_LEVERAGE", 20) or 20),
            )
            return
        watch = filtered
        cleanup_coins = [e.api_coin for e in watch]
        tune_leverage = watch[0].leverage if watch else cfg.LEVERAGE
        leverage_by_coin = build_leverage_map(watch)
        mover_buckets = dict(fresh.buckets)
        universe_built_at = time.time()

    def run_tune() -> None:
        nonlocal watch, cleanup_coins, tune_leverage, leverage_by_coin
        refresh_universe_if_needed()
        reverse_live = bool(cfg.reverse_orders_enabled())
        side_map = (
            allowed_sides_for_movers(mover_buckets)
            if is_mover_mode(pair_mode)
            else None
        )
        if side_map:
            logger.info(
                "Movers: tune WITH 24h move (gainers LONG, losers SHORT). "
                "Live reverse=%s → orders %s",
                reverse_live,
                "flipped" if reverse_live else "match backtest",
            )
        results = run_full_tune(
            client.info,
            [e.api_coin for e in watch],
            list(cfg.INTERVALS),
            leverage=tune_leverage,
            leverage_by_coin=leverage_by_coin,
            data_dir=DATA_DIR,
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
            allowed_side_by_coin=side_map,
            logger=logger,
        )
        if results:
            max_live = int(getattr(cfg, "MAX_LIVE_PAIRS", 5) or 5)
            results = select_top_live_pairs(
                results,
                max_live,
                buckets=mover_buckets if is_mover_mode(pair_mode) else None,
                logger=logger,
            )
            store.save_results(
                results,
                leverage=tune_leverage,
                pair_selection_mode=pair_mode,
                reverse_orders=reverse_live,
                mover_tune=MOVER_TUNE_LOCK if is_mover_mode(pair_mode) else None,
            )
            # Auto pair modes: live-scan only kept winners (full set rebuilt on retune).
            if is_auto_pair_mode(pair_mode):
                winners = set(results.keys())
                kept = [e for e in watch if e.api_coin in winners]
                if kept:
                    watch = kept
                    cleanup_coins = [e.api_coin for e in watch]
                    tune_leverage = watch[0].leverage
                    leverage_by_coin = build_leverage_map(watch)
                    logger.info(
                        "Live watch trimmed to top %s: %s",
                        len(watch),
                        ",".join(e.api_coin for e in watch),
                    )
            housekeep_data_dir(DATA_DIR, logger=logger)
            logger.info("Active: %s", store.describe())
        else:
            store.mark_attempt()
            logger.warning(
                "Tune produced nothing — retry after cooldown (%.0fs)",
                store._retry_cooldown_s,
            )

    def maybe_tune(*, force: bool = False) -> None:
        # Empty setups: never hammer retune — honor cooldown after a failed attempt.
        mismatch = store.config_mismatch(
            pair_mode,
            cfg.reverse_orders_enabled(),
            mover_tune=MOVER_TUNE_LOCK if is_mover_mode(pair_mode) else None,
        )
        if mismatch:
            cooled = (
                store._last_attempt_ts <= 0
                or time.time() - store._last_attempt_ts >= store._retry_cooldown_s
            )
            if cooled:
                logger.warning("Saved setups stale (%s) — retuning", mismatch)
                run_tune()
            else:
                left = store._retry_cooldown_s - (
                    time.time() - store._last_attempt_ts
                )
                logger.info(
                    "Saved setups stale (%s) — next tune in ~%.0fs",
                    mismatch,
                    max(0.0, left),
                )
            return
        if not store.setups():
            cooled = (
                store._last_attempt_ts <= 0
                or time.time() - store._last_attempt_ts >= store._retry_cooldown_s
            )
            if cooled:
                run_tune()
            else:
                left = store._retry_cooldown_s - (
                    time.time() - store._last_attempt_ts
                )
                logger.info(
                    "No setups yet — next tune in ~%.0fs (last attempt found nothing)",
                    max(0.0, left),
                )
            return
        if force or store.refresh_due():
            run_tune()

    mode = "PAPER" if cfg.PAPER_TRADING else "LIVE"
    strat_mode = getattr(cfg, "STRATEGY_MODE", "mtf")
    logger.info(
        "Starting MULTI [%s] mode=%s | pair_mode=%s | pairs=%s | intervals=%s | exec=%s | "
        "lev_default=%sx | refresh=%.0fh | target≈%.1f/d | max_live=%s | TP/SL=%s signal=%s "
        "hold=%s | dca=%s extra=%s | reverse=%s",
        mode,
        strat_mode,
        pair_mode,
        ",".join(e.api_coin for e in watch),
        ",".join(cfg.INTERVALS),
        getattr(cfg, "MTF_EXEC_INTERVAL", "1m"),
        cfg.LEVERAGE,
        cfg.BACKTEST_REFRESH_HOURS,
        cfg.TARGET_TRADES_PER_DAY,
        int(getattr(cfg, "MAX_LIVE_PAIRS", 5) or 5),
        cfg.USE_TP_SL,
        cfg.USE_EXIT_SIGNAL,
        cfg.USE_MAX_HOLD,
        cfg.ALLOW_DCA,
        int(getattr(cfg, "DCA_MAX_ADDS", 1) or 0),
        flip_live,
    )
    concurrent = bool(getattr(cfg, "ALLOW_CONCURRENT_POSITIONS", True))
    max_concurrent = int(getattr(cfg, "MAX_CONCURRENT_POSITIONS", 0) or 0)
    if ema_dev_on or hft_on:
        concurrent = False
        max_concurrent = 1
    n_iv = len(cfg.INTERVALS)
    n_pairs = max(1, int(getattr(cfg, "MAX_LIVE_PAIRS", 5) or 5))
    budget = snapshot_weight_budget(n_pairs, n_iv, reserve=ip_reserve)
    logger.info(
        "Concurrent=%s max_open=%s | candle cache per TF close | IP reserve=%s "
        "hour-spike ~%s weight for %s pairs×%s TFs → hour-safe max ≈ %s pairs "
        "(typical minutes ≈ %s pairs)",
        concurrent,
        max_concurrent or "margin",
        ip_reserve,
        budget["hour_weight"],
        n_pairs,
        n_iv,
        budget["hour_max_pairs"],
        budget["steady_max_pairs"],
    )
    live_total, live_per, live_splits, live_legs = live_margin_pcts()
    logger.info(
        "Sizing: %.0f%% of equity budget ÷ %s positions = %.2f%% per pair, "
        "%s equal fill(s) → %.2f%% of equity each (not leftover free margin)",
        live_total,
        live_splits,
        live_total / float(live_splits),
        live_legs,
        live_per,
    )
    if ema_dev_on:
        entry_pct, dca_pct = ema_dev_fill_pcts(
            float(cfg.EMA_DEV_ENTRY_PCT),
            float(cfg.EMA_DEV_TOTAL_PCT),
            dca_on=bool(getattr(cfg, "EMA_DEV_ALLOW_DCA", False)),
        )
        logger.info(
            "EMA-dev strategy ON (no tune) | %s EMA(%s) | entry=%.1f%% of equity now "
            "dca=%s add=%.1f%% of equity at DCA (fits remaining free if smaller) | "
            "one pair | refresh pair list after every close | reverse=%s | "
            "rank_cross_age=%s | limit_orders=%s "
            "(%sx mid post-only entry, parked TP limit, %.0fs wait, then market leftover) | "
            "max_hold=%.1fh from original entry",
            str(getattr(cfg, "EMA_DEV_INTERVAL", "1m") or "1m"),
            int(getattr(cfg, "EMA_DEV_PERIOD", 100) or 100),
            entry_pct,
            "on" if dca_pct > 0 else "off",
            dca_pct,
            "on (momentum: long above / short below, fixed TP/SL = ±D%)"
            if flip_live
            else "off (mean-revert, fixed TP/SL = ±D%)",
            "on (%s crosses better)"
            % ("newer" if flip_live else "older")
            if bool(getattr(cfg, "EMA_DEV_RANK_CROSS_AGE", False))
            else "off",
            "on" if bool(getattr(cfg, "EMA_DEV_LIMIT_ORDERS", False)) else "off",
            int(getattr(cfg, "EMA_DEV_LIMIT_ATTEMPTS", 3) or 3),
            float(getattr(cfg, "EMA_DEV_LIMIT_WAIT_SECONDS", 10.0) or 10.0),
            float(getattr(cfg, "EMA_DEV_MAX_POSITION_HOURS", 0) or 0),
        )
        if flip_live:
            logger.info(
                "REVERSE_STRATEGY on for EMA-dev — momentum, fixed TP/SL ±D% from fill"
            )
    if hft_on:
        logger.info(
            "HFT ping-pong ON | poll=%.0fs clip≤$%.0f notional lev≤%s lookback=%s "
            "er≤%.2f spread=%.1f–%.1fbps timeout=%.0fs box_break=%.1fbps | "
            "scan minLev≥%s maxLev≤%s | HFT maxLev≤%sx | EMA-dev and MTF live paths are off",
            float(getattr(cfg, "HFT_POLL_SECONDS", 3) or 3),
            float(getattr(cfg, "HFT_CLIP_MAX_NOTIONAL_USD", 15) or 15),
            int(getattr(cfg, "HFT_MAX_LEVERAGE", 20) or 20),
            int(getattr(cfg, "HFT_LOOKBACK_BARS", 45) or 45),
            float(getattr(cfg, "HFT_MAX_ER", 0.32) or 0.32),
            float(getattr(cfg, "HFT_MIN_SPREAD_BPS", 2.8) or 2.8),
            float(getattr(cfg, "HFT_MAX_SPREAD_BPS", 16) or 16),
            float(getattr(cfg, "HFT_INVENTORY_TIMEOUT_S", 40) or 40),
            float(getattr(cfg, "HFT_BOX_BREAK_BPS", 8) or 8),
            int(getattr(cfg, "MIN_MAX_LEVERAGE", 1) or 1),
            universe_max_lev or "off",
            int(getattr(cfg, "HFT_MAX_MAX_LEVERAGE", 20) or 20),
        )
    if is_volume_mode(pair_mode):
        logger.info(
            "Top-volume settings: count=%s xyz_mode=%s use_max_lev=%s min_maxLev≥%s max_maxLev≤%s minVol≥$%s",
            int(getattr(cfg, "TOP_VOLUME_COUNT", 50) or 50),
            cfg.xyz_pair_mode() if hasattr(cfg, "xyz_pair_mode") else getattr(cfg, "INCLUDE_XYZ_PAIRS", False),
            bool(getattr(cfg, "USE_MAX_LEVERAGE", True)),
            int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0),
            int(getattr(cfg, "MAX_MAX_LEVERAGE", 0) or 0) or "off",
            int(getattr(cfg, "MIN_DAY_NOTIONAL_USD", 0) or 0) or "off",
        )
    if is_mover_mode(pair_mode):
        gainer_side = mover_tune_side("gainer")
        loser_side = mover_tune_side("loser")
        gainers = [c for c, b in mover_buckets.items() if b == "gainer"]
        losers = [c for c, b in mover_buckets.items() if b == "loser"]
        logger.info(
            "Top-mover settings: count=%s (%s gainers + %s losers) xyz_mode=%s "
            "use_max_lev=%s min_maxLev≥%s max_maxLev≤%s minVol≥$%s",
            int(
                getattr(cfg, "TOP_MOVER_COUNT", 0)
                or getattr(cfg, "TOP_VOLUME_COUNT", 14)
                or 14
            ),
            len(gainers),
            len(losers),
            cfg.xyz_pair_mode() if hasattr(cfg, "xyz_pair_mode") else getattr(cfg, "INCLUDE_XYZ_PAIRS", False),
            bool(getattr(cfg, "USE_MAX_LEVERAGE", True)),
            int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0),
            int(getattr(cfg, "MAX_MAX_LEVERAGE", 0) or 0) or "off",
            int(getattr(cfg, "MIN_DAY_NOTIONAL_USD", 0) or 0) or "off",
        )
        logger.info(
            "Movers: tune with-trend gainers=%s losers=%s | live reverse=%s "
            "so orders %s",
            "LONG" if gainer_side > 0 else "SHORT",
            "LONG" if loser_side > 0 else "SHORT",
            flip_live,
            "flipped (SHORT gainers / LONG losers)"
            if flip_live
            else "same as tune (LONG gainers / SHORT losers)",
        )
        if gainers:
            logger.info("24h gainers: %s", ",".join(gainers))
        if losers:
            logger.info("24h losers: %s", ",".join(losers))
    if getattr(cfg, "PAIR_LEVERAGE", None):
        logger.info(
            "Per-pair leverage: %s",
            ", ".join(f"{e.api_coin}={e.leverage}x" for e in watch),
        )
    if flip_live:
        logger.warning(
            "REVERSE_STRATEGY on — same signal bars as backtest, opposite order "
            "(LONG→SHORT / SHORT→LONG); DCA/TP/SL follow the real position"
        )

    stop = threading.Event()

    def _stop(*_a: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    pos_ok, open_positions = client.fetch_open_positions(force=True)
    mismatch0 = store.config_mismatch(
        pair_mode,
        cfg.reverse_orders_enabled(),
        mover_tune=MOVER_TUNE_LOCK if is_mover_mode(pair_mode) else None,
    )
    if not ema_dev_on and not hft_on and mismatch0 and pos_ok and open_positions:
        logger.warning(
            "Saved setups stale (%s) but positions are open — will retune once flat",
            mismatch0,
        )
    if pos_ok and not open_positions:
        if hft_on:
            logger.info("Flat — HFT ping-pong will pick a choppy book and quote (no tune)")
        elif ema_dev_on:
            logger.info("Flat — EMA-dev will scan the pair list (no tune)")
        else:
            logger.info("Flat — auto-tune if no params or refresh due")
            maybe_tune(force=not store.setups())
    elif pos_ok and open_positions:
        logger.info(
            "Open position(s) at start (%s) — manage + keep scanning if concurrent",
            len(open_positions),
        )

    last_entry_key: dict[str, tuple] = {}
    last_manage_bar: dict[str, int] = {}
    open_setup_mem: dict[str, LiveSetup] = {}
    protected_coins: set[str] = set()
    was_in_position = bool(open_positions) if pos_ok else False
    first_iter = True

    def occupied_api(positions: list) -> set[str]:
        out: set[str] = set()
        for coin, _pos in positions:
            entry = find_watch_entry(watch, coin)
            out.add(entry.api_coin if entry else str(coin))
        return out

    def wake_seconds() -> float:
        if hft_on:
            return max(2.0, float(getattr(cfg, "HFT_POLL_SECONDS", 3.0) or 3.0))
        if ema_dev_on:
            wait = seconds_until_next_candle(
                str(getattr(cfg, "EMA_DEV_INTERVAL", "1m") or "1m")
            )
            if trade_store.trades:
                wait = min(wait, max(1.0, float(cfg.POSITION_POLL_SECONDS)))
            hold_h = float(getattr(cfg, "EMA_DEV_MAX_POSITION_HOURS", 0) or 0)
            if hold_h > 0 and trade_store.trades:
                now = time.time()
                upcoming = [
                    hold_h * 3600.0 - (now - t.opened_at)
                    for t in trade_store.trades.values()
                ]
                upcoming = [left for left in upcoming if left > 0]
                if upcoming:
                    wait = min(wait, max(1.0, min(upcoming)))
            return wait
        intervals = {
            s.interval for c in store.watch_coins() for s in store.setups_for(c)
        }
        for c in store.watch_coins():
            for s in store.setups_for(c):
                if s.is_mtf and s.mtf_intervals:
                    intervals.update(s.mtf_intervals)
        wake_iv = "1m"
        if intervals:
            wake_iv = min(intervals, key=lambda x: INTERVAL_MS.get(x, 10**12))
        wait = seconds_until_next_candle(wake_iv)
        if cfg.USE_MAX_HOLD and trade_store.trades:
            now = time.time()
            upcoming = [
                cfg.MAX_POSITION_HOURS * 3600.0 - (now - t.opened_at)
                for t in trade_store.trades.values()
            ]
            upcoming = [left for left in upcoming if left > 0]
            if upcoming:
                wait = min(wait, max(1.0, min(upcoming)))
        return wait

    def drop_local(api_coin: str) -> None:
        open_setup_mem.pop(api_coin, None)
        protected_coins.discard(api_coin)
        last_manage_bar.pop(api_coin, None)
        clear_active_setup(api_coin)
        if open_setup_mem:
            save_active_setups(open_setup_mem)

    def finish_close(entry: PairSetup) -> None:
        wait_until_flat(
            client,
            trade_store,
            logger,
            coin=entry.api_coin,
            coin_names=entry.position_coin_names(),
        )
        drop_local(entry.api_coin)

    def manage_one(coin: str, position) -> bool:
        """Manage one open coin. Return True if it was closed this pass."""
        entry = find_watch_entry(watch, coin)
        if entry is None:
            logger.warning(
                "Position on %s outside watch — managing max-hold only",
                coin,
            )
            client.configure_coin(str(coin))
            side = str(getattr(position, "side", "") or "")
            size = float(getattr(position, "size", 0) or 0)
            entry_px = float(getattr(position, "entry_price", 0) or 0)
            tracked = trade_store.get(client.coin) or trade_store.get(str(coin))
            if tracked is None or tracked.side != side:
                trade_store.soft_adopt(
                    client.coin, side, entry_px, size, 0.01, 0.01
                )
            t = trade_store.get(client.coin)
            hold_s = (
                max(0.0, time.time() - t.opened_at)
                if t and cfg.USE_MAX_HOLD
                else 0.0
            )
            if hold_s >= cfg.MAX_POSITION_HOURS * 3600.0:
                logger.warning(
                    "Max hold %.1fh — closing %s",
                    cfg.MAX_POSITION_HOURS,
                    client.coin,
                )
                if not executor.execute_rsi_exit():
                    executor.emergency_flatten("max_hold")
                wait_until_flat(
                    client,
                    trade_store,
                    logger,
                    coin=client.coin,
                    coin_names=frozenset(
                        {client.coin, str(coin), client.market.symbol}
                    ),
                )
                drop_local(client.coin)
                return True
            return False
        activate_pair(client, entry)
        setup = resolve_setup_for_position(entry.api_coin, store, open_setup_mem)
        if setup is None:
            logger.warning(
                "Open %s but no tuned setup — managing with max-hold / "
                "config TP/SL fallback only",
                entry.api_coin,
            )
        side = str(getattr(position, "side", "") or "")
        size = float(getattr(position, "size", 0) or 0)
        entry_px = float(getattr(position, "entry_price", 0) or 0)
        use_tpsl = bool(
            setup and setup.use_tpsl and cfg.USE_TP_SL and setup.tp_pct > 0
        )
        tp = float(setup.tp_pct) if use_tpsl else 0.01
        tracked = trade_store.get(entry.api_coin)
        need_adopt = (
            tracked is None
            or tracked.coin != entry.api_coin
            or tracked.side != side
        )
        if need_adopt:
            trade_store.soft_adopt(
                entry.api_coin,
                side,
                entry_px,
                size,
                tp,
                tp,
            )
            logger.info(
                "Tracking %s %s size=%s @ %s",
                entry.api_coin,
                side,
                size,
                entry_px,
            )
        else:
            t = tracked
            if entry_px > 0 and (
                abs(t.size - size) > 1e-12 or abs(t.entry_price - entry_px) > 1e-12
            ):
                tp_px, sl_px = TradeStateStore._tp_sl_prices(side, entry_px, tp, tp)
                t.size = size
                t.entry_price = entry_px
                t.take_profit_pct = tp
                t.stop_loss_pct = tp
                t.take_profit_price = tp_px
                t.stop_loss_price = sl_px
                trade_store._save()

        if use_tpsl and entry.api_coin not in protected_coins:
            ok = ensure_protected_position(
                client,
                executor,
                position,
                trade_store,
                take_profit_pct=setup.tp_pct,
                stop_loss_pct=setup.sl_pct,
                logger=logger,
            )
            if ok:
                protected_coins.add(entry.api_coin)
            else:
                drop_local(entry.api_coin)
                return True
        elif need_adopt and not use_tpsl:
            client.cancel_all_orders_for_coin()

        manage_iv = setup.interval if setup else "1m"
        need_bars = max(80, int(getattr(setup, "aux", 0) or 0) + 50)
        candles = client.get_closed_candles_for(
            entry.api_coin, manage_iv, min_bars=need_bars
        )
        bar_t = int(candles[-1]["t"]) if len(candles) >= 40 else -1
        new_bar = bar_t > 0 and last_manage_bar.get(entry.api_coin) != bar_t
        if new_bar:
            last_manage_bar[entry.api_coin] = bar_t
            close_px = float(candles[-1]["c"])
            t = trade_store.get(entry.api_coin)
            dca_pol = live_dca_policy(setup)
            if dca_pol is not None and use_tpsl and t is not None:
                if dca_should_add(
                    dca_pol,
                    avg_entry_px=float(t.entry_price),
                    mark_or_close=close_px,
                    position_side=t.side,
                    dca_adds_done=int(t.dca_adds),
                ):
                    add_sz = float(t.initial_size or t.size)
                    total_bal, _, _, legs = live_margin_pcts()
                    logger.info(
                        "DCA add %s on closed bar +size≈%s (equal leg %s/%s, "
                        "total bal=%.0f%%%s)",
                        entry.api_coin,
                        add_sz,
                        int(t.dca_adds) + 2,
                        legs,
                        total_bal,
                        " REVERSE" if flip_live else "",
                    )
                    new_pos = executor.execute_dca_add(
                        add_sz,
                        take_profit_pct=setup.tp_pct,
                        stop_loss_pct=setup.sl_pct,
                    )
                    if new_pos is not None:
                        trade_store.record_dca_add(
                            float(new_pos.entry_price or close_px),
                            float(new_pos.size),
                            coin=entry.api_coin,
                        )
                        protected_coins.add(entry.api_coin)

            if setup and setup.use_exit_signal and cfg.USE_EXIT_SIGNAL:
                avg_px = float(
                    trade_store.get(entry.api_coin).entry_price
                    if trade_store.get(entry.api_coin)
                    else entry_px
                )
                if exit_signal(
                    setup,
                    candles,
                    avg_entry_px=avg_px,
                    position_side=pos_side_int(side),
                ):
                    logger.warning(
                        "Exit signal %s — closing %s",
                        setup.exit_name,
                        entry.api_coin,
                    )
                    activate_pair_for_trade(client, entry)
                    if not executor.execute_rsi_exit():
                        executor.emergency_flatten("exit_signal")
                    finish_close(entry)
                    return True

        if (setup is None or setup.use_max_hold) and cfg.USE_MAX_HOLD:
            t = trade_store.get(entry.api_coin)
            hold_s = (
                max(0.0, time.time() - t.opened_at)
                if t and t.coin == entry.api_coin
                else 0.0
            )
            if hold_s >= cfg.MAX_POSITION_HOURS * 3600.0:
                logger.warning(
                    "Max hold %.1fh — closing %s",
                    cfg.MAX_POSITION_HOURS,
                    entry.api_coin,
                )
                activate_pair_for_trade(client, entry)
                if not executor.execute_rsi_exit():
                    executor.emergency_flatten("max_hold")
                finish_close(entry)
                return True
        return False

    def try_open(entry: PairSetup, setup: LiveSetup, bar_t: int, confirm, confirm_multi) -> bool:
        rev = flip_live
        confirm_sig = entry_signal(setup, confirm, multi_candles=confirm_multi)
        if confirm_sig == 0:
            logger.info("Skip %s — signal gone on re-check", entry.api_coin)
            return False
        signal_side = int(confirm_sig)
        side = -signal_side if rev else signal_side
        if rev:
            logger.info(
                "REVERSE — signal %s → order %s",
                "LONG" if signal_side > 0 else "SHORT",
                "LONG" if side > 0 else "SHORT",
            )
        total_bal, bal, slices, legs = live_margin_pcts()
        activate_pair_for_trade(client, entry)
        est = client.estimate_order_size(
            bal,
            entry.leverage,
            sz_decimals=entry.market.sz_decimals,
            min_notional_usd=cfg.MIN_ORDER_NOTIONAL_USD,
            margin_from="equity",
        )
        if not est.ok:
            logger.warning("Skip %s — %s", entry.api_coin, est.reason)
            return False
        try:
            equity = client.get_account_value(force=True)
        except Exception:
            equity = 0.0
        if equity > 0 and est.available_margin < equity * cfg.MIN_FREE_MARGIN_FRAC:
            logger.warning(
                "Skip %s — free margin $%.2f low vs equity $%.2f (MIN_FREE_MARGIN_FRAC)",
                entry.api_coin,
                est.available_margin,
                equity,
            )
            return False
        is_buy = side > 0
        use_tpsl = bool(setup.use_tpsl and cfg.USE_TP_SL and setup.tp_pct > 0)
        logger.info(
            "Entry %s %s via %s@%s size=%s notional=$%.2f "
            "margin=%.2f%% of equity (equal leg 1/%s, pair 1/%s of %.0f%%) "
            "tpsl=%s [%s%s]",
            entry.api_coin,
            "LONG" if is_buy else "SHORT",
            setup.name,
            setup.interval,
            est.size,
            est.notional_usd,
            bal,
            legs,
            slices,
            total_bal,
            f"{setup.tp_pct:.2f}%" if use_tpsl else "OFF",
            mode,
            "|REVERSE" if rev else "",
        )
        try:
            if use_tpsl:
                ok = executor.execute_protected_entry(
                    is_buy=is_buy,
                    target_sz=est.size,
                    take_profit_pct=setup.tp_pct,
                    stop_loss_pct=setup.sl_pct,
                )
            else:
                client.cancel_all_orders_for_coin()
                ok = executor.execute_open(is_buy=is_buy, target_sz=est.size)
        except RuntimeError as exc:
            if "insufficient margin" in str(exc).lower():
                logger.warning("Insufficient margin for %s", entry.api_coin)
                return False
            raise
        if not ok:
            cleanup_closed_coin(client, trade_store, entry.api_coin)
            return False
        pos = client.get_position(force=True)
        if pos is None:
            cleanup_closed_coin(client, trade_store, entry.api_coin)
            return False
        if use_tpsl and not client.has_exchange_tpsl():
            executor.emergency_flatten("unprotected")
            cleanup_closed_coin(client, trade_store, entry.api_coin)
            return False
        fill = pos.entry_price or client.get_mark_price()
        trade_store.open_trade(
            client.coin,
            pos.side,
            fill,
            pos.size,
            setup.tp_pct if use_tpsl else 0.01,
            setup.sl_pct if use_tpsl else 0.01,
            equity_at_entry=client.get_account_value(force=True),
        )
        open_setup_mem[entry.api_coin] = setup
        save_active_setups(open_setup_mem)
        last_manage_bar[entry.api_coin] = bar_t
        if use_tpsl:
            protected_coins.add(entry.api_coin)
        logger.info(
            "Opened %s %s size=%s @ %s [%s] open=%s",
            client.coin,
            pos.side,
            pos.size,
            fill,
            mode,
            len(trade_store.trades),
        )
        return True

    def _ema_iv() -> str:
        return str(getattr(cfg, "EMA_DEV_INTERVAL", "1m") or "1m")

    def _ema_period() -> int:
        return int(getattr(cfg, "EMA_DEV_PERIOD", 100) or 100)

    def _ema_bars_need() -> int:
        return max(40, _ema_period() + 30)

    def _ema_rank_age() -> bool:
        return bool(getattr(cfg, "EMA_DEV_RANK_CROSS_AGE", False))

    def _ema_scan_bars_need() -> int:
        need = _ema_bars_need()
        if _ema_rank_age():
            # ~8h of 1m bars so "hours above EMA" can rank vs a fresh cross.
            need = max(need, _ema_period() + 480)
        return need

    def _ema_dca_on() -> bool:
        return bool(getattr(cfg, "EMA_DEV_ALLOW_DCA", False))

    def _ema_max_hold_hours() -> float:
        return max(0.0, float(getattr(cfg, "EMA_DEV_MAX_POSITION_HOURS", 0) or 0))

    def ema_hold_started_at(trade: EmaDevTrade, entry: PairSetup) -> float:
        """Earliest known fill time so a restart does not reset the 3h clock."""
        cands: list[float] = []
        if float(trade.opened_at or 0) > 0:
            cands.append(float(trade.opened_at))
        tracked = trade_store.get(entry.api_coin)
        if (
            tracked is not None
            and tracked.side == trade.side
            and float(tracked.opened_at or 0) > 0
        ):
            cands.append(float(tracked.opened_at))
        bar = int(trade.opened_bar_t or 0)
        if bar > 10_000_000_000:
            cands.append(bar / 1000.0)
        elif bar > 1_000_000_000:
            cands.append(float(bar))
        if not cands:
            return time.time()
        return min(cands)

    def _ema_protect_kw() -> dict:
        return {"dca_on": _ema_dca_on(), "reverse": flip_live}

    def force_attach_tpsl(tp_pct: float, sl_pct: float) -> str:
        if executor.mid_limit_then_market:
            return client.protect_ema_maker(tp_pct, sl_pct, max_attempts=3)
        client.cancel_all_orders_for_coin()
        client.invalidate_user_state()
        pos = client.get_position(force=True)
        if pos is None:
            return "flat"
        ok = client.attach_position_tpsl(
            pos, tp_pct, sl_pct, max_attempts=3
        )
        return "ok" if ok else "failed"

    def ema_is_protected() -> bool:
        if executor.mid_limit_then_market:
            return bool(client.has_ema_maker_protect())
        return bool(client.has_exchange_tpsl())

    def should_chase_maker_tp(side: str, mark: float, target_px: float) -> bool:
        if target_px <= 0 or not ema_dev_should_tp(side, mark, target_px):
            return False
        if not executor.mid_limit_then_market:
            return True
        through = ema_dev_tp_through_pct(side, mark, target_px)
        resting = client.resting_tp_px()
        resting_ok = (
            resting is not None
            and abs(float(resting) - target_px) / target_px * 100.0 < 0.2
        )
        return (not resting_ok) or through >= 0.25

    def ema_tp_target(trade: EmaDevTrade, avg: float, ema: float) -> float:
        ref = float(trade.entry_px or avg or 0)
        return ema_dev_tp_price_from_dev(trade.side, ref, trade.dev_pct)

    def flatten_ema(entry: PairSetup, reason: str) -> bool:
        activate_pair_for_trade(client, entry)
        logger.info("EMA-dev close %s (%s)", entry.api_coin, reason)
        limit_close = executor.mid_limit_then_market and (
            str(reason).startswith("tp_") or str(reason).startswith("exit_ema")
        )
        closed = (
            executor.execute_mid_limit_close()
            if limit_close
            else executor.execute_rsi_exit()
        )
        if not closed:
            executor.emergency_flatten(reason)
        bar_t = 0
        candles = client.get_closed_candles_for(
            entry.api_coin, _ema_iv(), min_bars=_ema_bars_need()
        )
        if candles:
            bar_t = int(candles[-1]["t"])
        finish_close(entry)
        ema_store.close(coin=entry.api_coin, bar_t=bar_t)
        protected_coins.discard(entry.api_coin)
        return True

    def hydrate_ema_trade(entry: PairSetup, position) -> EmaDevTrade:
        existing = ema_store.active()
        side = str(getattr(position, "side", "") or "")
        fill = float(getattr(position, "entry_price", 0) or 0) or client.get_mark_price()
        if (
            existing is not None
            and existing.coin == entry.api_coin
            and existing.side == side
        ):
            if float(existing.opened_at or 0) <= 0:
                tracked = trade_store.get(entry.api_coin)
                if (
                    tracked is not None
                    and tracked.side == side
                    and float(tracked.opened_at or 0) > 0
                ):
                    existing.opened_at = float(tracked.opened_at)
                    ema_store._save()
            return existing
        candles = client.get_closed_candles_for(
            entry.api_coin, _ema_iv(), min_bars=_ema_bars_need()
        )
        snap = snap_from_candles(entry.api_coin, candles, _ema_period())
        ema = float(snap.ema) if snap else fill
        bar_t = int(snap.bar_t) if snap else 0
        d = abs(signed_dev_pct(fill, ema)) if ema else 1.0
        tracked = trade_store.get(entry.api_coin)
        opened_at = 0.0
        if tracked is not None and tracked.side == side:
            opened_at = float(tracked.opened_at or 0)
        trade = EmaDevTrade(
            coin=entry.api_coin,
            side=side,
            dev_pct=max(0.0, d),
            entry_px=fill,
            last_fill_px=fill,
            dca_done=False,
            entry_ema=ema,
            opened_bar_t=bar_t,
            opened_at=opened_at,
        )
        ema_store.open_trade(trade)
        logger.warning(
            "EMA-dev rehydrated %s %s @ %s (D≈%.2f%%)",
            entry.api_coin,
            side,
            fill,
            trade.dev_pct,
        )
        return trade

    def refresh_protect(entry: PairSetup, trade: EmaDevTrade, avg_entry: float, ema: float) -> bool:
        pcts = ema_dev_protect_pcts(
            trade, avg_entry, ema, **_ema_protect_kw()
        )
        if pcts is None:
            return flatten_ema(entry, "protect_invalid")
        tp_pct, sl_pct = pcts
        tracked = trade_store.get(entry.api_coin)
        if executor.mid_limit_then_market:
            resting = client.resting_tp_px()
            target = ema_tp_target(trade, avg_entry, ema)
            tp_ok = (
                resting is not None
                and target > 0
                and abs(float(resting) - target) / target * 100.0 < 0.12
            )
            same = (
                tracked is not None
                and abs(tracked.stop_loss_pct - sl_pct) < 0.08
                and client.has_exchange_sl()
                and tp_ok
            )
        else:
            same = (
                tracked is not None
                and abs(tracked.take_profit_pct - tp_pct) < 0.08
                and abs(tracked.stop_loss_pct - sl_pct) < 0.08
                and client.has_exchange_tpsl()
            )
        if same:
            return False
        status = force_attach_tpsl(tp_pct, sl_pct)
        if status == "would_take" or status == "flat":
            return flatten_ema(entry, "tp_dev")
        if status != "ok":
            return flatten_ema(entry, "tpsl_attach_failed")
        pos = client.get_position(force=True)
        fill = float(pos.entry_price or avg_entry) if pos else avg_entry
        size = float(pos.size) if pos else 0.0
        if trade_store.get(entry.api_coin) is None:
            trade_store.open_trade(
                entry.api_coin,
                trade.side,
                fill,
                size,
                tp_pct,
                sl_pct,
                equity_at_entry=client.get_account_value(force=True),
            )
            t = trade_store.get(entry.api_coin)
            started = ema_hold_started_at(trade, entry)
            if t is not None and started > 0 and started < float(t.opened_at or 0):
                t.opened_at = started
                trade_store._save()
        else:
            t = trade_store.get(entry.api_coin)
            t.take_profit_pct = tp_pct
            t.stop_loss_pct = sl_pct
            tp_px, sl_px = TradeStateStore._tp_sl_prices(
                t.side, t.entry_price, tp_pct, sl_pct
            )
            t.take_profit_price = tp_px
            t.stop_loss_price = sl_px
            trade_store._save()
        protected_coins.add(entry.api_coin)
        return False

    def manage_ema_one(entry: PairSetup, position) -> bool:
        activate_pair(client, entry)
        trade = hydrate_ema_trade(entry, position)
        candles = client.get_closed_candles_for(
            entry.api_coin, _ema_iv(), min_bars=_ema_bars_need()
        )
        snap = snap_from_candles(entry.api_coin, candles, _ema_period())
        mark = client.get_mark_price()
        ema = float(snap.ema) if snap else float(trade.entry_ema or 0)
        bar_t = int(snap.bar_t) if snap else int(trade.opened_bar_t)
        avg = float(getattr(position, "entry_price", 0) or 0) or mark
        hold_h = _ema_max_hold_hours()
        if hold_h > 0:
            started = ema_hold_started_at(trade, entry)
            hold_s = max(0.0, time.time() - started)
            if float(trade.opened_at or 0) <= 0 and started > 0:
                trade.opened_at = started
                ema_store._save()
            if hold_s >= hold_h * 3600.0:
                return flatten_ema(
                    entry,
                    f"max_hold {hold_h:.1f}h ({hold_s / 3600.0:.2f}h since entry)",
                )
        tp_target = ema_tp_target(trade, avg, ema)
        if tp_target > 0 and should_chase_maker_tp(trade.side, mark, tp_target):
            return flatten_ema(
                entry,
                f"tp_dev target={tp_target:.6g} mark={mark:.6g}",
            )
        ref = float(trade.entry_px or avg)
        if adverse_pct(trade.side, ref, mark) + 1e-12 >= trade.dev_pct:
            return flatten_ema(
                entry,
                f"stop_loss D={trade.dev_pct:.2f}% from {ref:.6g}",
            )
        dca_on = _ema_dca_on()
        if dca_on and not trade.dca_done:
            adv = adverse_pct(trade.side, trade.entry_px, mark)
            # Gapped through entry-D and the post-DCA D in one move — SL, don't add.
            if adv + 1e-12 >= 2.0 * trade.dev_pct:
                return flatten_ema(
                    entry,
                    f"stop_loss D×2={2.0 * trade.dev_pct:.2f}% from {trade.entry_px:.6g}",
                )
            if adv + 1e-12 >= trade.dev_pct:
                _entry_pct, dca_pct = ema_dev_fill_pcts(
                    float(cfg.EMA_DEV_ENTRY_PCT),
                    float(cfg.EMA_DEV_TOTAL_PCT),
                    dca_on=True,
                )
                if dca_pct <= 0:
                    return flatten_ema(entry, "sl_dca_size_zero")
                activate_pair_for_trade(client, entry)
                size_kw = dict(
                    leverage=entry.leverage,
                    sz_decimals=entry.market.sz_decimals,
                    min_notional_usd=cfg.MIN_ORDER_NOTIONAL_USD,
                )
                est = client.estimate_order_size(
                    dca_pct, margin_from="equity", **size_kw
                )
                if not est.ok:
                    # Losing trades shrink equity AND free margin. Still add with
                    # whatever collateral is left instead of flattening.
                    logger.warning(
                        "EMA-dev DCA %s %.1f%% of current equity won't fit (%s) "
                        "— sizing add to remaining free margin",
                        entry.api_coin,
                        dca_pct,
                        est.reason,
                    )
                    est = client.estimate_order_size(
                        95.0, margin_from="available", **size_kw
                    )
                if not est.ok:
                    logger.warning(
                        "EMA-dev DCA %s still cannot add (%s) — keeping first "
                        "fill, retry next poll (not SL)",
                        entry.api_coin,
                        est.reason,
                    )
                    return False
                guess = ema_dev_protect_pcts(
                    EmaDevTrade(
                        coin=trade.coin,
                        side=trade.side,
                        dev_pct=trade.dev_pct,
                        entry_px=trade.entry_px,
                        last_fill_px=mark,
                        dca_done=True,
                        entry_ema=trade.entry_ema,
                        opened_bar_t=trade.opened_bar_t,
                    ),
                    avg,
                    ema,
                    **_ema_protect_kw(),
                )
                tp_g, sl_g = guess if guess else (max(0.05, trade.dev_pct), trade.dev_pct)
                logger.info(
                    "EMA-dev DCA %s size=%s (%.1f%% of equity now, D=%.2f%% against entry)",
                    entry.api_coin,
                    est.size,
                    dca_pct,
                    trade.dev_pct,
                )
                old_avg = avg
                old_sz = float(getattr(position, "size", 0) or 0)
                new_pos = executor.execute_dca_add(est.size, tp_g, sl_g)
                if new_pos is None:
                    logger.warning(
                        "EMA-dev DCA %s add order failed — keeping first fill, retry next poll",
                        entry.api_coin,
                    )
                    return False
                new_avg = float(new_pos.entry_price or mark)
                new_sz = float(new_pos.size)
                added = new_sz - old_sz
                if added > 1e-12 and old_avg > 0:
                    fill_px = (new_avg * new_sz - old_avg * old_sz) / added
                else:
                    fill_px = mark
                ema_store.mark_dca(fill_px)
                trade = ema_store.active() or trade
                trade_store.record_dca_add(new_avg, new_sz, coin=entry.api_coin)
                protected_coins.add(entry.api_coin)
                if should_chase_maker_tp(
                    trade.side,
                    client.get_mark_price(),
                    ema_tp_target(trade, new_avg, ema),
                ):
                    return flatten_ema(entry, "tp_dev")
                return refresh_protect(entry, trade, new_avg, ema)
        if bar_t > 0:
            last_manage_bar[entry.api_coin] = bar_t
        if entry.api_coin not in protected_coins or not ema_is_protected():
            return refresh_protect(entry, trade, avg, ema)
        if executor.mid_limit_then_market:
            resting = client.resting_tp_px()
            target = ema_tp_target(trade, avg, ema)
            if target > 0 and (
                resting is None
                or abs(float(resting) - target) / target * 100.0 >= 0.12
            ):
                return refresh_protect(entry, trade, avg, ema)
        tracked = trade_store.get(entry.api_coin)
        pcts = ema_dev_protect_pcts(trade, avg, ema, **_ema_protect_kw())
        if pcts is None:
            return False
        tp_pct, sl_pct = pcts
        if tracked is None or (
            abs(tracked.take_profit_pct - tp_pct) > 0.08
            or abs(tracked.stop_loss_pct - sl_pct) > 0.08
        ):
            return refresh_protect(entry, trade, avg, ema)
        return False

    def manage_ema_positions(open_positions: list) -> bool:
        closed = False
        tracked = ema_store.active()
        keep = tracked.coin if tracked else None
        if keep is None and open_positions:
            first = find_watch_entry(watch, open_positions[0][0])
            keep = first.api_coin if first else str(open_positions[0][0])
        seen: set[str] = set()
        for coin, position in list(open_positions):
            entry = find_watch_entry(watch, coin)
            key = entry.api_coin if entry else str(coin)
            if key in seen:
                continue
            seen.add(key)
            if keep and key != keep:
                logger.warning(
                    "EMA-dev is one pair — flattening extra %s", key
                )
                if entry is None:
                    client.configure_coin(str(coin))
                    executor.emergency_flatten("ema_dev_extra_position")
                    wait_until_flat(
                        client,
                        trade_store,
                        logger,
                        coin=client.coin,
                        coin_names=frozenset({client.coin, str(coin)}),
                    )
                    drop_local(client.coin)
                else:
                    flatten_ema(entry, "ema_dev_one_pair_only")
                closed = True
                continue
            if entry is None:
                logger.warning(
                    "EMA-dev position on %s outside watch — flattening", coin
                )
                client.configure_coin(str(coin))
                executor.emergency_flatten("ema_dev_outside_watch")
                wait_until_flat(
                    client,
                    trade_store,
                    logger,
                    coin=client.coin,
                    coin_names=frozenset({client.coin, str(coin)}),
                )
                drop_local(client.coin)
                ema_store.close(coin=str(coin), bar_t=0)
                closed = True
                continue
            if manage_ema_one(entry, position):
                closed = True
        return closed

    def try_open_ema() -> bool:
        if ema_store.active() is not None:
            return False
        period = _ema_period()
        iv = _ema_iv()
        min_dev = float(getattr(cfg, "EMA_DEV_MIN_DEV_PCT", 0) or 0)
        snaps = []
        parts: list[str] = []
        by_coin = {e.api_coin: e for e in watch}
        for entry in watch:
            candles = client.get_closed_candles_for(
                entry.api_coin, iv, min_bars=_ema_scan_bars_need()
            )
            snap = snap_from_candles(entry.api_coin, candles, period)
            if snap is None:
                parts.append(f"{entry.api_coin} no-ema")
                continue
            age = f" {snap.cross_bars}b" if _ema_rank_age() else ""
            parts.append(
                f"{entry.api_coin} {snap.abs_dev_pct:.2f}% "
                f"{'below' if snap.signal_side > 0 else 'above'} ema{age}"
            )
            snaps.append(snap)
        prev = ema_store.trade
        skip_coin = prev.last_exit_coin if prev else None
        skip_bar = prev.last_exit_bar_t if prev else 0
        winner = pick_farthest(
            snaps,
            min_dev_pct=min_dev,
            skip_coin=skip_coin or None,
            skip_bar_t=skip_bar,
            rank_cross_age=_ema_rank_age(),
            reverse=flip_live,
        )
        logger.info(
            "EMA-dev scan | %s",
            " || ".join(parts) if parts else "idle",
        )
        if winner is None:
            return False
        entry = by_coin.get(winner.coin)
        if entry is None:
            return False
        signal_side = int(winner.signal_side)
        side = -signal_side if flip_live else signal_side
        entry_pct, dca_pct = ema_dev_fill_pcts(
            float(cfg.EMA_DEV_ENTRY_PCT),
            float(cfg.EMA_DEV_TOTAL_PCT),
            dca_on=_ema_dca_on(),
        )
        activate_pair_for_trade(client, entry)
        est = client.estimate_order_size(
            entry_pct,
            entry.leverage,
            sz_decimals=entry.market.sz_decimals,
            min_notional_usd=cfg.MIN_ORDER_NOTIONAL_USD,
            margin_from="equity",
        )
        if not est.ok:
            logger.warning("Skip %s — %s", entry.api_coin, est.reason)
            return False
        try:
            equity = client.get_account_value(force=True)
        except Exception:
            equity = 0.0
        if equity > 0 and est.available_margin < equity * cfg.MIN_FREE_MARGIN_FRAC:
            logger.warning(
                "Skip %s — free margin $%.2f low vs equity $%.2f",
                entry.api_coin,
                est.available_margin,
                equity,
            )
            return False
        draft = EmaDevTrade(
            coin=entry.api_coin,
            side="long" if side > 0 else "short",
            dev_pct=max(0.0, float(winner.abs_dev_pct)),
            entry_px=float(winner.close),
            last_fill_px=float(winner.close),
            dca_done=False,
            entry_ema=float(winner.ema),
            opened_bar_t=int(winner.bar_t),
        )
        pcts = ema_dev_protect_pcts(
            draft, float(winner.close), float(winner.ema), **_ema_protect_kw()
        )
        if pcts is None:
            logger.info("Skip %s — D unusable", entry.api_coin)
            return False
        tp_pct, sl_pct = pcts
        logger.info(
            "EMA-dev entry %s %s size=%s notional=$%.2f margin=%.2f%% equity "
            "D=%.2f%% ema=%.6g close=%.6g tpsl=%.2f%%/%.2f%% [%s%s]",
            entry.api_coin,
            "LONG" if side > 0 else "SHORT",
            est.size,
            est.notional_usd,
            entry_pct,
            draft.dev_pct,
            winner.ema,
            winner.close,
            tp_pct,
            sl_pct,
            mode,
            "|REVERSE" if flip_live else "",
        )
        try:
            ok = executor.execute_protected_entry(
                is_buy=side > 0,
                target_sz=est.size,
                take_profit_pct=tp_pct,
                stop_loss_pct=sl_pct,
                attach_protect=True,
            )
        except RuntimeError as exc:
            if "insufficient margin" in str(exc).lower():
                logger.warning("Insufficient margin for %s", entry.api_coin)
                return False
            raise
        if not ok:
            cleanup_closed_coin(client, trade_store, entry.api_coin)
            return False
        pos = client.get_position(force=True)
        if pos is None:
            cleanup_closed_coin(client, trade_store, entry.api_coin)
            return False
        if executor.mid_limit_then_market:
            if not client.has_exchange_sl():
                executor.emergency_flatten("unprotected")
                cleanup_closed_coin(client, trade_store, entry.api_coin)
                return False
            if not client.has_resting_tp_limit():
                logger.warning(
                    "EMA-dev %s SL is on but maker TP is not resting — retry next bar",
                    entry.api_coin,
                )
        elif not client.has_exchange_tpsl():
            executor.emergency_flatten("unprotected")
            cleanup_closed_coin(client, trade_store, entry.api_coin)
            return False
        fill = pos.entry_price or client.get_mark_price()
        draft.entry_px = float(fill)
        draft.last_fill_px = float(fill)
        draft.opened_at = time.time()
        if float(winner.ema) > 0 and float(fill) > 0:
            draft.dev_pct = abs(signed_dev_pct(float(fill), float(winner.ema)))
        ema_store.open_trade(draft)
        trade_store.open_trade(
            client.coin,
            pos.side,
            fill,
            pos.size,
            tp_pct,
            sl_pct,
            equity_at_entry=client.get_account_value(force=True),
        )
        last_manage_bar[entry.api_coin] = int(winner.bar_t)
        protected_coins.add(entry.api_coin)
        if refresh_protect(entry, draft, float(fill), float(winner.ema)):
            return True
        locked = ema_dev_protect_pcts(
            draft, float(fill), float(winner.ema), **_ema_protect_kw()
        )
        tp_now, sl_now = locked if locked else (tp_pct, sl_pct)
        logger.info(
            "Opened EMA-dev %s %s size=%s @ %s D=%.2f%% tpsl=%.2f%%/%.2f%%",
            client.coin,
            pos.side,
            pos.size,
            fill,
            draft.dev_pct,
            tp_now,
            sl_now,
        )
        return True

    def maybe_refresh_ema_universe(*, after_close: bool) -> None:
        if not after_close:
            return
        logger.info("EMA-dev refreshing pair list after close (then pick farthest from EMA)")
        refresh_universe_if_needed(force=True)

    last_hft_universe = time.time()
    _hft_log_last: dict[str, tuple[str, float]] = {}
    _hft_chop_cache: dict[str, tuple[float, object]] = {}

    def hft_log(key: str, msg: str, *args, every_s: float = 0.0) -> None:
        """Log HFT diagnostics; same text under `key` is rate-limited."""
        text = msg % args if args else msg
        now_l = time.time()
        prev = _hft_log_last.get(key)
        if (
            every_s > 0
            and prev is not None
            and prev[0] == text
            and now_l - prev[1] < every_s
        ):
            return
        _hft_log_last[key] = (text, now_l)
        logger.info(msg, *args)

    def _hft_lookback() -> int:
        return max(16, int(getattr(cfg, "HFT_LOOKBACK_BARS", 45) or 45))

    def _hft_clip_size(entry: PairSetup) -> float:
        lev = min(
            int(entry.leverage),
            max(1, int(getattr(cfg, "HFT_MAX_LEVERAGE", 20) or 20)),
        )
        try:
            equity = client.get_account_value(force=True)
        except Exception:
            equity = 0.0
        try:
            available = client.get_available_margin(force=True)
        except Exception:
            available = 0.0
        try:
            mid = client.get_mark_price()
        except Exception:
            mid = 0.0
        if mid <= 0:
            logger.info("HFT skip size %s — no mark", entry.api_coin)
            return 0.0
        notion = target_clip_notional(
            equity=equity,
            available=available,
            leverage=lev,
            min_notional=float(cfg.MIN_ORDER_NOTIONAL_USD),
            max_notional=float(getattr(cfg, "HFT_CLIP_MAX_NOTIONAL_USD", 15) or 15),
        )
        if notion is None:
            logger.info(
                "HFT skip size %s — cannot fund HL min (~$%.0f) with free $%.2f at %sx",
                entry.api_coin,
                float(cfg.MIN_ORDER_NOTIONAL_USD),
                available,
                lev,
            )
            return 0.0
        dec = entry.market.sz_decimals
        sz = ceil_size(notion / mid, dec)
        if sz * mid + 1e-9 < float(cfg.MIN_ORDER_NOTIONAL_USD):
            sz = ceil_size(float(cfg.MIN_ORDER_NOTIONAL_USD) / mid, dec)
        need = (sz * mid) / max(1, lev)
        if need > available + 1e-9:
            logger.info(
                "HFT skip size %s — clip $%.2f needs $%.2f margin, free $%.2f",
                entry.api_coin,
                sz * mid,
                need,
                available,
            )
            return 0.0
        return float(sz)

    def _hft_arm(entry: PairSetup, *, in_pos: bool) -> None:
        activate_pair(client, entry)
        if in_pos:
            return
        lev = min(
            int(entry.leverage),
            max(1, int(getattr(cfg, "HFT_MAX_LEVERAGE", 20) or 20)),
        )
        try:
            client.set_leverage(lev, is_cross=entry.use_cross_margin)
        except Exception as exc:
            logger.warning("HFT leverage %s: %s", entry.api_coin, exc)

    def _hft_chop(entry: PairSetup, *, force: bool = False):
        now_c = time.time()
        cached = _hft_chop_cache.get(entry.api_coin)
        if not force and cached is not None and now_c - cached[0] < 15.0:
            return cached[1]
        candles = client.get_closed_candles_for(
            entry.api_coin, "1m", min_bars=_hft_lookback() + 5
        )
        snap = chop_from_candles(entry.api_coin, candles, _hft_lookback())
        _hft_chop_cache[entry.api_coin] = (now_c, snap)
        return snap

    def _hft_book():
        client.invalidate_l2()
        try:
            return book_from_l2(client.l2_book(), client.sz_decimals)
        except Exception as exc:
            logger.warning("HFT L2 failed: %s", exc)
            return None

    def _hft_target_px(
        book, is_buy: bool, nudge: int = 0, *, aggressive: bool = False
    ) -> float:
        client.invalidate_l2()
        l2 = client.l2_book()
        if aggressive:
            return maker_limit_price(
                l2,
                is_buy,
                client.sz_decimals,
                attempt_index=min(2, 1 + nudge),
                passive_nudge=0,
            )
        if book.spread_bps >= 8.0:
            return mid_post_only_price(
                l2, is_buy, client.sz_decimals, passive_nudge=nudge
            )
        return maker_limit_price(
            l2,
            is_buy,
            client.sz_decimals,
            attempt_index=0,
            passive_nudge=nudge,
        )

    def hft_flatten(entry: PairSetup, reason: str) -> bool:
        activate_pair_for_trade(client, entry)
        logger.info("HFT flatten exec %s (%s)", entry.api_coin, reason)
        if not executor.execute_rsi_exit():
            executor.emergency_flatten(reason)
        wait_until_flat(
            client,
            trade_store,
            logger,
            coin=entry.api_coin,
            coin_names=entry.position_coin_names(),
        )
        drop_local(entry.api_coin)
        cool = float(getattr(cfg, "HFT_COOLDOWN_SECONDS", 30) or 30)
        hft_store.close(coin=entry.api_coin, until=time.time() + cool)
        return True

    def hft_place_quote(
        is_buy: bool, sz: float, reduce_only: bool, book, *, aggressive: bool = False
    ) -> str:
        """Place one maker quote. Returns 'filled', 'resting', or 'fail'."""
        sz = round_size(sz, client.sz_decimals)
        if sz <= 0:
            return "fail"
        for nudge in range(0, 6):
            try:
                px = _hft_target_px(book, is_buy, nudge, aggressive=aggressive)
            except Exception as exc:
                logger.warning("HFT quote px: %s", exc)
                return "fail"
            try:
                result = client.place_limit(is_buy, sz, px, reduce_only=reduce_only)
            except Exception as exc:
                logger.warning("HFT place: %s", exc)
                return "fail"
            try:
                filled, oid, alo = client.parse_fill_from_result(result)
            except RuntimeError as exc:
                msg = str(exc)
                logger.warning("HFT place reject: %s", msg)
                if reduce_only and "increase position" in msg.lower():
                    live = client.get_position(force=True)
                    if live is not None:
                        hft_store.touch_fill(
                            now=time.time(), is_buy=(live.side == "long")
                        )
                return "fail"
            if alo:
                continue
            if filled > 0:
                if not reduce_only:
                    hft_store.touch_fill(now=time.time(), is_buy=is_buy)
                logger.info(
                    "HFT fill %s %s sz=%s px=%s%s",
                    client.coin,
                    "BUY" if is_buy else "SELL",
                    filled,
                    px,
                    " reduce" if reduce_only else "",
                )
                return "filled"
            if oid is not None:
                return "resting"
        logger.warning("HFT could not rest %s on %s", "bid" if is_buy else "ask", client.coin)
        return "fail"

    def run_hft_coin(entry: PairSetup, position) -> str:
        activate_pair(client, entry)
        st = hft_store.active()
        now = time.time()
        if st is not None and not st.cleared_legacy:
            client.cancel_all_orders_for_coin()
            hft_store.mark_cleared_legacy()
        book = _hft_book()
        if book is None:
            return "idle"
        chop = _hft_chop(entry, force=True)
        clip = _hft_clip_size(entry)
        live = position
        if live is None:
            live = client.get_position(force=True)
        side = str(getattr(live, "side", "") or "") or None
        size = float(getattr(live, "size", 0) or 0)
        entry_px = float(getattr(live, "entry_price", 0) or 0)
        last_fill = float(st.last_fill_at if st else 0) or 0.0
        min_n = float(cfg.MIN_ORDER_NOTIONAL_USD)
        if live is not None and size > 1e-12:
            hft_store.reset_flat_streak()
            if last_fill <= 0:
                last_fill = now
                hft_store.touch_fill(now=now, is_buy=(side == "long"))
            elif st is not None and bool(st.last_fill_buy) != (side == "long"):
                hft_store.touch_fill(now=st.last_fill_at or now, is_buy=(side == "long"))
                last_fill = float((hft_store.active() or st).last_fill_at or last_fill)
            if book.mid > 0:
                hft_store.mark_fav(book.mid, side or "")
                st = hft_store.active() or st
        elif last_fill > 0:
            streak = hft_store.bump_flat()
            logger.info(
                "HFT %s wait-flat %s/3 — reduce-only until exchange confirms",
                entry.api_coin,
                streak,
            )
            if streak >= 3:
                cool = float(getattr(cfg, "HFT_COOLDOWN_SECONDS", 20) or 20)
                logger.info(
                    "HFT %s clip done — cooldown %.0fs",
                    entry.api_coin,
                    cool,
                )
                client.cancel_all_orders_for_coin()
                hft_store.close(coin=entry.api_coin, until=now + cool)
                return "idle"
            # Fetch missed or cover just filled. Reduce-only until confirmed flat.
            side = "long" if bool(st and st.last_fill_buy) else "short"
            size = max(clip, 1e-8)
            if entry_px <= 0:
                entry_px = book.mid
        if st is not None and now < float(st.pause_until or 0) and size <= 1e-12:
            client.cancel_entry_orders_for_coin()
            return "paused"
        loc = 0.5
        if chop is not None and chop.box_high > chop.box_low:
            loc = (book.mid - chop.box_low) / (chop.box_high - chop.box_low)
        pnl_bps = 0.0
        hold = (now - last_fill) if last_fill > 0 else 0.0
        if size > 1e-12 and entry_px > 0:
            if side == "long":
                pnl_bps = (book.mid - entry_px) / entry_px * 10_000.0
            else:
                pnl_bps = (entry_px - book.mid) / entry_px * 10_000.0
        def _ctx() -> str:
            er = chop.er if chop else 0.0
            atr = chop.atr_bps if chop else 0.0
            last_b = chop.last_bar_bps if chop else 0.0
            notion = clip * book.mid if book.mid > 0 else 0.0
            pos_bit = ""
            if size > 1e-12:
                pos_bit = (
                    f" {side} sz={size:g} entry={entry_px:g} pnl={pnl_bps:+.1f}bps "
                    f"hold={hold:.1f}s"
                )
            return (
                f"spr={book.spread_bps:.1f}bps bid={book.bid:g} ask={book.ask:g} "
                f"imb={book.imbalance:+.2f} loc={loc:.2f} er={er:.2f} atr={atr:.1f}b "
                f"last={last_b:.0f}b clip={clip:g} (${notion:.1f}){pos_bit}"
            )

        if live is not None and size > 1e-12 and book.mid > 0 and size * book.mid + 1e-9 < min_n:
            logger.info("HFT flatten %s dust leftover $%.2f | %s", entry.api_coin, size * book.mid, _ctx())
            hft_flatten(entry, "dust")
            return "flatten"
        if size > 1e-12:
            client.cancel_entry_orders_for_coin()
        holding = bool(st is not None and st.coin == entry.api_coin)
        decision = hft_decide(
            book=book,
            chop=chop,
            side=side if size > 1e-12 else None,
            size=size,
            entry_px=entry_px,
            last_fill_at=last_fill,
            now=now,
            min_spread_bps=float(getattr(cfg, "HFT_MIN_SPREAD_BPS", 2.8) or 2.8),
            max_spread_bps=float(getattr(cfg, "HFT_MAX_SPREAD_BPS", 16) or 16),
            max_er=float(getattr(cfg, "HFT_MAX_ER", 0.32) or 0.32),
            box_break_bps=float(getattr(cfg, "HFT_BOX_BREAK_BPS", 8) or 8),
            base_timeout_s=float(getattr(cfg, "HFT_INVENTORY_TIMEOUT_S", 90) or 90),
            clip_sz=clip,
            fav_px=float(st.fav_px if st else 0) or 0.0,
            holding_quotes=holding,
        )
        if decision.flatten:
            logger.info(
                "HFT flatten %s %s | %s",
                entry.api_coin,
                decision.flatten,
                _ctx(),
            )
            hft_flatten(entry, decision.flatten)
            return "flatten"
        if decision.pause:
            if size <= 1e-12:
                client.cancel_entry_orders_for_coin()
            hft_log(
                f"pause:{entry.api_coin}:{decision.pause}",
                "HFT skip %s %s | %s",
                entry.api_coin,
                decision.pause,
                _ctx(),
                every_s=20.0,
            )
            return "paused"
        if clip <= 0 and not (live is not None and size > 1e-12):
            return "idle"
        if st is None or st.coin != entry.api_coin:
            hft_store.set_coin(entry.api_coin, now=now)
            _hft_arm(entry, in_pos=size > 1e-12)
        bid_o, ask_o = client.working_limit_quotes()
        add_sz = clip
        exit_sz = size if (live is not None and size > 1e-12) else clip

        def _keep(order, is_buy: bool, *, aggressive: bool) -> bool:
            if order is None:
                return False
            raw = order.get("limitPx") or order.get("px")
            try:
                tpx = _hft_target_px(book, is_buy, aggressive=aggressive)
            except Exception:
                return False
            return quote_px_ok(float(raw or 0), tpx, book.tick, book.mid)

        placed = False
        bid_status = ""
        ask_status = ""
        if decision.quote_bid:
            if not _keep(bid_o, True, aggressive=decision.bid_reduce):
                if bid_o is not None:
                    client.cancel_oid(bid_o.get("oid"))
                bid_status = hft_place_quote(
                    True,
                    exit_sz if decision.bid_reduce else add_sz,
                    decision.bid_reduce,
                    book,
                    aggressive=decision.bid_reduce,
                )
                if bid_status != "fail":
                    placed = True
        elif bid_o is not None:
            client.cancel_oid(bid_o.get("oid"))
            bid_o = None
        if bid_status == "filled":
            if ask_o is not None:
                client.cancel_oid(ask_o.get("oid"))
            if not decision.bid_reduce:
                hft_place_quote(False, add_sz, True, book, aggressive=True)
                logger.info("HFT rest %s ask-red (cover buy) | %s", entry.api_coin, _ctx())
            return "quoted"
        if decision.quote_ask:
            if not _keep(ask_o, False, aggressive=decision.ask_reduce):
                if ask_o is not None:
                    client.cancel_oid(ask_o.get("oid"))
                ask_status = hft_place_quote(
                    False,
                    exit_sz if decision.ask_reduce else add_sz,
                    decision.ask_reduce,
                    book,
                    aggressive=decision.ask_reduce,
                )
                if ask_status != "fail":
                    placed = True
        elif ask_o is not None:
            client.cancel_oid(ask_o.get("oid"))
        if ask_status == "filled":
            if bid_o is not None:
                client.cancel_oid(bid_o.get("oid"))
            if not decision.ask_reduce:
                hft_place_quote(True, add_sz, True, book, aggressive=True)
                logger.info("HFT rest %s bid-red (cover sell) | %s", entry.api_coin, _ctx())
            return "quoted"
        if placed:
            sides = []
            if decision.quote_bid:
                sides.append("bid" + ("-red" if decision.bid_reduce else ""))
            if decision.quote_ask:
                sides.append("ask" + ("-red" if decision.ask_reduce else ""))
            logger.info(
                "HFT rest %s %s | %s",
                entry.api_coin,
                "+".join(sides) or decision.note,
                _ctx(),
            )
        return "quoted"

    def manage_hft_positions(open_positions: list) -> bool:
        closed = False
        seen: set[str] = set()
        st = hft_store.active()
        keep = st.coin if st else None
        if keep is None and open_positions:
            first = find_watch_entry(watch, open_positions[0][0])
            keep = first.api_coin if first else str(open_positions[0][0])
        for coin, position in list(open_positions):
            entry = find_watch_entry(watch, coin)
            key = entry.api_coin if entry else str(coin)
            if key in seen:
                continue
            seen.add(key)
            if keep and key != keep:
                logger.warning("HFT is one pair — flattening extra %s", key)
                if entry is None:
                    client.configure_coin(str(coin))
                    executor.emergency_flatten("hft_extra_position")
                    wait_until_flat(
                        client,
                        trade_store,
                        logger,
                        coin=client.coin,
                        coin_names=frozenset({client.coin, str(coin)}),
                    )
                    drop_local(client.coin)
                else:
                    hft_flatten(entry, "hft_one_pair_only")
                closed = True
                continue
            if entry is None:
                logger.warning("HFT position on %s outside watch — flattening", coin)
                client.configure_coin(str(coin))
                executor.emergency_flatten("hft_outside_watch")
                wait_until_flat(
                    client,
                    trade_store,
                    logger,
                    coin=client.coin,
                    coin_names=frozenset({client.coin, str(coin)}),
                )
                drop_local(client.coin)
                closed = True
                continue
            if st is None or st.coin != entry.api_coin:
                hft_store.set_coin(entry.api_coin, now=time.time())
                hft_store.touch_fill(now=time.time())
                _hft_arm(entry, in_pos=True)
                logger.info(
                    "HFT adopted %s %s size=%s @ %s (reduce-only until flat)",
                    entry.api_coin,
                    position.side,
                    position.size,
                    position.entry_price,
                )
            if run_hft_coin(entry, position) == "flatten":
                closed = True
        return closed

    def tick_hft_flat() -> None:
        st = hft_store.active()
        now = time.time()
        by_coin = {e.api_coin: e for e in watch}
        rescore_s = float(getattr(cfg, 'HFT_RESCORE_SECONDS', 90) or 90)
        max_er = float(getattr(cfg, 'HFT_MAX_ER', 0.32) or 0.32)
        min_sp = float(getattr(cfg, 'HFT_MIN_SPREAD_BPS', 2.8) or 2.8)
        max_sp = float(getattr(cfg, 'HFT_MAX_SPREAD_BPS', 16) or 16)
        max_range = float(getattr(cfg, 'HFT_MAX_RANGE_BPS', 400) or 400)
        max_n = max(1, int(getattr(cfg, 'HFT_MAX_CANDIDATES', 8) or 8))
        prev = hft_store.state
        skip_coin = prev.last_exit_coin if prev else None
        skip_until = prev.last_exit_until if prev else 0.0

        def _scan_snaps():
            snaps = []
            why = []
            for e in watch:
                ch = _hft_chop(e)
                if ch is None:
                    why.append(f'{e.api_coin}:no-chop')
                    continue
                rej = chop_reject_reason(
                    ch,
                    max_er=max_er,
                    max_range_bps=max_range,
                    skip_coin=skip_coin,
                    skip_until=skip_until,
                    now=now,
                )
                if rej:
                    why.append(f'{e.api_coin}:{rej}')
                    continue
                snaps.append(ch)
            snaps.sort(key=lambda s: (-s.score, s.coin))
            return snaps, why

        def _try_quote(snaps, *, reason: str) -> bool:
            ranked = rank_chop(
                snaps,
                max_er=max_er,
                skip_coin=skip_coin,
                skip_until=skip_until,
                now=now,
                max_range_bps=max_range,
            )
            book_bits = []
            ready = []
            for snap in ranked[:max_n]:
                entry = by_coin.get(snap.coin)
                if entry is None:
                    continue
                activate_pair(client, entry)
                book = _hft_book()
                if book is None:
                    book_bits.append(f'{snap.coin}:no-book')
                    continue
                if book.spread_bps < min_sp or book.spread_bps > max_sp:
                    cmp = '<' if book.spread_bps < min_sp else '>'
                    lim = min_sp if book.spread_bps < min_sp else max_sp
                    book_bits.append(f'{snap.coin}:spr={book.spread_bps:.1f}b{cmp}{lim:.1f}')
                    continue
                min_n = float(cfg.MIN_ORDER_NOTIONAL_USD)
                if book.bid_sz * book.mid < min_n or book.ask_sz * book.mid < min_n:
                    book_bits.append(f'{snap.coin}:thin')
                    continue
                clip = _hft_clip_size(entry)
                if clip <= 0:
                    book_bits.append(f'{snap.coin}:no-size')
                    continue
                ready.append((book.spread_bps, snap, entry))
                book_bits.append(f'{snap.coin}:ok spr={book.spread_bps:.1f}b')
            ready.sort(key=lambda row: abs(row[0] - 6.0))
            for _spr, snap, entry in ready:
                outcome = run_hft_coin(entry, None)
                if outcome in ('quoted', 'flatten'):
                    return True
                book_bits.append(f'{snap.coin}:{outcome}')
            hft_log(
                'idle',
                'HFT idle %s need>=%.1fbps | %s',
                reason,
                min_sp,
                ' '.join(book_bits) if book_bits else 'no L2 names',
                every_s=25.0,
            )
            return False

        if st is not None:
            entry = by_coin.get(st.coin)
            if entry is None:
                logger.info('HFT %s left watch — dropping quotes', st.coin)
                client.cancel_all_orders_for_coin_named(st.coin)
                cool = float(getattr(cfg, 'HFT_COOLDOWN_SECONDS', 30) or 30)
                hft_store.close(coin=st.coin, until=now + cool)
                return
            activate_pair(client, entry)
            live = client.get_position(force=True)
            if live is not None:
                run_hft_coin(entry, live)
                return
            if now - float(st.last_score_at or 0) >= rescore_s:
                snaps, why = _scan_snaps()
                hft_store.mark_scored(now)
                ranked = rank_chop(
                    snaps,
                    max_er=max_er,
                    skip_coin=st.coin,
                    skip_until=0.0,
                    now=now,
                    max_range_bps=max_range,
                )
                better = ranked[0] if ranked else None
                if (
                    better is not None
                    and better.coin != st.coin
                    and float(st.last_fill_at or 0) <= 0
                ):
                    logger.info(
                        'HFT rescore %s -> try %s | %s',
                        st.coin,
                        better.coin,
                        ' '.join(why) if why else 'all chop-ok',
                    )
                    client.cancel_all_orders_for_coin_named(st.coin)
                    hft_store.close(coin=st.coin, until=now)
                    if _try_quote(snaps, reason='rescore'):
                        return
            run_hft_coin(entry, None)
            return
        snaps, why = _scan_snaps()
        if why:
            hft_log(
                'chop',
                'HFT chop | ok=%s skip=%s',
                ','.join(s.coin for s in snaps) or 'none',
                ' '.join(why),
                every_s=25.0,
            )
        _try_quote(snaps, reason='pick')

    while not stop.is_set():
        try:
            if not first_iter:
                if stop.wait(wake_seconds()):
                    break
            first_iter = False

            pos_ok, open_positions = client.fetch_open_positions(force=True)
            if not pos_ok:
                if stop.wait(cfg.POSITION_POLL_SECONDS):
                    break
                continue

            closed_any = False
            if hft_on:
                closed_any = manage_hft_positions(open_positions)
            elif ema_dev_on:
                closed_any = manage_ema_positions(open_positions)
            else:
                seen_manage: set[str] = set()
                for coin, position in list(open_positions):
                    entry = find_watch_entry(watch, coin)
                    key = entry.api_coin if entry else str(coin)
                    if key in seen_manage:
                        continue
                    seen_manage.add(key)
                    if manage_one(coin, position):
                        closed_any = True

            if closed_any:
                pos_ok, open_positions = client.fetch_open_positions(force=True)
                if not pos_ok:
                    continue
            occupied = occupied_api(open_positions)

            if hft_on:
                st = hft_store.active()
                hft_busy = bool(
                    occupied
                    or (st is not None and float(st.last_fill_at or 0) > 0)
                )
                if hft_busy:
                    was_in_position = True
                    if not occupied and st is not None:
                        entry = find_watch_entry(watch, st.coin)
                        if entry is not None:
                            run_hft_coin(entry, None)
                    continue
                if was_in_position:
                    was_in_position = False
                refresh_s = float(
                    getattr(cfg, "HFT_UNIVERSE_REFRESH_SECONDS", 600) or 600
                )
                if time.time() - last_hft_universe >= refresh_s:
                    logger.info("HFT refreshing pair list")
                    refresh_universe_if_needed(force=True)
                    last_hft_universe = time.time()
                tick_hft_flat()
                continue

            if ema_dev_on:
                tracked = ema_store.active()
                if tracked is not None and tracked.coin not in occupied:
                    logger.info(
                        "EMA-dev %s closed on exchange — clearing local state",
                        tracked.coin,
                    )
                    bar_t = 0
                    try:
                        candles = client.get_closed_candles_for(
                            tracked.coin, _ema_iv(), min_bars=_ema_bars_need()
                        )
                        if candles:
                            bar_t = int(candles[-1]["t"])
                    except Exception:
                        bar_t = 0
                    ema_store.close(coin=tracked.coin, bar_t=bar_t)
                    drop_local(tracked.coin)

            if not occupied:
                just_closed = bool(was_in_position)
                if was_in_position or trade_store.trades:
                    cleanup_when_flat(
                        client, trade_store, extra_coins=cleanup_coins
                    )
                    clear_active_setup()
                    open_setup_mem.clear()
                    protected_coins.clear()
                    was_in_position = False
                if ema_dev_on:
                    maybe_refresh_ema_universe(after_close=just_closed)
                else:
                    maybe_tune(force=not store.setups())
            else:
                was_in_position = True

            if ema_dev_on:
                if not occupied:
                    try_open_ema()
                continue

            can_scan = concurrent or not occupied
            if can_scan and max_concurrent > 0 and len(occupied) >= max_concurrent:
                logger.info(
                    "At max concurrent positions (%s) — not scanning new entries",
                    max_concurrent,
                )
                can_scan = False
            if not can_scan:
                continue

            rev = flip_live
            candidates: list[tuple] = []
            parts: list[str] = []
            for entry in watch:
                if entry.api_coin in occupied or occupied.intersection(
                    entry.position_coin_names()
                ):
                    parts.append(f"{entry.api_coin} in-pos")
                    continue
                setups = store.setups_for(entry.api_coin)
                if not setups:
                    parts.append(f"{entry.api_coin} no-setup")
                    continue
                hits: list[tuple] = []
                for setup in setups:
                    need = max(80, int(setup.aux) + 50, 120)
                    if setup.is_mtf:
                        vote_ivs = list(setup.mtf_intervals) or list(cfg.INTERVALS)
                        if setup.interval not in vote_ivs:
                            vote_ivs = [setup.interval] + vote_ivs
                        multi: dict[str, list] = {}
                        for iv in vote_ivs:
                            multi[iv] = client.get_closed_candles_for(
                                entry.api_coin,
                                iv,
                                min_bars=need if iv == setup.interval else 80,
                            )
                        exec_candles = multi.get(setup.interval) or []
                        if len(exec_candles) < 40:
                            continue
                        bar_t = int(exec_candles[-1]["t"])
                        sig = entry_signal(
                            setup, exec_candles, multi_candles=multi
                        )
                        snap = mtf_consensus_snapshot(setup, multi)
                        tag = f"MTF@{setup.interval}:{setup.name}"
                        if sig != 0:
                            order = -sig if rev else sig
                            hits.append((entry, sig, setup, bar_t, multi))
                            parts.append(
                                f"{entry.api_coin} [{tag}] YES "
                                f"{'LONG' if sig > 0 else 'SHORT'}"
                                f"→{'SHORT' if order < 0 else 'LONG'}"
                                f"{'(REV)' if rev else ''} ({snap})"
                            )
                        else:
                            parts.append(f"{entry.api_coin} [{tag}] no ({snap})")
                    else:
                        candles = client.get_closed_candles_for(
                            entry.api_coin, setup.interval, min_bars=need
                        )
                        if len(candles) < 40:
                            continue
                        bar_t = int(candles[-1]["t"])
                        sig = entry_signal(setup, candles)
                        tag = f"{setup.interval}:{setup.name}"
                        if sig != 0:
                            order = -sig if rev else sig
                            hits.append((entry, sig, setup, bar_t, None))
                            parts.append(
                                f"{entry.api_coin} [{tag}] YES "
                                f"{'LONG' if sig > 0 else 'SHORT'}"
                                f"→{'SHORT' if order < 0 else 'LONG'}"
                                f"{'(REV)' if rev else ''}"
                            )
                        else:
                            parts.append(f"{entry.api_coin} [{tag}] no")
                if hits:
                    candidates.append(max(hits, key=lambda h: h[2].rank_score))

            logger.info("Scan | %s", " || ".join(parts) if parts else "idle")
            if not candidates:
                continue

            candidates.sort(key=lambda c: c[2].rank_score, reverse=True)
            for entry, _sig, setup, bar_t, multi in candidates:
                if entry.api_coin in occupied:
                    continue
                if max_concurrent > 0 and len(occupied) >= max_concurrent:
                    break
                key = (entry.api_coin, setup.interval, bar_t, setup.sid)
                if last_entry_key.get(entry.api_coin) == key:
                    continue
                last_entry_key[entry.api_coin] = key
                need = max(80, int(setup.aux) + 50, 120)
                confirm_multi: dict[str, list] | None = None
                if setup.is_mtf:
                    vote_ivs = list(setup.mtf_intervals) or list(cfg.INTERVALS)
                    if setup.interval not in vote_ivs:
                        vote_ivs = [setup.interval] + vote_ivs
                    confirm_multi = {}
                    for iv in vote_ivs:
                        confirm_multi[iv] = client.get_closed_candles_for(
                            entry.api_coin,
                            iv,
                            min_bars=need if iv == setup.interval else 80,
                        )
                    confirm = confirm_multi.get(setup.interval) or []
                else:
                    confirm = client.get_closed_candles_for(
                        entry.api_coin, setup.interval, min_bars=need
                    )
                if len(confirm) < 40 or int(confirm[-1]["t"]) != bar_t:
                    logger.info("Skip %s — candle rolled before entry", entry.api_coin)
                    continue
                if try_open(entry, setup, bar_t, confirm, confirm_multi):
                    occupied.add(entry.api_coin)

        except Exception as exc:
            if _is_transient_network_error(exc):
                logger.warning("Network glitch — retry in 15s: %s", exc)
                if stop.wait(15):
                    break
                continue
            logger.exception("Main loop error: %s", exc)
            if stop.wait(5):
                break

    logger.info("Stopped")


if __name__ == "__main__":
    main()
