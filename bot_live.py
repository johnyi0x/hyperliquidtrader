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
from src.engine import (
    dca_should_add,
    entry_signal,
    exit_signal,
    position_leg_count,
    setup_to_dict,
    total_balance_pct,
)
from src.exchange_client import HyperliquidClient
from src.hl_rate_limit import (
    RequestGuard,
    ThrottledInfo,
    _is_transient_network_error,
    default_shared_budget,
)
from src.logger import setup_logger
from src.market_resolver import parse_coin_input, sdk_perp_dexs_for_dexes
from src.mtf import mtf_consensus_snapshot
from src.order_executor import OrderExecutor
from src.pair_universe import resolve_pair_universe
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
            "Watch %s | api=%s szDecimals=%s maxLev=%s using %sx",
            raw,
            client.coin,
            client.market.sz_decimals,
            client.max_leverage,
            lev,
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


def live_margin_pcts(_setup: object | None = None) -> tuple[float, float, int]:
    """Live size: (total_budget_pct, per_position_pct, split_count).

    Per trade uses TOTAL_BALANCE_PCT / BALANCE_SPLIT_POSITIONS of *equity*,
    not leftover free margin. DCA is off; one fill per position.
    """
    total = min(95.0, max(1.0, float(getattr(cfg, "TOTAL_BALANCE_PCT", 95.0) or 95.0)))
    n = max(1, int(getattr(cfg, "BALANCE_SPLIT_POSITIONS", 3) or 3))
    return total, total / float(n), n


def pos_side_int(side: str) -> int:
    return 1 if side == "long" else -1


def main() -> None:
    if not cfg.USE_TP_SL and not cfg.USE_EXIT_SIGNAL and not cfg.USE_MAX_HOLD:
        raise ValueError("Enable at least one of USE_TP_SL, USE_EXIT_SIGNAL, USE_MAX_HOLD")
    pair_mode = str(getattr(cfg, "PAIR_SELECTION_MODE", "manual") or "manual").strip().lower()
    if pair_mode not in ("manual", "top_volume", "volume", "auto_volume"):
        raise ValueError(
            f"PAIR_SELECTION_MODE must be 'manual' or 'top_volume' (got {pair_mode!r})"
        )
    if pair_mode == "manual":
        pairs = tuple(cfg.PAIRS) if not isinstance(cfg.PAIRS, str) else (cfg.PAIRS,)
        if not pairs:
            raise ValueError("PAIRS must list at least one pair")
    else:
        if int(getattr(cfg, "TOP_VOLUME_COUNT", 0) or 0) < 1:
            raise ValueError("TOP_VOLUME_COUNT must be >= 1 for top_volume mode")
        if int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0) < 0:
            raise ValueError("MIN_MAX_LEVERAGE must be >= 0")
        if int(getattr(cfg, "MAX_MAX_LEVERAGE", 0) or 0) < 0:
            raise ValueError("MAX_MAX_LEVERAGE must be >= 0")
        pairs = ()
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

    logger = setup_logger("hl-multi-bot", LOG_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    housekeep_data_dir(DATA_DIR, logger=logger)
    trade_store = TradeStateStore(DATA_DIR / "trade_state.json")
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
    pair_inputs = [c for c, _ in universe]
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

    watch = init_pairs(client, universe, logger)
    cleanup_coins = [e.api_coin for e in watch]
    # Default tune leverage; per-coin overrides come from each PairSetup.
    tune_leverage = watch[0].leverage if watch else cfg.LEVERAGE
    leverage_by_coin = build_leverage_map(watch)
    universe_built_at = time.time()

    executor = OrderExecutor(
        client,
        wait_seconds=30,
        max_attempts=5,
        logger=logger,
        use_market_orders=cfg.USE_MARKET_ORDERS,
        market_slippage=cfg.MARKET_ORDER_SLIPPAGE,
    )

    def refresh_universe_if_needed() -> None:
        """Re-rank volume leaders before each tune in top_volume mode."""
        nonlocal watch, cleanup_coins, tune_leverage, leverage_by_coin, universe_built_at
        if pair_mode not in ("top_volume", "volume", "auto_volume"):
            return
        # Startup already ranked volume — skip duplicate refresh on first tune.
        if time.time() - universe_built_at < 90.0:
            return
        logger.info(
            "Refreshing top-volume universe (n=%s xyz_mode=%s min_maxLev≥%s max_maxLev≤%s)",
            int(getattr(cfg, "TOP_VOLUME_COUNT", 50) or 50),
            cfg.xyz_pair_mode() if hasattr(cfg, "xyz_pair_mode") else getattr(cfg, "INCLUDE_XYZ_PAIRS", False),
            int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0),
            int(getattr(cfg, "MAX_MAX_LEVERAGE", 0) or 0) or "off",
        )
        fresh = resolve_pair_universe(
            client.info,
            mode=pair_mode,
            manual_pairs=tuple(cfg.PAIRS)
            if not isinstance(cfg.PAIRS, str)
            else (cfg.PAIRS,),
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
        watch = init_pairs(client, fresh, logger)
        cleanup_coins = [e.api_coin for e in watch]
        tune_leverage = watch[0].leverage if watch else cfg.LEVERAGE
        leverage_by_coin = build_leverage_map(watch)
        universe_built_at = time.time()

    def run_tune() -> None:
        nonlocal watch, cleanup_coins, tune_leverage, leverage_by_coin
        refresh_universe_if_needed()
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
            store.save_results(results, leverage=tune_leverage)
            # In top_volume mode, live-scan only kept winners (full set rebuilt on retune).
            if pair_mode in ("top_volume", "volume", "auto_volume"):
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
        "hold=%s | dca=%s | reverse=%s",
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
        cfg.reverse_orders_enabled(),
    )
    concurrent = bool(getattr(cfg, "ALLOW_CONCURRENT_POSITIONS", True))
    max_concurrent = int(getattr(cfg, "MAX_CONCURRENT_POSITIONS", 0) or 0)
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
    live_total, live_per, live_splits = live_margin_pcts()
    logger.info(
        "Sizing: %.0f%% of equity budget ÷ %s positions = %.2f%% of equity per trade "
        "(no DCA; equal slices, not leftover free margin)",
        live_total,
        live_splits,
        live_per,
    )
    if pair_mode in ("top_volume", "volume", "auto_volume"):
        logger.info(
            "Top-volume settings: count=%s xyz_mode=%s use_max_lev=%s min_maxLev≥%s max_maxLev≤%s",
            int(getattr(cfg, "TOP_VOLUME_COUNT", 50) or 50),
            cfg.xyz_pair_mode() if hasattr(cfg, "xyz_pair_mode") else getattr(cfg, "INCLUDE_XYZ_PAIRS", False),
            bool(getattr(cfg, "USE_MAX_LEVERAGE", True)),
            int(getattr(cfg, "MIN_MAX_LEVERAGE", 0) or 0),
            int(getattr(cfg, "MAX_MAX_LEVERAGE", 0) or 0) or "off",
        )
    if getattr(cfg, "PAIR_LEVERAGE", None):
        logger.info(
            "Per-pair leverage: %s",
            ", ".join(f"{e.api_coin}={e.leverage}x" for e in watch),
        )
    if cfg.reverse_orders_enabled():
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
    if pos_ok and not open_positions:
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
            lefts = [
                cfg.MAX_POSITION_HOURS * 3600.0 - (now - t.opened_at)
                for t in trade_store.trades.values()
            ]
            if lefts:
                wait = min(wait, max(0.4, min(lefts)))
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
            if (
                setup
                and setup.dca_enabled
                and cfg.ALLOW_DCA
                and use_tpsl
                and t is not None
            ):
                if dca_should_add(
                    setup,
                    avg_entry_px=float(t.entry_price),
                    mark_or_close=close_px,
                    position_side=t.side,
                    dca_adds_done=int(t.dca_adds),
                ):
                    add_sz = float(t.initial_size or t.size)
                    legs = position_leg_count(setup)
                    total_bal, _, _ = live_margin_pcts(setup)
                    logger.info(
                        "DCA add %s on closed bar +size≈%s (equal leg %s/%s, "
                        "total bal=%.0f%%%s)",
                        entry.api_coin,
                        add_sz,
                        int(t.dca_adds) + 2,
                        legs,
                        total_bal,
                        " REVERSE" if cfg.reverse_orders_enabled() else "",
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
        rev = cfg.reverse_orders_enabled()
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
        total_bal, bal, slices = live_margin_pcts()
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
            "margin=%.2f%% of equity (1/%s of %.0f%% budget) tpsl=%s [%s%s]",
            entry.api_coin,
            "LONG" if is_buy else "SHORT",
            setup.name,
            setup.interval,
            est.size,
            est.notional_usd,
            bal,
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

            if not occupied:
                if was_in_position or trade_store.trades:
                    cleanup_when_flat(
                        client, trade_store, extra_coins=cleanup_coins
                    )
                    clear_active_setup()
                    open_setup_mem.clear()
                    protected_coins.clear()
                    was_in_position = False
                maybe_tune(force=not store.setups())
            else:
                was_in_position = True

            can_scan = concurrent or not occupied
            if can_scan and max_concurrent > 0 and len(occupied) >= max_concurrent:
                logger.info(
                    "At max concurrent positions (%s) — not scanning new entries",
                    max_concurrent,
                )
                can_scan = False
            if not can_scan:
                continue

            rev = cfg.reverse_orders_enabled()
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
