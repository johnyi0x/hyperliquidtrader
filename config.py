"""
User configuration for hl-multi-strategy-bot.

Edit this file. Live bot and one-shot backtest both import from here.
"""

from __future__ import annotations

# =============================================================================
# PAIR SELECTION
# =============================================================================
# "manual"     = use PAIRS (+ PAIR_LEVERAGE) exactly as listed (current behavior).
# "top_volume" = auto-pick the highest 24h notional-volume perps, then tune all,
#                then keep only MAX_LIVE_PAIRS winners for live (same ratio idea
#                as 14→5, but e.g. 50→15 for more frequent trades).
PAIR_SELECTION_MODE = "top_volume"

# --- top_volume mode only ---
# How many highest-volume perps to backtest each tune (before MAX_LIVE_PAIRS cut).
TOP_VOLUME_COUNT = 27
# Skip markets whose exchange max leverage is below this (e.g. 3x/5x memes).
# 0 = no filter. 10 = only pairs with maxLev ≥ 10, then take top TOP_VOLUME_COUNT
# by volume among those (still fills N if ≥N qualifying markets exist).
MIN_MAX_LEVERAGE = 10
# Skip markets whose exchange max leverage is ABOVE this (exclude ultra-high lev).
# 0 = no ceiling. 20 = only pairs with maxLev ≤ 20 (after the min filter).
MAX_MAX_LEVERAGE = 20
# Which books to rank by volume:
#   "native"   = Hyperliquid main perps only (no HIP-3)
#   "include"  = native + HIP-3 (xyz:...)
#   "xyz_only" = HIP-3 builder dexes only (e.g. xyz:SKHY)
XYZ_PAIR_MODE = "include"
# Legacy alias (used only if XYZ_PAIR_MODE is missing/invalid):
# False → native, True → include. Prefer XYZ_PAIR_MODE.
INCLUDE_XYZ_PAIRS = False
# True = each discovered pair uses that market's exchange max leverage.
USE_MAX_LEVERAGE = True


def xyz_pair_mode() -> str:
    """Resolve volume-scan book scope for top_volume mode."""
    raw = str(globals().get("XYZ_PAIR_MODE", "") or "").strip().lower()
    aliases = {
        "native": "native",
        "main": "native",
        "hl": "native",
        "include": "include",
        "both": "include",
        "all": "include",
        "xyz_only": "xyz_only",
        "xyz": "xyz_only",
        "hip3": "xyz_only",
        "hip-3": "xyz_only",
        "only_xyz": "xyz_only",
    }
    if raw in aliases:
        return aliases[raw]
    try:
        return "include" if bool(INCLUDE_XYZ_PAIRS) else "native"
    except NameError:
        return "native"

# --- manual mode (kept; ignored for coin list when PAIR_SELECTION_MODE=top_volume) ---
# Examples:
#   ("BTC",)                    # main book
#   ("CASHCAT",)                # main book meme
#   ("xyz:SPCX",)               # HIP-3 builder dex
#   ("BTC", "ETH", "xyz:SPCX")  # multi-pair
# HIP-3 tickers must match HL exactly (not company names).
# SK Hynix -> xyz:SKHY | Samsung -> xyz:SMSN | Sandisk -> xyz:SNDK
PAIRS: tuple[str, ...] = ("BTC","ETH","SOL","HYPE","ZEC", "UNI", "PUMP", "AAVE", "BNB","DOGE","NEAR","ONDO","WLD","SUI")

# Default leverage for any pair not listed in PAIR_LEVERAGE (manual mode).
# In top_volume + USE_MAX_LEVERAGE, exchange max is used instead (overrides still apply).
LEVERAGE = 10
# Optional per-pair overrides (keys: same as PAIRS, or api name like "BTC").
# Example: {"BTC": 40, "ETH": 25, "UNI": 10}
# Missing / empty → every pair uses LEVERAGE. Safe to leave {}.
PAIR_LEVERAGE: dict[str, int] = {"BTC": 40, "ETH": 25, "SOL": 20}

USE_CROSS_MARGIN = True  # auto-falls back to isolated if required
BALANCE_PCT_DEFAULT = 30.0  # fallback total margin budget if tune has no balance

# =============================================================================
# CANDLES / INTERVALS
# =============================================================================
# All intervals feed the multi-TF consensus (votes + weights).
INTERVALS: tuple[str, ...] = ("1m", "3m", "5m", "15m", "30m", "1h")

# Requested bars per interval (capped by live HL max after probe).
REQUESTED_CANDLES = 5000

# =============================================================================
# STRATEGY MODE
# =============================================================================
# "mtf"    = true multi-timeframe: ALL intervals vote; trade only when consensus
#            allows the direction AND an execution-TF trigger fires.
# "legacy" = old behavior: independent strategy per interval, pick best fire.
STRATEGY_MODE = "mtf"

# Execution clock for MTF entries/exits/DCA (must be in INTERVALS).
MTF_EXEC_INTERVAL = "1m"

# Legacy-only: keep one best setup per interval (ignored when STRATEGY_MODE=mtf).
KEEP_BEST_PER_INTERVAL = False

# =============================================================================
# BACKTEST / TUNE
# =============================================================================
# "fast" = smaller refine/consensus grids (much quicker; still screens all entries).
# "full" = original thorough search (slower, denser grids).
TUNE_PROFILE = "full"

BACKTEST_REFRESH_HOURS = 24.0  # live retune period (change freely)
MIN_WIN_RATE_PCT = 52.0
# Soft frequency target for ranking (not a hard per-day quota).
# Higher values (e.g. 12) make MTF winners rarer / harder to pass the screen.
TARGET_TRADES_PER_DAY = 8.0
MIN_TRADES_ABS = 5

# If more than this many pairs produce a winner, keep only the top-N by rank_score
# for live scanning (saves Hyperliquid IP weight). ≤N winners → keep all.
# IMPORTANT: 1 starves live to a single coin (often silent for days).
# Example ratios: manual 14→5, top_volume 50→15.
MAX_LIVE_PAIRS = 9

# Tuner-only: which margin % to simulate when ranking setups. Live sizing ignores
# this — see TOTAL_BALANCE_PCT / BALANCE_SPLIT_POSITIONS below.
BALANCE_PCT_GRID: tuple[float, ...] = (
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
)

# Staged search: screen entries cheaply, then refine top-N with exits/DCA/balance.
# In TUNE_PROFILE=fast this is capped at 6 automatically.
SCREEN_TOP_N = 12
TAKER_FEE_PCT = 0.045

# =============================================================================
# EXIT LAYERS (any combination; at least one must be True)
# =============================================================================
USE_TP_SL = True  # exchange TP/SL when live; backtest grids distances
USE_EXIT_SIGNAL = True  # indicator / math exit on closed bar
USE_MAX_HOLD = True  # timeout safety

MAX_POSITION_HOURS = 2.0  # hard time stop when USE_MAX_HOLD
# Fallback TP/SL spot % before first tune (live only).
TAKE_PROFIT_PCT = 1.0
STOP_LOSS_PCT = 1.0

# =============================================================================
# DCA
# =============================================================================
# True = add one extra same-size fill when price moves against the position.
# Pair budget (TOTAL_BALANCE_PCT / BALANCE_SPLIT_POSITIONS) is split equally
# across entry + DCA_MAX_ADDS extra fills. 1 extra add = two equal legs.
ALLOW_DCA = False
DCA_MAX_ADDS = 1


# =============================================================================
# LIVE / PAPER EXECUTION
# =============================================================================
# True  = paper (simulated fills @ live mids, same fees/TP-SL/DCA logic, no real orders)
# False = real Hyperliquid orders
# Switch only this flag — strategy, sizing, exits, and daily tune stay identical.
PAPER_TRADING = False
PAPER_START_BALANCE = 1000.0
USE_MARKET_ORDERS = True
MARKET_ORDER_SLIPPAGE = 0.05
POSITION_POLL_SECONDS = 5
USE_TESTNET = False
MIN_ORDER_NOTIONAL_USD = 10.0
# Skip a NEW entry when free margin is below this fraction of equity.
MIN_FREE_MARGIN_FRAC = 0.04
# True: keep scanning for new coins while others are open.
ALLOW_CONCURRENT_POSITIONS = True
# Live/paper sizing (ignores the tuner's saved balance_pct):
#   TOTAL_BALANCE_PCT = max combined margin vs account equity (leave a little free).
#   BALANCE_SPLIT_POSITIONS = how many equal slices that 95% is split into.
# Example $1000 equity: budget $950 (capped 95% in live sizing), 5 pairs.
# Each pair gets 19% of equity total; with DCA_MAX_ADDS=1 that is two equal
# ~9.5% fills (entry + one add). A 6th coin is blocked.
TOTAL_BALANCE_PCT = 55.0
BALANCE_SPLIT_POSITIONS = 5
MAX_CONCURRENT_POSITIONS = 5
# IP-weight headroom (Hyperliquid 1200/min). 50 is for a dedicated Railway IP.
# Raise to 250 if a browser or extra bots share the same IP.
IP_WEIGHT_RESERVE = 50

# Live/paper only: reverse every order vs the backtested signal.
# True  → same entry bars as the tuned mask, but buy↔sell flipped at order time.
#         Frequency matches the original; DCA/TP/SL follow the real position.
# False → orders match the backtest side.
# Backtest/tune is NEVER reversed — only live/paper execution.
REVERSE_STRATEGY = True
FLIP_EXECUTION = True # legacy alias; either True enables reverse


def reverse_orders_enabled() -> bool:
    """True when live/paper should invert buy↔sell vs backtest signals."""
    try:
        rev = bool(REVERSE_STRATEGY)
    except NameError:
        rev = False
    try:
        flip = bool(FLIP_EXECUTION)
    except NameError:
        flip = False
    return rev or flip


def requested_leverage_for(pair: str, *aliases: str) -> int:
    """Resolve leverage for a pair; falls back to LEVERAGE if not overridden."""
    default = max(1, int(LEVERAGE))
    overrides = PAIR_LEVERAGE if isinstance(PAIR_LEVERAGE, dict) else {}
    if not overrides:
        return default
    keys = [str(pair).strip()]
    keys.extend(str(a).strip() for a in aliases if a)
    # Also try bare symbol when key is dex:SYMBOL
    for k in list(keys):
        if ":" in k:
            keys.append(k.split(":", 1)[-1])
    for k in keys:
        if not k:
            continue
        if k in overrides:
            return max(1, int(overrides[k]))
        # case-insensitive match
        for ok, ov in overrides.items():
            if str(ok).strip().upper() == k.upper():
                return max(1, int(ov))
    return default
