"""
Per-instance overrides: local | cloud | research.

Set PMF_PROFILE=local|cloud|research (default: local).
Data folders: data-local / data-cloud / data-research.

cloud  = live trade only (do not change while running).
local  = gather-only (identical to research; data-local/).
research = gather-only (data-research/).
"""

from __future__ import annotations

# =============================================================================
# CROWD LIST SIZES — adjust here; all profiles inherit these.
# =============================================================================
# Wallets local/research gather into books.jsonl (full research pool).
RESEARCH_GATHER_SIZE = 200
# Wallets live cloud polls and backtest uses for votes / agreement %.
TRADE_BASKET_SIZE = 50

# Copy-mode knobs (RUN_MODE=copy | copy_reverse). Crowd mode ignores these.
# Pick highest RANK_WINDOW (week=7d) ROI wallets that pass minimum fill/gap/WR floors.
# No max ROI / max-fills caps. Full watchlist (COPY_TOP_N) required before trading.
COPY_TOP_N = 2
COPY_REQUIRE_FULL_WATCHLIST = True
COPY_CANDIDATE_SCAN = 2000
COPY_FILL_FETCH_MAX = 350
COPY_MAX_ROI = 0.0
COPY_MIN_EQUITY = 1000.0
COPY_REFRESH_HOURS = 4.0
COPY_INCOMPLETE_RETRY_MIN = 8.0
COPY_RESELECT_HOURS = 4.0
COPY_LOOKBACK_DAYS = 7.0
COPY_HISTORY_DAYS = 30.0
# Frequent traders OK; reject sub-~2m tape only. 0 = no maximum.
COPY_IDEAL_TRADES_PER_HOUR = 3.0
COPY_MIN_MEDIAN_GAP_S = 90.0
COPY_MAX_MEDIAN_GAP_S = 0.0
COPY_MIN_FILLS = 10
COPY_MAX_FILLS = 0
COPY_MIN_FILLS_PER_DAY = 6.0
COPY_MAX_FILLS_PER_DAY = 0.0
COPY_MIN_WIN_RATE = 0.40
COPY_MIN_HIST_WIN_RATE = 0.38
COPY_MIN_RECENT_PNL = 0.0
COPY_MIN_HIST_PNL = 0.0
COPY_MIN_PROFIT_FACTOR = 1.0
COPY_MAX_FAST_FLIP_RATIO = 0.0
COPY_MIN_HOLD_S = 90.0
COPY_MAX_POSITIONS = 3
COPY_MIN_FRESH_LEADERS_PCT = 0.5
COPY_IDEAL_GAP_S = 600.0
COPY_REBALANCE_COOLDOWN_S = 45.0
COPY_FILL_SLEEP_S = 0.7

# Live trading — cloud only. Leave alone while Railway is live.
# Cloud _TRADE last tuned: 2026-08-27 05:08 UTC strategy=cloud_all score=-16.036601627936605 ret=4.991863669691288%
_TRADE: dict = {
    "BASKET_FILTER_MODE": 'off',
    "MAX_BOOK_CHANGES_PER_HOUR": 6,
    "FLOW_EMA_ALPHA": 0.28,
    "OPEN_CONFIRM_S": 330.0,
    "EXIT_RAW_FLOW": -0.045,
    "EXIT_AGREEMENT_GIVEBACK": 0.34,
    "REBALANCE_COOLDOWN_S": 300.0,
    "STICKY_BOOK_SLOTS": True,
    "RANK_WINDOW": "week",
    "BASKET_SIZE": TRADE_BASKET_SIZE,
    "CANDIDATE_POOL": TRADE_BASKET_SIZE,
    "RESEARCH_DATA_ENABLED": False,
    "RESEARCH_ONLY": False,
    "BACKTEST_LIVE_STRATEGY": 'cloud_all',
    "CONV_GIVEBACK": 0.38,
    "EXIT_AVG_CONVICTION": 0.024,
    "EXIT_FLOW": -0.013,
    "EXIT_SIDE_AGREEMENT": 0.055,
    "LIVE_CANDLE_SEED": False,
    "MIN_AVG_CONVICTION": 0.04,
    "MIN_ENTRY_FLOW": 0.006,
    "MIN_SIDE_AGREEMENT": 0.12,
    "LIVE_CANDLES_PER_TICK": 1,
    "LIVE_CANDLE_BARS_15M": 64,
    "LIVE_CANDLE_BARS_1H": 48,
    "LIVE_CANDLE_BARS_1M": 120,
    "LIVE_CANDLE_COOLDOWN_S": 8.0,
    "SWING_BAND_PCT": 0.02,
    "SWING_BREAK_PCT": 0.008,
    "SWING_ENTRY": 'ema_pullback',
    "SWING_EXIT_RSI": 60.0,
    "SWING_LOOKBACK_S": 5400.0,
    "SWING_MAX_HOLD_S": 54000.0,
    "SWING_META_MODE": 'follow',
    "SWING_REENTRY_S": 300.0,
    "SWING_RSI_BUY": 45.0,
    "SWING_RSI_SELL": 62.5,
    "SWING_SL_PCT": 1.8,
    "SWING_TF": '1h',
    "SWING_TP_PCT": 3.0,
    "DUMP_LOOKBACK_S": 1350.0,
    "DUMP_RANGE_PCT": -0.04,
    "DUMP_RET_PCT": -0.03,
    # Copy mode — enable with PMF_RUN_MODE=copy on deploy (crowd unchanged).
    "RUN_MODE": "crowd",
    "COPY_TOP_N": COPY_TOP_N,
    "COPY_REQUIRE_FULL_WATCHLIST": COPY_REQUIRE_FULL_WATCHLIST,
    "COPY_CANDIDATE_SCAN": COPY_CANDIDATE_SCAN,
    "COPY_FILL_FETCH_MAX": COPY_FILL_FETCH_MAX,
    "COPY_MAX_ROI": COPY_MAX_ROI,
    "COPY_MIN_EQUITY": COPY_MIN_EQUITY,
    "COPY_REFRESH_HOURS": COPY_REFRESH_HOURS,
    "COPY_INCOMPLETE_RETRY_MIN": COPY_INCOMPLETE_RETRY_MIN,
    "COPY_RESELECT_HOURS": COPY_RESELECT_HOURS,
    "COPY_LOOKBACK_DAYS": COPY_LOOKBACK_DAYS,
    "COPY_HISTORY_DAYS": COPY_HISTORY_DAYS,
    "COPY_IDEAL_TRADES_PER_HOUR": COPY_IDEAL_TRADES_PER_HOUR,
    "COPY_MIN_MEDIAN_GAP_S": COPY_MIN_MEDIAN_GAP_S,
    "COPY_MAX_MEDIAN_GAP_S": COPY_MAX_MEDIAN_GAP_S,
    "COPY_MIN_FILLS": COPY_MIN_FILLS,
    "COPY_MAX_FILLS": COPY_MAX_FILLS,
    "COPY_MIN_FILLS_PER_DAY": COPY_MIN_FILLS_PER_DAY,
    "COPY_MAX_FILLS_PER_DAY": COPY_MAX_FILLS_PER_DAY,
    "COPY_MIN_WIN_RATE": COPY_MIN_WIN_RATE,
    "COPY_MIN_HIST_WIN_RATE": COPY_MIN_HIST_WIN_RATE,
    "COPY_MIN_RECENT_PNL": COPY_MIN_RECENT_PNL,
    "COPY_MIN_HIST_PNL": COPY_MIN_HIST_PNL,
    "COPY_MIN_PROFIT_FACTOR": COPY_MIN_PROFIT_FACTOR,
    "COPY_MAX_FAST_FLIP_RATIO": COPY_MAX_FAST_FLIP_RATIO,
    "COPY_MIN_HOLD_S": COPY_MIN_HOLD_S,
    "COPY_MAX_POSITIONS": COPY_MAX_POSITIONS,
    "COPY_MIN_FRESH_LEADERS_PCT": COPY_MIN_FRESH_LEADERS_PCT,
    "COPY_IDEAL_GAP_S": COPY_IDEAL_GAP_S,
    "COPY_REBALANCE_COOLDOWN_S": COPY_REBALANCE_COOLDOWN_S,
    "COPY_FILL_SLEEP_S": COPY_FILL_SLEEP_S,
}

# Gather-only: crowd books first (live-parity cadence), then marks, then candles.
# Consensus knobs match cloud so logged trade ticks / offline replay are not
# distorted vs Railway — still RESEARCH_ONLY (no orders).
_GATHER: dict = {
    "PAPER_TRADING": True,
    "RESEARCH_ONLY": True,
    "RESEARCH_DATA_ENABLED": True,
    "RANK_WINDOW": "week",
    "BASKET_FILTER_MODE": "off",
    "BASKET_SIZE": TRADE_BASKET_SIZE,
    "CANDIDATE_POOL": TRADE_BASKET_SIZE,
    "RESEARCH_POOL_SIZE": RESEARCH_GATHER_SIZE,
    "HOLDER_SCAN_POOL": 400,
    "BASKET_REFRESH_HOURS": 12.0,
    "LEADERBOARD_CACHE_HOURS": 6.0,
    # Research gathers RESEARCH_GATHER_SIZE wallets; trade/backtest use TRADE_BASKET_SIZE.
    # Long stale window so a full gather lap can finish before books drop wallets.
    "RESEARCH_WALLETS_PER_TICK": 25,
    "RESEARCH_SNAPSHOT_INTERVAL_S": 8.0,
    "RESEARCH_RECORD_INTERVAL_S": 60.0,
    "RESEARCH_MARKS_INTERVAL_S": 30.0,
    # Write books once we have this many wallet snapshots (no % gate that blocks forever).
    "RESEARCH_MIN_FRESH_WALLETS": 20,
    "RESEARCH_MIN_COVERAGE": 0.0,
    # Candles AFTER books — never block crowd samples.
    "RESEARCH_CANDLE_INTERVALS": ("1m", "15m", "1h"),
    "RESEARCH_CANDLE_BARS": 300,
    "RESEARCH_CANDLES_PER_TICK": 1,
    "RESEARCH_CANDLE_COOLDOWN_S": 2.0,
    # Must exceed one full research lap (~15–25 min on a home PC).
    "STALE_SNAPSHOT_S": 1500.0,
    "LOOP_SLEEP_S": 2.0,
    "MAX_BOOK_CHANGES_PER_HOUR": 0,
    # Cloud trade consensus (logged trade ticks only; no live orders).
    "FLOW_EMA_ALPHA": 0.20,
    "OPEN_CONFIRM_S": 180.0,
    "EXIT_RAW_FLOW": -0.024,
    "EXIT_AGREEMENT_GIVEBACK": 0.32,
    "STICKY_BOOK_SLOTS": True,
    "REBALANCE_COOLDOWN_S": 240.0,
}

LOCAL: dict = {
    "INSTANCE_NAME": "local",
    **_GATHER,
}

CLOUD: dict = {
    "INSTANCE_NAME": "cloud",
    **_TRADE,
}

RESEARCH: dict = {
    "INSTANCE_NAME": "research",
    **_GATHER,
}

PROFILES: dict[str, dict] = {
    "local": LOCAL,
    "cloud": CLOUD,
    "research": RESEARCH,
}
