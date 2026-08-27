"""
Per-instance overrides: local | cloud | research.

Set PMF_PROFILE=local|cloud|research (default: local).
Data folders: data-local / data-cloud / data-research.

cloud  = live trade only (do not change while running).
local  = gather-only (identical to research; data-local/).
research = gather-only (data-research/).
"""

from __future__ import annotations

# Live trading — cloud only. Leave alone while Railway is live.
# Cloud _TRADE last tuned: 2026-08-27 03:50 UTC strategy=crowd_dump_all score=24.321782809242592 ret=41.45447486243077%
_TRADE: dict = {
    "BASKET_FILTER_MODE": 'off',
    "MAX_BOOK_CHANGES_PER_HOUR": 6,
    "FLOW_EMA_ALPHA": 0.36,
    "OPEN_CONFIRM_S": 330.0,
    "EXIT_RAW_FLOW": -0.024,
    "EXIT_AGREEMENT_GIVEBACK": 0.34,
    "REBALANCE_COOLDOWN_S": 480.0,
    "STICKY_BOOK_SLOTS": True,
    "RANK_WINDOW": "week",
    "BASKET_SIZE": 200,
    "CANDIDATE_POOL": 200,
    "RESEARCH_DATA_ENABLED": False,
    "RESEARCH_ONLY": False,
    "BACKTEST_LIVE_STRATEGY": 'crowd_dump_all',
    "CONV_GIVEBACK": 0.36,
    "EXIT_AVG_CONVICTION": 0.024,
    "EXIT_FLOW": -0.015,
    "EXIT_SIDE_AGREEMENT": 0.04,
    "LIVE_CANDLE_SEED": True,
    "MIN_AVG_CONVICTION": 0.028,
    "MIN_ENTRY_FLOW": 0.001,
    "MIN_SIDE_AGREEMENT": 0.06,
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
    "BASKET_SIZE": 200,
    "CANDIDATE_POOL": 200,
    "RESEARCH_POOL_SIZE": 200,
    "HOLDER_SCAN_POOL": 200,
    "BASKET_REFRESH_HOURS": 12.0,
    "LEADERBOARD_CACHE_HOURS": 6.0,
    # 200 wallets × slow HL REST (~2 min/tick with market cache) needs a long stale
    # window or coverage never climbs (8 min stale capped you at ~40% forever).
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
