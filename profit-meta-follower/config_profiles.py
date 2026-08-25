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
# Cloud _TRADE last tuned: 2026-08-25 18:50 UTC strategy=swing_meta_all score=58.87490083936898 ret=68.17781770753466%
_TRADE: dict = {
    "BASKET_FILTER_MODE": 'off',
    "MAX_BOOK_CHANGES_PER_HOUR": 6,
    "FLOW_EMA_ALPHA": 0.2,
    "OPEN_CONFIRM_S": 210.0,
    "EXIT_RAW_FLOW": -0.02,
    "EXIT_AGREEMENT_GIVEBACK": 0.34,
    "REBALANCE_COOLDOWN_S": 300.0,
    "STICKY_BOOK_SLOTS": True,
    "RANK_WINDOW": "week",
    "BASKET_SIZE": 100,
    "CANDIDATE_POOL": 100,
    "RESEARCH_DATA_ENABLED": False,
    "RESEARCH_ONLY": False,
    "BACKTEST_LIVE_STRATEGY": 'swing_meta_all',
    "CONV_GIVEBACK": 0.34,
    "EXIT_AVG_CONVICTION": 0.016,
    "EXIT_FLOW": -0.011,
    "EXIT_SIDE_AGREEMENT": 0.06,
    "LIVE_CANDLE_SEED": True,
    "MIN_AVG_CONVICTION": 0.018,
    "MIN_ENTRY_FLOW": 0.0,
    "MIN_SIDE_AGREEMENT": 0.08,
    "LIVE_CANDLES_PER_TICK": 2,
    "LIVE_CANDLE_BARS_15M": 64,
    "LIVE_CANDLE_BARS_1H": 48,
    "LIVE_CANDLE_BARS_1M": 160,
    "LIVE_CANDLE_COOLDOWN_S": 8.0,
    "SWING_BAND_PCT": 0.02,
    "SWING_BREAK_PCT": 0.012,
    "SWING_ENTRY": 'ema_pullback',
    "SWING_EXIT_RSI": 0.0,
    "SWING_LOOKBACK_S": 600.0,
    "SWING_MAX_HOLD_S": 86400.0,
    "SWING_META_MODE": 'reverse',
    "SWING_REENTRY_S": 3600.0,
    "SWING_RSI_BUY": 40.0,
    "SWING_RSI_SELL": 70.0,
    "SWING_SL_PCT": 5.0,
    "SWING_TF": '1h',
    "SWING_TP_PCT": 5.0,
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
