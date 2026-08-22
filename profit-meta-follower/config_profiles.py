"""
Per-instance overrides for running two bots (local PC vs cloud) on separate wallets.

Set PMF_PROFILE=local or PMF_PROFILE=cloud (default: local).
Each profile uses its own data folder via run.py (data-local / data-cloud).

Cloud and local share sticky holds / raw top-50; both rank the Hyperliquid 7d ROI board.
Enter % / size / leverage stay in config.py — not profile-specific.
"""

from __future__ import annotations

# Shared knobs. Local and cloud only differ by INSTANCE_NAME (data/logs/wallet).
_STICKY: dict = {
    "BASKET_FILTER_MODE": "off",
    "MAX_BOOK_CHANGES_PER_HOUR": 0,
    "FLOW_EMA_ALPHA": 0.20,
    "OPEN_CONFIRM_S": 180.0,
    "EXIT_RAW_FLOW": -0.024,
    "EXIT_AGREEMENT_GIVEBACK": 0.32,
    "REBALANCE_COOLDOWN_S": 240.0,
    "STICKY_BOOK_SLOTS": True,
}

LOCAL: dict = {
    "INSTANCE_NAME": "local",
    **_STICKY,
    # Hyperliquid leaderboard 7d ROI column.
    "RANK_WINDOW": "week",
}

CLOUD: dict = {
    "INSTANCE_NAME": "cloud",
    **_STICKY,
    # Hyperliquid leaderboard 7d ROI column (same as local).
    "RANK_WINDOW": "week",
}

PROFILES: dict[str, dict] = {
    "local": LOCAL,
    "cloud": CLOUD,
}
