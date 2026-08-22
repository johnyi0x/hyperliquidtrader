"""
Per-instance overrides for running two bots (local PC vs cloud) on separate wallets.

Set PMF_PROFILE=local or PMF_PROFILE=cloud (default: local).
Each profile uses its own data folder via run.py (data-local / data-cloud).

Cloud = previous local trial (raw top-50, sticky slots, stickier holds, 7d ROI).
Local = same as cloud except 24h ROI rank instead of 7d.
Enter % / size / leverage stay in config.py — not profile-specific.
"""

from __future__ import annotations

# Shared "improved cloud" knobs. Rank window is the only local vs cloud split.
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
    # Hyperliquid leaderboard 24h ROI column.
    "RANK_WINDOW": "day",
}

CLOUD: dict = {
    "INSTANCE_NAME": "cloud",
    **_STICKY,
    # Hyperliquid leaderboard 7d ROI column.
    "RANK_WINDOW": "week",
}

PROFILES: dict[str, dict] = {
    "local": LOCAL,
    "cloud": CLOUD,
}
