"""
Per-instance overrides for running two bots (local PC vs cloud) on separate wallets.

Set PMF_PROFILE=local or PMF_PROFILE=cloud (default: local).
Each profile uses its own data folder via run.py (data-local / data-cloud).

Local  = holder-filtered basket (you can watch it; fewer tape wallets).
Cloud  = raw top-50 ROI board (run-and-forget; EMA averages scalpers).
Enter % / size / leverage stay in config.py — not profile-specific.
"""

from __future__ import annotations

LOCAL: dict = {
    "INSTANCE_NAME": "local",
    # Sitters / slow traders only. Fill scan at basket build, not the live loop.
    "BASKET_FILTER_MODE": "holder",
    # Mute a listed wallet that starts flipping after the list was built.
    "MAX_BOOK_CHANGES_PER_HOUR": 6,
    # PC: a bit more responsive so you can see crowd shifts.
    "FLOW_EMA_ALPHA": 0.26,
    "OPEN_CONFIRM_S": 120.0,
    "EXIT_RAW_FLOW": -0.018,
    "EXIT_AGREEMENT_GIVEBACK": 0.22,
    "REBALANCE_COOLDOWN_S": 120.0,
}

CLOUD: dict = {
    "INSTANCE_NAME": "cloud",
    # Full 7d ROI board. Scalpers stay in; we trade the average, not their flips.
    "BASKET_FILTER_MODE": "off",
    "MAX_BOOK_CHANGES_PER_HOUR": 0,
    # Railway: stickier holds, longer confirm before a new name, slower re-opens.
    "FLOW_EMA_ALPHA": 0.22,
    "OPEN_CONFIRM_S": 150.0,
    "EXIT_RAW_FLOW": -0.022,
    "EXIT_AGREEMENT_GIVEBACK": 0.28,
    "REBALANCE_COOLDOWN_S": 180.0,
}

PROFILES: dict[str, dict] = {
    "local": LOCAL,
    "cloud": CLOUD,
}
