"""
Per-instance overrides for running two bots (local PC vs cloud) on separate wallets.

Set PMF_PROFILE=local or PMF_PROFILE=cloud (default: local).
Each profile uses its own data folder via run.py (data-local / data-cloud).

Cloud: top 200, 7d ROI. Local: top 100, 7d ROI. Same sticky holds / filter off.
Enter % / size / leverage stay in config.py — not profile-specific.
"""

from __future__ import annotations

# Shared knobs. Local 100 wallets; cloud 200.
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
    "RANK_WINDOW": "week",
    "BASKET_SIZE": 100,
    "CANDIDATE_POOL": 100,
}

CLOUD: dict = {
    "INSTANCE_NAME": "cloud",
    **_STICKY,
    "RANK_WINDOW": "week",
    "BASKET_SIZE": 200,
    "CANDIDATE_POOL": 200,
}

PROFILES: dict[str, dict] = {
    "local": LOCAL,
    "cloud": CLOUD,
}
