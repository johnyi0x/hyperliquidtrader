"""
Per-instance overrides for running two bots (local PC vs cloud) on separate wallets.

Set PMF_PROFILE=local or PMF_PROFILE=cloud (default: local).
Each profile uses its own data folder via run.py (data-local / data-cloud).

Both: holder filter, sticky holds, top 100 by 7d ROI (local settings applied to cloud).
Local only: RESEARCH_DATA_ENABLED for offline backtest logs.
"""

from __future__ import annotations

_STICKY: dict = {
    "BASKET_FILTER_MODE": "holder",
    "MAX_BOOK_CHANGES_PER_HOUR": 6,
    "FLOW_EMA_ALPHA": 0.20,
    "OPEN_CONFIRM_S": 180.0,
    "EXIT_RAW_FLOW": -0.024,
    "EXIT_AGREEMENT_GIVEBACK": 0.32,
    "REBALANCE_COOLDOWN_S": 240.0,
    "STICKY_BOOK_SLOTS": True,
    "RANK_WINDOW": "week",
    "BASKET_SIZE": 100,
    "CANDIDATE_POOL": 100,
}

LOCAL: dict = {
    "INSTANCE_NAME": "local",
    **_STICKY,
    # Full books + marks for later filter on/off PnL backtests. Cloud stays off.
    "RESEARCH_DATA_ENABLED": True,
}

CLOUD: dict = {
    "INSTANCE_NAME": "cloud",
    **_STICKY,
    "RESEARCH_DATA_ENABLED": False,
}

PROFILES: dict[str, dict] = {
    "local": LOCAL,
    "cloud": CLOUD,
}
