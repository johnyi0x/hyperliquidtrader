"""
SMOOTHENED snapshot (saved 2026-08-20) — hour+/12h-style holds, majors-heavy.
Copy these values into config.py to restore this behavior.

This was the live config after anti-scalp smoothening + 30% margin per coin.
"""

# --- Holder filter ---
HOLD_MAX_FILLS = 12
HOLD_MAX_FILLS_PER_DAY = 3.0
HOLD_MIN_MEDIAN_GAP_S = 7200.0
MAX_BOOK_CHANGES_PER_HOUR = 4

# --- Basket flow / enter-exit ---
MIN_WALLETS_ON_COIN_PCT = 0.08
MIN_SIDE_AGREEMENT = 0.10
EXIT_SIDE_AGREEMENT = 0.05
FLOW_EMA_ALPHA = 0.18
CONV_GIVEBACK = 0.35
EXIT_FLOW = -0.015
OPEN_CONFIRM_S = 180.0

# --- Rebalance ---
REBALANCE_DRIFT_PCT = 50.0
REBALANCE_COOLDOWN_S = 300.0

# --- Size (30% margin per coin, up to 3) ---
OUR_GROSS_MARGIN_PCT = 90.0
MAX_MARGIN_PER_COIN_PCT = 33.33
SINGLE_NAME_SIZE_MULT = 0.3333
