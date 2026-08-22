"""
Snapshot of profit-meta-follower config BEFORE stable-hold tuning (2026-08-19).
Restore by copying these values back into config.py if you want the old behavior.
Only the parameters that changed are listed here; everything else was identical.
"""

# --- Holder filter (list build) ---
HOLD_MAX_FILLS = 18
HOLD_MAX_FILLS_PER_DAY = 5.0
HOLD_MIN_MEDIAN_GAP_S = 3600.0
MAX_BOOK_CHANGES_PER_HOUR = 8

# --- Basket flow / enter-exit ---
MIN_WALLETS_ON_COIN_PCT = 0.08
MIN_SIDE_AGREEMENT = 0.10
EXIT_SIDE_AGREEMENT = 0.05
FLOW_EMA_ALPHA = 0.30
CONV_GIVEBACK = 0.30
EXIT_FLOW = -0.008
OPEN_CONFIRM_S = 120.0

# --- Rebalance ---
REBALANCE_DRIFT_PCT = 25.0
REBALANCE_COOLDOWN_S = 30.0

# --- Size (fixed mode) ---
OUR_GROSS_MARGIN_PCT = 95.0
MAX_MARGIN_PER_COIN_PCT = 50.0
SINGLE_NAME_SIZE_MULT = 0.55
