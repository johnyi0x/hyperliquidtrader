"""
Profit-meta follower — follow a basket of high-PnL Hyperliquid wallets.

This bot does not use candles. It treats top-PnL wallets as one liquid basket:
  snapshot everyone's current book (scalpers included),
  average it, smooth it, hold the crowded side,
  exit when that net flow fades — not when one wallet flips.


Edit this file. Run:  python profit-meta-follower/run.py
"""

from __future__ import annotations

# =============================================================================
# LIVE / PAPER
# =============================================================================
PAPER_TRADING = False
PAPER_START_BALANCE = 1000.0
USE_TESTNET = False
USE_CROSS_MARGIN = True
USE_MARKET_ORDERS = True
MARKET_ORDER_SLIPPAGE = 0.015  # 1.5% — thin HIP-3 needs more; keep tighter than 5%
MIN_ORDER_NOTIONAL_USD = 10.0

# Do not run this at the same time as bot_live.py on this wallet
# (two bots will fight the same positions). Same wallet is fine if THIS
# is the only script running.

# =============================================================================
# WALLET BASKET
# =============================================================================
# How many qualified wallets to keep in the live basket.
# Snapshot cost is weight 2. IP budget is 1200/min (~950 usable). 50 wallets
# every ~15s is ~400 weight/min — under the cap. More than ~50 and books go
# stale before we finish a lap (each REST snapshot is ~1–6s on a real link).
BASKET_SIZE = 50
# Score a larger pool first, then keep BASKET_SIZE. With filters off, set equal.
CANDIDATE_POOL = 50
BASKET_REFRESH_HOURS = 12.0
# Reuse a cached leaderboard dump this long (avoids re-downloading 15k+ rows).
LEADERBOARD_CACHE_HOURS = 6.0

# Rank like the HL Leaderboard tab ROI column.
# "day" = 24h ROI, "week" = 7d ROI. (month is ignored for ranking.)
# Profiles override: local and cloud both use 24h ROI.
RANK_WINDOW = "day"
# Optional second window that must not be a large loss ("" = off).
CONFIRM_WINDOW = ""

# Ranking filters. Applied only when BASKET_FILTER_MODE = "holder".
# "off"    = raw 7d ROI board, no fill check, fill BASKET_SIZE (cloud default).
# "holder" = walk the ROI board, fetch recent fills ONCE per wallet at list
#            build (not the live loop). Keep sitters / slow traders, skip tape.
#            $0 week volume with no fills = holder. List can be shorter than 50.
# Profiles override this: local and cloud both "off".
BASKET_FILTER_MODE = "off"
MIN_ACCOUNT_VALUE = 0.0
MIN_WINDOW_PNL = 0.0
MIN_CONFIRM_PNL = 0.0
MIN_WINDOW_VOLUME = 0.0
MAX_VOLUME_TO_EQUITY = 0.0
MAX_VOLUME_TO_PROFIT = 0.0
MAX_WINDOW_ROI = 0.0
MIN_WINDOW_ROI = 0.0
RANK_TILT_PNL = False
# How far down the ROI board to look for holders (fill calls, weight ~20 each).
HOLDER_SCAN_POOL = 400
# Fill-tape rules (7d). 0 fills = sitting holder.
HOLD_LOOKBACK_DAYS = 7.0
HOLD_MAX_FILLS = 18
HOLD_MAX_FILLS_PER_DAY = 5.0
HOLD_MIN_MEDIAN_GAP_S = 3600.0
# Live backup: mute a listed wallet if its book flips this many times / hour. 0 = off.
MAX_BOOK_CHANGES_PER_HOUR = 8

# Extra /info calls (ledger, fills, vault role) on the shortlist. False = rank from
# the leaderboard dump only, then snapshot positions. That is the cheap path.
DEEP_AUDIT = False
MAX_DEPOSIT_TO_EQUITY = 0.35
AUDIT_FILLS = False
FILL_PNL_MISMATCH_RATIO = 0.25
SKIP_VAULTS = False

# =============================================================================
# DEX / PAIR SCOPE
# =============================================================================
# "native"   = main Hyperliquid perps only
# "include"  = native + HIP-3 (xyz:...)
# "xyz_only" = HIP-3 only
DEX_SCOPE = "include"
# Extra allow/deny (empty = no extra filter). Tickers as API names, e.g. "BTC", "xyz:SNDK".
ALLOW_COINS: tuple[str, ...] = ()
DENY_COINS: tuple[str, ...] = ()
# Ride the crowded majors. Empty = every in-scope coin is equal.
PREFERRED_COINS: tuple[str, ...] = ()
# Used only when PREFERRED_COINS is set. Fraction of BASKET_SIZE.
MIN_WALLETS_OTHER_PCT = 0.12
MIN_WALLETS_OTHER = 0

# =============================================================================
# SNAPSHOT / API BUDGET
# =============================================================================
# Position snapshots are cheap (weight 2). Fills are NOT used on the live loop.
SNAPSHOT_INTERVAL_S = 12.0
# How many wallets to refresh per loop tick (stagger = smooth IP weight).
WALLETS_PER_TICK = 12
# Need this fraction of BASKET_SIZE with a fresh snapshot before we trade.
MIN_LIVE_VOTERS_PCT = 0.45
MIN_LIVE_VOTERS = 0
# A full 50-wallet lap takes ~4–5 min (12 REST snapshots/tick). Keep books that
# are still on that lap instead of dropping ~10 voters and making agreement noisy.
STALE_SNAPSHOT_S = 480.0
LOOP_SLEEP_S = 3.0

# =============================================================================
# BASKET FLOW (scalpers stay in — we trade the AVERAGE, not their flips)
# =============================================================================
# Each wallet votes with signed conviction = position_notional / that_wallet_equity.
# A scalper long BTC for 40s still counts as inventory while they are long.
# One wallet flipping does nothing; many flipping the same way moves the average.
MIN_WALLET_CONVICTION = 0.015
# Fractions of BASKET_SIZE (not a hardcoded headcount). 0.10 of 50 = 5 wallets.
MIN_WALLETS_ON_COIN_PCT = 0.08
MIN_WALLETS_ON_COIN = 0
MIN_SIDE_AGREEMENT = 0.10
# Hold until the crowd falls to this fraction of the list (hysteresis vs enter).
EXIT_SIDE_AGREEMENT = 0.05
MIN_AVG_CONVICTION = 0.022  # applied to the SMOOTHED basket, not one noisy snapshot
MAX_COINS_IN_BOOK = 3
# If True, a held coin keeps its slot until a real exit; a 4th name cannot kick it.
# Profiles turn this on for both local and cloud.
STICKY_BOOK_SLOTS = False

# Smooth the basket so individual scalp noise cancels.
# Mid setting: old was 0.30 (jumpy), smoothened was 0.18 (very sticky). Target ~30m–3h holds.
FLOW_EMA_ALPHA = 0.26
# Exit when smoothed inventory has given back this fraction of its peak (0.30 = 30%).
# That is "flow is gone" without waiting until the book is flat (too late).
CONV_GIVEBACK = 0.30
# 0 = do not require the pile to be growing every tick. Tiny negative EMA noise is ignored.
MIN_ENTRY_FLOW = 0.0
# Mid exit: old -0.008 scalped; smoothened -0.015 sat 12h+. Aim ~30m–3h when pile fades.
EXIT_FLOW = -0.011
# Fast lane: unsmoothed raw basket tick (wallets reducing size NOW). 0 = off.
EXIT_RAW_FLOW = -0.020
# Exit if wallet-count agreement fell this fraction from its peak while holding. 0 = off.
EXIT_AGREEMENT_GIVEBACK = 0.25
EXIT_AVG_CONVICTION = 0.020
OPEN_CONFIRM_S = 120.0
# Mute only blow-ups, not scalpers: live equity vs leaderboard snapshot.
MAX_LIVE_EQUITY_DROP = 0.40

# Skip illiquid / hostile-funding / blown-out basis names.
MIN_COIN_DAY_VOLUME = 250_000.0
# Hourly funding. Positive = longs pay shorts. Skip if WE would be the paying side
# above this (0.0004 ≈ 0.04%/hour ≈ 1%/day).
MAX_HOSTILE_FUNDING = 0.0004
# |mark-oracle| / oracle. Copying into a 2% dislocation is usually late.
MAX_BASIS_ABS = 0.012
# Top-of-book spread. Market orders through a wide HIP-3 spread are a donation.
MAX_SPREAD_PCT = 0.004  # 0.40%
MARKET_CACHE_S = 60.0


# =============================================================================
# YOUR SIZE / LEVERAGE
# =============================================================================
# "fixed"      = use OUR_GROSS_MARGIN_PCT (split across coins). Current default.
# "wallet_avg" = each coin uses the equal-wallet average of how much of THEIR
#                equity those wallets posted as margin on that pair
#                (30% @10x and 50% @10x → you use 40% margin). Wallets with
#                no position on that pair are not in the average.
SIZE_MODE = "fixed"
# "follow"  = trade the same side as the wallet basket (current).
# "reverse" = perfectly invert: basket long → we short, basket short → we long.
#             Same coin, same size, same leverage, opposite side. Exits invert too.
TRADE_MODE = "follow"
# "auto" = median leverage in fixed mode, mean leverage in wallet_avg mode.
# "median" / "mean" force one or the other. Always clamped to OUR_MIN/MAX.
LEVERAGE_MODE = "mean"
# Hard cap on copied margin % (0 = no cap). Safety so a 200% average cannot
# request more than your full equity as margin.
COPY_MARGIN_CAP_PCT = 100.0
# Total margin budget across ALL copied positions (percent of equity).
# 90% = up to 3 coins × 30% margin each. Only used when SIZE_MODE = "fixed".
OUR_GROSS_MARGIN_PCT = 90.0
# Per-coin cap as percent of OUR_GROSS_MARGIN_PCT. 33.33% of 90% = 30% of equity per pair.
MAX_MARGIN_PER_COIN_PCT = 33.33
# Leverage we use: clamp(wallet median or mean, these bounds, exchange max).
OUR_MIN_LEVERAGE = 2
OUR_MAX_LEVERAGE = 20
# Scale our per-coin size by the group's average conviction (already %). This is
# how "they trade tiny vs they go heavy" maps onto your balance.
# When only one coin qualifies, still use 30% margin (1/3 of gross budget), not a bigger single-name size.
SINGLE_NAME_SIZE_MULT = 0.3333
# =============================================================================
# REBALANCE SAFETY
# =============================================================================
# Ignore noise: same side, notional within this band of target → do nothing.
REBALANCE_DRIFT_PCT = 35.0
# Blocks 10–20 min close/reopen loops; does not force multi-hour holds by itself.
REBALANCE_COOLDOWN_S = 180.0
MAX_ACTIONS_PER_CYCLE = 3
# Flatten a managed coin if it leaves the consensus book.
FLATTEN_WHEN_DROPPED = True
# Only touch coins this bot opened (plus new targets). Never close unrelated positions.
MANAGED_ONLY = True

# =============================================================================
# TELEMETRY / BACKTEST
# =============================================================================
# Daily JSONL under data/telemetry/YYYY-MM-DD/ for offline param tuning.
TELEMETRY_ENABLED = True
# Set by config_profiles (local | cloud). Override with PMF_PROFILE env.
INSTANCE_NAME = "local"

# After a reconnect of hours: we do NOT replay missed fills. We snapshot wallets
# as they are NOW, compare to OUR exchange positions NOW, and rebalance with the
# same deadband. That is the crash/power-loss model.

from pmf.config_loader import apply_profile  # noqa: E402

apply_profile(globals())
