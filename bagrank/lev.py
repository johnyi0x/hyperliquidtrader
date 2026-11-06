"""Per-coin size leverage from exchange maxLev.

Live-usable (use_lev=0) size is floor(maxLev / lev_x) as a whole number.
If that is below 1 (maxLev < lev_x), size is 1x. Default lev_x=3 →
maxLev 10 → 3x, 20 → 6x, 50 → 16x, a 3x-max coin → 1x.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("bagrank.lev")

DEFAULT_LEV_X = 3.0
DEFAULT_MAX_LEV = 50.0
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
# Edit these two numbers when you want a different size.
# TRADE_MARGIN_USD is the dollars of margin posted on every trade.
# It does not grow after a win. Notional = this * leverage.
# Example: 100 at 6x is a $600 position, every time.
TRADE_MARGIN_USD = 100.0
# Starting cash for a backtest or paper run. Raised to the trade margin if smaller,
# so the first trade always fits. Raise it to leave room for a string of losses.
BACKTEST_EQUITY = 10000.0
# Live orders still leave 5% of free collateral unused so the exchange does not reject the size.
ORDER_MARGIN_BUFFER = 0.95
LIVE_GROSS_PCT = 95.0
MARGIN_FRAC = ORDER_MARGIN_BUFFER
# Soft-keep: top up when held size is below this fraction of target.
TOP_UP_RATIO = 0.90


def starting_equity(requested: float | None = None) -> float:
    """Backtest/paper cash. Never below the fixed trade margin."""
    if requested is None:
        eq = float(BACKTEST_EQUITY)
    else:
        eq = float(requested)
    floor = float(TRADE_MARGIN_USD)
    if eq < floor:
        return floor
    return eq


def clamp_gross_frac(gross_frac: float = 0.0) -> float:
    """Kept so older callers import. Size no longer uses a percent of balance."""
    _ = gross_frac
    return 1.0


def target_notional(
    capital: float,
    gross_frac: float,
    weight: float,
    lev: float,
) -> float:
    """Position notional = TRADE_MARGIN_USD * weight * leverage. Balance is ignored."""
    _ = (capital, gross_frac)
    w = float(weight or 0)
    lv = float(lev or 0)
    if w <= 0 or lv < 1.0:
        return 0.0
    return float(TRADE_MARGIN_USD) * w * lv


def pair_size_lev(max_lev: float, lev_x: float) -> float:
    """floor(maxLev / lev_x) as an integer; 1x if maxLev is below lev_x or lev_x is off."""
    x = float(lev_x or 0)
    if x <= 0:
        return 1.0
    mx = float(max_lev or 0)
    if mx < x:
        return 1.0
    n = int(mx / x)
    if n < 1:
        return 1.0
    return float(n)


def fetch_exchange_max_lev() -> dict[str, float]:
    """coin name → max leverage from HL meta (main + xyz). Empty on failure."""
    out: dict[str, float] = {}
    try:
        import requests
    except ImportError:
        return out
    from src.market_resolver import _max_leverage_from_asset

    for payload in ({"type": "meta"}, {"type": "meta", "dex": "xyz"}):
        try:
            resp = requests.post(HL_INFO_URL, json=payload, timeout=20)
            resp.raise_for_status()
            meta = resp.json()
        except Exception as exc:
            log.warning("HL meta %s failed: %s", payload.get("dex") or "main", exc)
            continue
        universe = meta.get("universe") if isinstance(meta, dict) else None
        if not isinstance(universe, list):
            continue
        dex = str(payload.get("dex") or "")
        for asset in universe:
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name") or "")
            if not name:
                continue
            if dex and ":" not in name:
                name = f"{dex}:{name}"
            out[name] = float(_max_leverage_from_asset(asset))
    if out:
        log.info("Exchange maxLev for %s coins", len(out))
    return out


def apply_exchange_max_lev(panel: Any, mapping: dict[str, float] | None = None) -> dict[str, float]:
    """Write per-coin maxLev onto the panel. Fetches HL if mapping is None."""
    if mapping is None:
        mapping = fetch_exchange_max_lev()
    n = int(getattr(panel, "n_coins", 0) or 0)
    arr = np.full(n, DEFAULT_MAX_LEV, dtype=np.float64)
    coins = list(getattr(panel, "coins", []) or [])
    for i, coin in enumerate(coins):
        if i >= n:
            break
        if coin in mapping:
            arr[i] = float(mapping[coin])
        elif coin.replace("xyz:", "") in mapping:
            arr[i] = float(mapping[coin.replace("xyz:", "")])
    panel.max_leverage = arr
    return mapping
