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
