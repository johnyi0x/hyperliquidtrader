"""Price/size rounding and maker (ALO-safe) limit prices."""

from __future__ import annotations

import math

MAX_DECIMALS_PERP = 6


def size_step(sz_decimals: int) -> float:
    return 10.0 ** (-int(sz_decimals))


def round_size(sz: float, sz_decimals: int) -> float:
    return round(sz, sz_decimals)


def floor_size(sz: float, sz_decimals: int) -> float:
    step = size_step(sz_decimals)
    if step <= 0:
        return sz
    return math.floor(sz / step) * step


def ceil_size(sz: float, sz_decimals: int) -> float:
    step = size_step(sz_decimals)
    if step <= 0:
        return sz
    return math.ceil(sz / step) * step


def round_price(px: float, sz_decimals: int) -> float:
    if px > 100_000:
        return float(round(px))
    return round(float(f"{px:.5g}"), MAX_DECIMALS_PERP - sz_decimals)


def infer_tick(l2: dict, sz_decimals: int) -> float:
    bids = l2["levels"][0]
    asks = l2["levels"][1]
    if len(bids) >= 2:
        return abs(float(bids[0]["px"]) - float(bids[1]["px"]))
    if len(asks) >= 2:
        return abs(float(asks[0]["px"]) - float(asks[1]["px"]))
    px = float(bids[0]["px"]) if bids else float(asks[0]["px"])
    if px > 100_000:
        return 1.0
    return 10 ** (-(MAX_DECIMALS_PERP - sz_decimals))


def maker_limit_price(
    l2: dict,
    is_buy: bool,
    sz_decimals: int,
    *,
    attempt_index: int = 0,
    passive_nudge: int = 0,
) -> float:
    """
    Post-only (ALO) safe price: join maker side, nudge toward spread on later attempts.
    passive_nudge increases after an immediate ALO cancel (more passive, still maker).
    """
    bids = l2["levels"][0]
    asks = l2["levels"][1]
    if not bids or not asks:
        raise ValueError("Empty L2 book")

    best_bid = float(bids[0]["px"])
    best_ask = float(asks[0]["px"])
    tick = infer_tick(l2, sz_decimals)

    if best_ask <= best_bid:
        raise ValueError("Invalid book: ask <= bid")

    if is_buy:
        # Buy: at or below bid side; must stay strictly below best ask for ALO
        max_px = best_ask - tick
        px = best_bid + attempt_index * tick - passive_nudge * tick
        px = min(px, max_px)
        px = max(px, best_bid - passive_nudge * tick)
    else:
        min_px = best_bid + tick
        px = best_ask - attempt_index * tick + passive_nudge * tick
        px = max(px, min_px)
        px = min(px, best_ask + passive_nudge * tick)

    return round_price(px, sz_decimals)


def mid_post_only_price(
    l2: dict,
    is_buy: bool,
    sz_decimals: int,
    *,
    passive_nudge: int = 0,
) -> float:
    """Mid price, clamped so a post-only (ALO) order cannot take."""
    bids = l2["levels"][0]
    asks = l2["levels"][1]
    if not bids or not asks:
        raise ValueError("Empty L2 book")
    best_bid = float(bids[0]["px"])
    best_ask = float(asks[0]["px"])
    tick = infer_tick(l2, sz_decimals)
    if best_ask <= best_bid:
        raise ValueError("Invalid book: ask <= bid")
    mid = round_price((best_bid + best_ask) / 2.0, sz_decimals)
    if is_buy:
        max_px = best_ask - tick
        px = mid - passive_nudge * tick
        px = min(px, max_px)
        px = max(px, best_bid - passive_nudge * tick)
    else:
        min_px = best_bid + tick
        px = mid + passive_nudge * tick
        px = max(px, min_px)
        px = min(px, best_ask + passive_nudge * tick)
    return round_price(px, sz_decimals)


def limit_would_take(l2: dict, is_buy: bool, px: float) -> bool:
    """True if a limit at px would cross the spread (taker)."""
    bids = l2["levels"][0]
    asks = l2["levels"][1]
    if not bids or not asks:
        return True
    best_bid = float(bids[0]["px"])
    best_ask = float(asks[0]["px"])
    if is_buy:
        return px + 1e-12 >= best_ask
    return px - 1e-12 <= best_bid
