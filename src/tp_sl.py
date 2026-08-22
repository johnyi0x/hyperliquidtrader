"""TP/SL trigger prices from entry (spot % move, not leverage)."""

from __future__ import annotations

from .pricing import round_price


def tp_sl_from_entry(
    side: str,
    entry: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    sz_decimals: int,
) -> tuple[float, float]:
    tp_m = take_profit_pct / 100.0
    sl_m = stop_loss_pct / 100.0
    if side == "long":
        tp = round_price(entry * (1 + tp_m), sz_decimals)
        sl = round_price(entry * (1 - sl_m), sz_decimals)
    else:
        tp = round_price(entry * (1 - tp_m), sz_decimals)
        sl = round_price(entry * (1 + sl_m), sz_decimals)
    return tp, sl
