"""
Ensure every open position has exchange TP+SL; flatten and cancel if not.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .exchange_client import HyperliquidClient, Position
    from .order_executor import OrderExecutor
    from .trade_state import TradeStateStore


def ensure_protected_position(
    client: HyperliquidClient,
    executor: OrderExecutor,
    position: Position,
    trade_store: TradeStateStore,
    *,
    take_profit_pct: float,
    stop_loss_pct: float,
    logger: logging.Logger,
    tpsl_attach_retries: int = 3,
) -> bool:
    """
    Return True when position is protected by exchange TP+SL.
    On failure: emergency market flatten + cancel all orders (no naked leverage).
    """
    if client.has_exchange_tpsl():
        if client.has_open_entry_orders():
            client.cancel_entry_orders_for_coin()
        entry = position.entry_price or client.get_mark_price()
        trade_store.sync_from_exchange(
            client.coin,
            position.side,
            entry,
            position.size,
            take_profit_pct,
            stop_loss_pct,
        )
        return True

    logger.warning(
        "Unprotected %s size=%s — attaching exchange TP/SL (market triggers)",
        position.side,
        position.size,
    )
    if client.attach_position_tpsl(
        position,
        take_profit_pct,
        stop_loss_pct,
        max_attempts=tpsl_attach_retries,
    ):
        entry = position.entry_price or client.get_mark_price()
        trade_store.sync_from_exchange(
            client.coin,
            position.side,
            entry,
            position.size,
            take_profit_pct,
            stop_loss_pct,
        )
        logger.info("Position now protected with exchange TP/SL")
        client.cancel_entry_orders_for_coin()
        return True

    logger.error(
        "Could not attach TP/SL after %s tries — emergency flatten (no naked position)",
        tpsl_attach_retries,
    )
    executor.emergency_flatten("unprotected_position")
    trade_store.clear()
    return False


def cleanup_when_flat(
    client: HyperliquidClient,
    trade_store: TradeStateStore,
    *,
    extra_coins: list[str] | None = None,
) -> None:
    """No position: cancel stray orders so nothing tangles on next entry."""
    coins: list[str] = []
    if trade_store.trade and trade_store.trade.coin:
        coins.append(trade_store.trade.coin)
    if client.coin not in coins:
        coins.append(client.coin)
    if extra_coins:
        for coin in extra_coins:
            if coin not in coins:
                coins.append(coin)
    for coin in coins:
        client.cancel_all_orders_for_coin_named(coin)
    client.sweep_orphan_orders()
    trade_store.clear()


def wait_until_flat(
    client: HyperliquidClient,
    trade_store: TradeStateStore,
    logger: logging.Logger,
    *,
    poll_seconds: float = 2.0,
    max_wait: float = 45.0,
) -> bool:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if not client.has_any_open_position(force=True):
            cleanup_when_flat(client, trade_store)
            logger.info("Flat — all orders canceled")
            return True
        time.sleep(poll_seconds)
    return False
