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
    trade_store.clear(client.coin)
    return False


def cleanup_closed_coin(
    client: HyperliquidClient,
    trade_store: TradeStateStore,
    coin: str,
) -> None:
    """One coin went flat: cancel its leftover orders, drop local trade state."""
    client.cancel_all_orders_for_coin_named(coin)
    trade_store.clear(coin)


def cleanup_when_flat(
    client: HyperliquidClient,
    trade_store: TradeStateStore,
    *,
    extra_coins: list[str] | None = None,
) -> None:
    """No positions left: cancel stray orders so nothing tangles on next entry."""
    coins: list[str] = list(trade_store.coins())
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
    coin: str | None = None,
    coin_names: frozenset[str] | None = None,
    poll_seconds: float = 2.0,
    max_wait: float = 45.0,
) -> bool:
    """
    Wait until `coin` is flat (or the whole account if coin is None), then
    cancel leftover orders for that coin only.
    """
    names = coin_names or (frozenset({coin}) if coin else None)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        ok, positions = client.fetch_open_positions(force=True)
        if not ok:
            time.sleep(poll_seconds)
            continue
        if names is not None:
            still = any(pcoin in names for pcoin, _ in positions)
            if not still:
                cleanup_closed_coin(client, trade_store, coin or next(iter(names)))
                logger.info("Flat on %s — orders canceled", coin or next(iter(names)))
                return True
        elif not positions:
            cleanup_when_flat(client, trade_store)
            logger.info("Flat — all orders canceled")
            return True
        time.sleep(poll_seconds)
    return False
