"""Maker (ALO) limit execution; optional all-market entry/exit; market close on emergencies."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from .pricing import maker_limit_price, mid_post_only_price, round_size

if TYPE_CHECKING:
    from .exchange_client import HyperliquidClient, Position

# ALO reprices before one of the 5 outer waits (not counted as full attempts)
MAX_ALO_PLACE_RETRIES = 10


class OrderExecutor:
    def __init__(
        self,
        client: HyperliquidClient,
        wait_seconds: int,
        max_attempts: int,
        logger: logging.Logger,
        *,
        use_market_orders: bool = False,
        market_slippage: float = 0.05,
        mid_limit_then_market: bool = False,
        mid_limit_wait_seconds: float = 10.0,
        mid_limit_attempts: int = 3,
    ) -> None:
        self.client = client
        self.wait_seconds = wait_seconds
        self.max_attempts = max_attempts
        self.logger = logger
        self.use_market_orders = use_market_orders
        self.market_slippage = market_slippage
        self.mid_limit_then_market = bool(mid_limit_then_market)
        self.mid_limit_wait_seconds = max(1.0, float(mid_limit_wait_seconds))
        self.mid_limit_attempts = max(1, int(mid_limit_attempts))
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def execute_open(self, is_buy: bool, target_sz: float) -> bool:
        with self._lock:
            if self.use_market_orders:
                ok = self._market_open_unlocked(is_buy, target_sz)
            else:
                ok = self._fill_with_limits(
                    is_buy=is_buy,
                    target_sz=target_sz,
                    reduce_only=False,
                )
            if not ok:
                self.client.cancel_entry_orders_for_coin()
                pos = self.client.get_position(force=True)
                if pos is not None:
                    self._emergency_flatten_unlocked("open_aborted")
            return ok

    def _emergency_flatten_unlocked(self, reason: str) -> bool:
        """Market-close position and cancel all orders (caller must hold _lock if needed)."""
        self.logger.warning("Emergency flatten: %s", reason)
        self.client.cancel_all_orders_for_coin()
        pos = self.client.get_position(force=True)
        if pos is None:
            self.client.cancel_all_orders_for_coin()
            return True
        try:
            self.client.place_market_close(sz=pos.size)
        except Exception as exc:
            self.logger.error("Market close failed: %s", exc)
            return False
        time.sleep(1.0)
        self.client.cancel_all_orders_for_coin()
        flat = self.client.get_position(force=True) is None
        if not flat:
            self.logger.error("Emergency flatten incomplete — position still open")
        return flat

    def emergency_flatten(self, reason: str) -> bool:
        with self._lock:
            return self._emergency_flatten_unlocked(reason)

    def execute_protected_entry(
        self,
        is_buy: bool,
        target_sz: float,
        take_profit_pct: float,
        stop_loss_pct: float,
    ) -> bool:
        """
        1) ALO entry only until full size filled (partials keep accumulating, no TP/SL yet).
        2) positionTpsl on full position from exchange entryPx.
        3) If TP/SL cannot be verified, flatten — never leave naked leverage.
        """
        with self._lock:
            if self.mid_limit_then_market:
                filled = self._fill_mid_limit_then_market(
                    is_buy=is_buy,
                    target_sz=target_sz,
                    reduce_only=False,
                )
            elif self.use_market_orders:
                filled = self._market_open_unlocked(is_buy, target_sz)
            else:
                filled = self._fill_with_limits(
                    is_buy=is_buy,
                    target_sz=target_sz,
                    reduce_only=False,
                )
            if not filled:
                self.logger.warning("Entry not fully filled — flattening any partial")
                self._emergency_flatten_unlocked("entry_incomplete")
                return False

            self.client.cancel_entry_orders_for_coin()
            pos = self.client.get_position(force=True)
            if pos is None:
                self.client.cancel_all_orders_for_coin()
                return False

            eps = 10 ** (-self.client.sz_decimals)
            if pos.size + eps < target_sz:
                self.logger.warning(
                    "Position size %.8f < target %.8f — flattening",
                    pos.size,
                    target_sz,
                )
                self._emergency_flatten_unlocked("size_mismatch")
                return False

            if self.mid_limit_then_market:
                return self._protect_or_flatten_maker(
                    take_profit_pct, stop_loss_pct, after="entry"
                )

            if not self.client.attach_position_tpsl(
                pos,
                take_profit_pct,
                stop_loss_pct,
                max_attempts=3,
            ):
                self._emergency_flatten_unlocked("tpsl_attach_failed")
                return False

            self.client.cancel_entry_orders_for_coin()
            return True

    def execute_dca_add(
        self,
        add_sz: float,
        take_profit_pct: float,
        stop_loss_pct: float,
    ) -> "Position | None":
        """
        Add to the open position (AI-mode DCA), then re-center exchange TP/SL
        on the new average entry for the full size. Uses mid post-only then
        market when mid_limit_then_market is on; otherwise a market add.
        Returns the updated position, or None when the add failed / was skipped.
        Never leaves the position without both TP and SL (emergency flatten on
        attach failure).
        """
        with self._lock:
            pos = self.client.get_position(force=True)
            if pos is None:
                return None
            add_sz = round_size(add_sz, self.client.sz_decimals)
            if add_sz <= 0:
                return None
            is_buy = pos.side == "long"
            start_sz = pos.size
            if self.mid_limit_then_market:
                self.logger.info(
                    "DCA add: mid-limit then market %s +%s onto %s size=%s",
                    "BUY" if is_buy else "SELL",
                    add_sz,
                    pos.side,
                    pos.size,
                )
                added_ok = self._fill_mid_limit_then_market(
                    is_buy=is_buy,
                    target_sz=add_sz,
                    reduce_only=False,
                )
                if not added_ok:
                    self.client.cancel_entry_orders_for_coin()
                    pos_after = self.client.get_position(force=True)
                    if pos_after is None or pos_after.size <= start_sz:
                        return None
            else:
                self.logger.info(
                    "DCA add: market %s +%s onto %s size=%s",
                    "BUY" if is_buy else "SELL",
                    add_sz,
                    pos.side,
                    pos.size,
                )
                try:
                    result = self.client.place_market_open(
                        is_buy,
                        add_sz,
                        slippage=self.market_slippage,
                    )
                except Exception as exc:
                    self.logger.error("DCA add failed: %s", exc)
                    return None
                filled = self._parse_market_fill_sz(result)
                if filled is None or filled <= 0:
                    self.logger.warning("DCA add returned no fill: %s", result)
                    return None
                time.sleep(0.5)
                self.client.invalidate_user_state()
            pos = self.client.get_position(force=True)
            if pos is None:
                self.client.cancel_all_orders_for_coin()
                return None

            if self.mid_limit_then_market:
                status = self._protect_or_flatten_maker(
                    take_profit_pct, stop_loss_pct, after="dca"
                )
                if not status:
                    return None
                return self.client.get_position(force=True)

            # Old TP/SL triggers cover the old size/entry — replace them.
            self.client.cancel_all_orders_for_coin()
            if not self.client.attach_position_tpsl(
                pos,
                take_profit_pct,
                stop_loss_pct,
                max_attempts=3,
            ):
                self._emergency_flatten_unlocked("dca_tpsl_attach_failed")
                return None
            return self.client.get_position(force=True)

    def _position_flat(self) -> bool:
        eps = 10 ** (-self.client.sz_decimals)
        pos = self.client.get_position(force=True)
        return pos is None or pos.size <= eps

    def _protect_or_flatten_maker(
        self,
        take_profit_pct: float,
        stop_loss_pct: float,
        *,
        after: str,
    ) -> bool:
        """
        Park post-only TP + market SL. If TP would take (already at EMA),
        close with mid-limit then market. False means flatten / no open trade.
        """
        status = self.client.protect_ema_maker(
            take_profit_pct, stop_loss_pct, max_attempts=3
        )
        if status == "flat":
            self.client.cancel_all_orders_for_coin()
            return after == "dca"
        if status == "unprotected":
            self._emergency_flatten_unlocked(f"{after}_sl_attach_failed")
            return False
        if status == "would_take":
            pos = self.client.get_position(force=True)
            if pos is None:
                self.client.cancel_all_orders_for_coin()
                return after == "dca"
            self.logger.info(
                "Maker TP would take — closing at mid (post-only then market)"
            )
            is_buy = pos.side == "short"
            self.client.cancel_reduce_only_limits_for_coin()
            self.client.cancel_tp_triggers_for_coin()
            self._fill_mid_limit_then_market(
                is_buy=is_buy, target_sz=pos.size, reduce_only=True
            )
            self.client.cancel_all_orders_for_coin()
            if not self._position_flat():
                self._emergency_flatten_unlocked("maker_tp_would_take")
            return False
        return True

    def execute_close(self, is_buy: bool, target_sz: float) -> bool:
        with self._lock:
            return self._execute_close_unlocked(is_buy, target_sz)

    def execute_rsi_exit(self) -> bool:
        """
        RSI signal exit: cancel TP/SL triggers and limits, then close full position.
        Works for any coin (uses live position size + sz_decimals rounding).
        """
        with self._lock:
            self.client.cancel_all_orders_for_coin()
            self.client.invalidate_user_state()
            pos = self.client.get_position(force=True)
            if pos is None:
                self.client.cancel_all_orders_for_coin()
                return True
            is_buy = pos.side == "short"
            self.logger.info(
                "RSI exit — closing %s size=%s (szDecimals=%s)",
                pos.side,
                pos.size,
                self.client.sz_decimals,
            )
            return self._execute_close_unlocked(is_buy, pos.size)

    def execute_mid_limit_close(self) -> bool:
        """
        TP close: cancel leftover limits and the exchange TP trigger, keep SL,
        try post-only at mid (3 waits), then market the rest. Always cancels
        leftover limits at the end.
        """
        with self._lock:
            self.client.cancel_entry_orders_for_coin()
            self.client.cancel_reduce_only_limits_for_coin()
            self.client.cancel_tp_triggers_for_coin()
            self.client.invalidate_user_state()
            pos = self.client.get_position(force=True)
            if pos is None:
                self.client.cancel_all_orders_for_coin()
                return True
            is_buy = pos.side == "short"
            self.logger.info(
                "Mid-limit TP close %s size=%s (SL stays until flat)",
                pos.side,
                pos.size,
            )
            self._fill_mid_limit_then_market(
                is_buy=is_buy,
                target_sz=pos.size,
                reduce_only=True,
            )
            self.client.cancel_entry_orders_for_coin()
            if self._position_flat():
                self.client.cancel_all_orders_for_coin()
                return True
            current = self.client.get_position(force=True)
            if current is not None:
                self.logger.warning(
                    "Mid-limit TP close leftover size=%s — market flattening",
                    current.size,
                )
                try:
                    self.client.place_market_close(sz=current.size)
                except Exception as exc:
                    self.logger.error("Market close after mid-limit TP failed: %s", exc)
            self.client.cancel_all_orders_for_coin()
            return self._position_flat()

    def _execute_close_unlocked(self, is_buy: bool, target_sz: float) -> bool:
        self.client.cancel_all_orders_for_coin()
        start_pos = self.client.get_position(force=True)
        if start_pos is None:
            self.client.cancel_all_orders_for_coin()
            return True
        eps = 10 ** (-self.client.sz_decimals)
        close_sz = round_size(min(target_sz, start_pos.size), self.client.sz_decimals)
        if close_sz <= eps:
            self.client.cancel_all_orders_for_coin()
            return True

        if self.use_market_orders:
            self.logger.info("Market close size=%s", close_sz)
            try:
                result = self.client.place_market_close(sz=close_sz)
                self.logger.info("Market close result: %s", result)
            except Exception as exc:
                self.logger.error("Market close failed: %s", exc)
                return False
            self.client.cancel_all_orders_for_coin()
            return self._position_flat()

        filled = self._fill_with_limits(
            is_buy=is_buy,
            target_sz=close_sz,
            reduce_only=True,
        )
        self.client.cancel_all_orders_for_coin()
        if filled and self._position_flat():
            return True

        self.logger.warning(
            "Maker limit close incomplete after %s attempts — market closing remainder.",
            self.max_attempts,
        )
        current = self.client.get_position(force=True)
        if current is None or current.size <= eps:
            self.client.cancel_all_orders_for_coin()
            return True
        remainder = current.size
        self.logger.info("Market close remainder size=%s", remainder)
        try:
            result = self.client.place_market_close(sz=remainder)
            self.logger.info("Market close result: %s", result)
        except Exception as exc:
            self.logger.error("Market close failed: %s", exc)
            return False
        self.client.cancel_all_orders_for_coin()
        return self._position_flat()

    def _place_resting_maker_order(
        self,
        is_buy: bool,
        sz: float,
        reduce_only: bool,
        attempt_index: int,
    ) -> tuple[float, int | None]:
        """
        Try ALO limits until one rests or fills. Immediate ALO cancels do NOT
        consume one of the 5 outer wait attempts.
        """
        passive_nudge = 0
        for inner in range(1, MAX_ALO_PLACE_RETRIES + 1):
            try:
                l2 = self.client.l2_book()
                px = maker_limit_price(
                    l2,
                    is_buy,
                    self.client.sz_decimals,
                    attempt_index=attempt_index,
                    passive_nudge=passive_nudge,
                )
            except Exception as exc:
                self.logger.error("Maker price error: %s", exc)
                time.sleep(0.5)
                continue

            try:
                result = self.client.place_limit(
                    is_buy, sz, px, reduce_only=reduce_only
                )
            except Exception as exc:
                self.logger.error("ALO placement failed: %s", exc)
                time.sleep(0.5)
                continue

            filled, oid, alo_rejected = self.client.parse_fill_from_result(result)

            if alo_rejected:
                passive_nudge += 1
                self.logger.debug(
                    "ALO rejected (would take) — repricing maker px=%s inner=%s",
                    px,
                    inner,
                )
                time.sleep(0.4)
                continue

            if filled > 0:
                self.logger.info("ALO filled immediately (maker) sz=%s px=%s", filled, px)
                return filled, None

            if oid is not None:
                self.logger.info("ALO resting oid=%s px=%s sz=%s", oid, px, sz)
                return 0.0, oid

        self.logger.warning("Could not place resting ALO after %s reprices", MAX_ALO_PLACE_RETRIES)
        return 0.0, None

    def _cancel_oid(self, oid: int | None) -> None:
        if oid is None:
            return
        try:
            self.client.exchange.cancel(self.client.coin, oid)
            self.logger.info("Cancelled limit oid=%s", oid)
        except Exception as exc:
            self.logger.debug("Cancel oid=%s: %s", oid, exc)
        self.client.invalidate_user_state()

    def _place_resting_mid_order(
        self,
        is_buy: bool,
        sz: float,
        reduce_only: bool,
    ) -> tuple[float, int | None]:
        """Post-only (ALO) at mid, clamped so it cannot take. Inner ALO retries."""
        if hasattr(self.client, "_l2_cache"):
            self.client._l2_cache = None
        passive_nudge = 0
        for inner in range(1, MAX_ALO_PLACE_RETRIES + 1):
            try:
                l2 = self.client.l2_book()
                px = mid_post_only_price(
                    l2,
                    is_buy,
                    self.client.sz_decimals,
                    passive_nudge=passive_nudge,
                )
            except Exception as exc:
                self.logger.error("Mid price error: %s", exc)
                time.sleep(0.5)
                continue

            try:
                result = self.client.place_limit(
                    is_buy, sz, px, reduce_only=reduce_only
                )
            except Exception as exc:
                self.logger.error("Mid ALO placement failed: %s", exc)
                time.sleep(0.5)
                continue

            filled, oid, alo_rejected = self.client.parse_fill_from_result(result)

            if alo_rejected:
                passive_nudge += 1
                self.logger.info(
                    "Mid ALO rejected (would take) — more passive px=%s inner=%s",
                    px,
                    inner,
                )
                time.sleep(0.4)
                continue

            if filled > 0:
                self.logger.info("Mid ALO filled immediately sz=%s px=%s", filled, px)
                return filled, None

            if oid is not None:
                self.logger.info("Mid ALO resting oid=%s px=%s sz=%s", oid, px, sz)
                return 0.0, oid

        self.logger.warning(
            "Could not place mid ALO after %s reprices", MAX_ALO_PLACE_RETRIES
        )
        return 0.0, None

    @staticmethod
    def _position_filled_toward(
        start: Position | None,
        current: Position | None,
        is_buy: bool,
        target_sz: float,
        reduce_only: bool,
        sz_decimals: int,
    ) -> float:
        eps = 10 ** (-sz_decimals)

        if reduce_only:
            if start is None:
                return 0.0
            if current is None:
                return target_sz
            closed = start.size - current.size
            return min(target_sz, max(0.0, closed))

        want_side = "long" if is_buy else "short"
        start_same = start is not None and start.side == want_side
        base = start.size if start_same else 0.0
        if current is None or current.side != want_side:
            return 0.0
        gained = current.size - base
        return min(target_sz, max(0.0, gained))

    def _fill_with_limits(
        self,
        is_buy: bool,
        target_sz: float,
        reduce_only: bool,
    ) -> bool:
        start_pos = self.client.get_position(force=True)
        eps = 10 ** (-self.client.sz_decimals)

        for attempt in range(self.max_attempts):
            current = self.client.get_position(force=True)
            filled_so_far = self._position_filled_toward(
                start_pos,
                current,
                is_buy,
                target_sz,
                reduce_only,
                self.client.sz_decimals,
            )
            remaining = target_sz - filled_so_far
            if remaining <= eps:
                return True

            if reduce_only:
                self.client.cancel_all_orders_for_coin()
            else:
                self.client.cancel_entry_orders_for_coin()

            self.logger.info(
                "Maker limit %s fill attempt %s/%s remaining=%.8f",
                "BUY" if is_buy else "SELL",
                attempt + 1,
                self.max_attempts,
                remaining,
            )

            filled_now, oid = self._place_resting_maker_order(
                is_buy, remaining, reduce_only, attempt_index=attempt
            )

            if oid is not None:
                time.sleep(self.wait_seconds)
                try:
                    self.client.exchange.cancel(self.client.coin, oid)
                except Exception:
                    pass
                self.client.invalidate_user_state()
            elif filled_now <= 0:
                time.sleep(min(5, self.wait_seconds))

            current = self.client.get_position(force=True)
            filled_so_far = self._position_filled_toward(
                start_pos,
                current,
                is_buy,
                target_sz,
                reduce_only,
                self.client.sz_decimals,
            )
            if target_sz - filled_so_far <= eps:
                return True

        current = self.client.get_position(force=True)
        filled_so_far = self._position_filled_toward(
            start_pos,
            current,
            is_buy,
            target_sz,
            reduce_only,
            self.client.sz_decimals,
        )
        return target_sz - filled_so_far <= eps

    def _market_remainder(
        self,
        is_buy: bool,
        remaining: float,
        reduce_only: bool,
    ) -> None:
        remaining = round_size(remaining, self.client.sz_decimals)
        eps = 10 ** (-self.client.sz_decimals)
        if remaining <= eps:
            return
        self.logger.info(
            "Mid-limit leftover — market %s size=%s",
            "close" if reduce_only else ("BUY" if is_buy else "SELL"),
            remaining,
        )
        try:
            if reduce_only:
                result = self.client.place_market_close(sz=remaining)
            else:
                result = self.client.place_market_open(
                    is_buy,
                    remaining,
                    slippage=self.market_slippage,
                )
            self.logger.info("Market remainder result: %s", result)
        except Exception as exc:
            self.logger.error("Market remainder failed: %s", exc)
        time.sleep(0.5)
        self.client.invalidate_user_state()

    def _fill_mid_limit_then_market(
        self,
        is_buy: bool,
        target_sz: float,
        reduce_only: bool,
    ) -> bool:
        """
        Up to mid_limit_attempts post-only limits at refreshed mid, waiting
        mid_limit_wait_seconds each time. Cancels the resting oid before the
        next reprice. Unfilled size is sent as a market order.
        """
        start_pos = self.client.get_position(force=True)
        eps = 10 ** (-self.client.sz_decimals)
        target_sz = round_size(target_sz, self.client.sz_decimals)
        if target_sz <= eps:
            return True

        for attempt in range(self.mid_limit_attempts):
            current = self.client.get_position(force=True)
            filled_so_far = self._position_filled_toward(
                start_pos,
                current,
                is_buy,
                target_sz,
                reduce_only,
                self.client.sz_decimals,
            )
            remaining = round_size(target_sz - filled_so_far, self.client.sz_decimals)
            if remaining <= eps:
                self.client.cancel_entry_orders_for_coin()
                return True

            self.client.cancel_entry_orders_for_coin()
            self.logger.info(
                "Mid post-only %s attempt %s/%s remaining=%.8f wait=%.0fs",
                "BUY" if is_buy else "SELL",
                attempt + 1,
                self.mid_limit_attempts,
                remaining,
                self.mid_limit_wait_seconds,
            )
            filled_now, oid = self._place_resting_mid_order(
                is_buy, remaining, reduce_only
            )
            if oid is not None:
                time.sleep(self.mid_limit_wait_seconds)
                self._cancel_oid(oid)
            elif filled_now <= 0:
                time.sleep(min(1.0, self.mid_limit_wait_seconds))

            current = self.client.get_position(force=True)
            filled_so_far = self._position_filled_toward(
                start_pos,
                current,
                is_buy,
                target_sz,
                reduce_only,
                self.client.sz_decimals,
            )
            if target_sz - filled_so_far <= eps:
                self.client.cancel_entry_orders_for_coin()
                return True

        self.client.cancel_entry_orders_for_coin()
        current = self.client.get_position(force=True)
        filled_so_far = self._position_filled_toward(
            start_pos,
            current,
            is_buy,
            target_sz,
            reduce_only,
            self.client.sz_decimals,
        )
        remaining = round_size(target_sz - filled_so_far, self.client.sz_decimals)
        if remaining > eps:
            self._market_remainder(is_buy, remaining, reduce_only)
            current = self.client.get_position(force=True)
            filled_so_far = self._position_filled_toward(
                start_pos,
                current,
                is_buy,
                target_sz,
                reduce_only,
                self.client.sz_decimals,
            )
        self.client.cancel_entry_orders_for_coin()
        return target_sz - filled_so_far <= eps

    @staticmethod
    def _parse_market_fill_sz(result: dict[str, Any]) -> float | None:
        if result.get("status") != "ok":
            return None
        statuses = result.get("response", {}).get("data", {}).get("statuses", [])
        if not statuses or "filled" not in statuses[0]:
            return None
        return float(statuses[0]["filled"]["totalSz"])

    def _market_open_unlocked(self, is_buy: bool, target_sz: float) -> bool:
        """Single market open; caller verifies position + size."""
        self.client.cancel_entry_orders_for_coin()
        target_sz = round_size(target_sz, self.client.sz_decimals)
        self.logger.info(
            "Market %s open size=%s slippage=%.2f%%",
            "BUY" if is_buy else "SELL",
            target_sz,
            self.market_slippage * 100,
        )
        try:
            result = self.client.place_market_open(
                is_buy,
                target_sz,
                slippage=self.market_slippage,
            )
        except Exception as exc:
            self.logger.error("Market open failed: %s", exc)
            return False
        self.logger.info("Market open result: %s", result)
        filled = self._parse_market_fill_sz(result)
        if filled is None or filled <= 0:
            self.logger.warning("Market open returned no fill")
            return False
        time.sleep(0.5)
        self.client.invalidate_user_state()
        pos = self.client.get_position(force=True)
        if pos is None:
            return False
        eps = 10 ** (-self.client.sz_decimals)
        return pos.size + eps >= target_sz
