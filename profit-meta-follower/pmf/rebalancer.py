"""Execute consensus targets against our live (or paper) book."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from src.exchange_client import HyperliquidClient
from src.market_resolver import parse_coin_input
from src.pricing import floor_size, round_size

from .snapshots import SnapshotClient, parse_positions, account_value_from_states
from .types import OurPos, TargetPos


@dataclass
class Action:
    kind: str  # close | open | resize
    coin: str
    side: str
    size: float
    leverage: int
    reason: str


class PaperBook:
    """Multi-coin paper ledger. Prices from live mids; no real orders."""

    def __init__(self, start_balance: float, fee_pct: float, logger: logging.Logger) -> None:
        self.cash = float(start_balance)
        self.fee_frac = max(0.0, fee_pct) / 100.0
        self.log = logger
        self.positions: dict[str, OurPos] = {}

    def equity(self, marks: dict[str, float]) -> float:
        eq = self.cash
        for coin, pos in self.positions.items():
            mark = marks.get(coin) or (pos.entry_px or 0.0)
            signed = pos.size if pos.side == "long" else -pos.size
            entry = pos.entry_px or mark
            eq += signed * (mark - entry)
        return eq

    def apply_close(self, coin: str, mark: float) -> None:
        pos = self.positions.pop(coin, None)
        if pos is None:
            return
        signed = pos.size if pos.side == "long" else -pos.size
        entry = pos.entry_px or mark
        pnl = signed * (mark - entry)
        fee = abs(pos.size * mark) * self.fee_frac
        self.cash += pnl - fee
        self.log.info("PAPER close %s %s sz=%.6f @ %.6f pnl=$%.2f", coin, pos.side, pos.size, mark, pnl - fee)

    def apply_open(self, coin: str, side: str, size: float, mark: float, leverage: int) -> None:
        fee = abs(size * mark) * self.fee_frac
        self.cash -= fee
        self.positions[coin] = OurPos(
            coin=coin,
            side=side,
            size=size,
            notional=abs(size * mark),
            entry_px=mark,
            leverage=leverage,
        )
        self.log.info("PAPER open %s %s sz=%.6f @ %.6f lev=%sx fee=$%.2f", coin, side, size, mark, leverage, fee)


def our_positions_from_client(client: HyperliquidClient, snapper: SnapshotClient) -> list[OurPos]:
    states = snapper.fetch_user_states(client.address)
    equity = account_value_from_states(states)
    parsed = parse_positions(states, equity or 1.0)
    out: list[OurPos] = []
    for p in parsed:
        out.append(
            OurPos(
                coin=p.coin,
                side=p.side,
                size=p.size,
                notional=p.notional,
                entry_px=p.entry_px,
                leverage=p.leverage,
            )
        )
    return out


def _target_notional(t: TargetPos, equity: float) -> float:
    return max(0.0, (t.margin_pct / 100.0) * equity * max(1, t.leverage))


def plan_actions(
    ours: list[OurPos],
    targets: list[TargetPos],
    equity: float,
    cfg: Any,
    managed: set[str],
) -> list[Action]:
    have = {p.coin: p for p in ours}
    want = {t.coin: t for t in targets}
    drift = float(cfg.REBALANCE_DRIFT_PCT) / 100.0
    actions: list[Action] = []

    for coin, pos in have.items():
        if bool(cfg.MANAGED_ONLY) and coin not in managed and coin not in want:
            continue
        t = want.get(coin)
        if t is None:
            if bool(cfg.FLATTEN_WHEN_DROPPED):
                actions.append(Action("close", coin, pos.side, pos.size, pos.leverage, "dropped_from_book"))
            continue
        if t.side != pos.side:
            actions.append(Action("close", coin, pos.side, pos.size, pos.leverage, "flip_close"))
            continue
        tgt_n = _target_notional(t, equity)
        if pos.notional <= 0 or tgt_n <= 0:
            continue
        if abs(pos.notional - tgt_n) / max(pos.notional, tgt_n) > drift:
            actions.append(Action("resize", coin, t.side, 0.0, t.leverage, "drift"))

    for coin, t in want.items():
        pos = have.get(coin)
        if pos is None:
            actions.append(Action("open", coin, t.side, 0.0, t.leverage, "new"))
        elif pos.side != t.side:
            actions.append(Action("open", coin, t.side, 0.0, t.leverage, "flip_open"))

    # Closes first, then resizes, then opens. Cap later.
    order = {"close": 0, "resize": 1, "open": 2}
    actions.sort(key=lambda a: (order.get(a.kind, 9), a.coin))
    return actions[: int(cfg.MAX_ACTIONS_PER_CYCLE)]


class Rebalancer:
    def __init__(
        self,
        client: HyperliquidClient,
        snapper: SnapshotClient,
        cfg: Any,
        logger: logging.Logger,
        paper: PaperBook | None = None,
    ) -> None:
        self.client = client
        self.snapper = snapper
        self.cfg = cfg
        self.log = logger
        self.paper = paper
        self._dust_skip_until: dict[str, float] = {}

    def _mark(self, coin: str) -> float:
        sym, dex = parse_coin_input(coin)
        self.client.configure_coin(f"{dex}:{sym}" if dex else sym, perp_dex=dex)
        return float(self.client.get_mark_price())

    def current_book(self) -> tuple[list[OurPos], float]:
        if self.paper is not None:
            marks = {}
            for coin in list(self.paper.positions):
                try:
                    marks[coin] = self._mark(coin)
                except Exception:
                    marks[coin] = self.paper.positions[coin].entry_px or 0.0
            eq = self.paper.equity(marks)
            ours = list(self.paper.positions.values())
            for p in ours:
                p.notional = abs(p.size * (marks.get(p.coin) or p.entry_px or 0.0))
            return ours, eq
        ours = our_positions_from_client(self.client, self.snapper)
        eq = float(self.client.get_account_value(force=True))
        return ours, eq

    def _size_for_target(self, t: TargetPos, equity: float) -> tuple[float, float]:
        """Return (size, notional) using live mark + exchange decimals."""
        mark = self._mark(t.coin)
        if mark <= 0:
            return 0.0, 0.0
        notional = _target_notional(t, equity)
        raw = notional / mark
        sz = floor_size(raw, self.client.sz_decimals)
        if sz <= 0:
            return 0.0, 0.0
        return sz, sz * mark

    def _close(self, coin: str, size: float) -> bool:
        mark = self._mark(coin)
        if self.paper is not None:
            self.paper.apply_close(coin, mark)
            return True
        self.client.cancel_all_orders_for_coin()
        pos = self.client.get_position(force=True)
        if pos is None:
            return True
        try:
            self.client.place_market_close(sz=pos.size)
        except Exception as exc:
            self.log.error("Close %s failed: %s", coin, exc)
            return False
        time.sleep(1.2)
        left = self.client.get_position(force=True)
        if left is not None:
            self.log.error("Close %s incomplete — still %s %.6f", coin, left.side, left.size)
            return False
        self.log.info("Closed %s", coin)
        return True

    def _spread_pct(self) -> float:
        try:
            book = self.client.l2_book() or {}
            levels = book.get("levels") if isinstance(book, dict) else None
            if not isinstance(levels, (list, tuple)) or len(levels) < 2:
                return 1.0
            bids, asks = levels[0], levels[1]
            if not bids or not asks:
                return 1.0
            bid = float(bids[0]["px"])
            ask = float(asks[0]["px"])
            mid = (bid + ask) / 2.0
            if mid <= 0:
                return 1.0
            return max(0.0, (ask - bid) / mid)
        except Exception:
            return 1.0

    def _open(self, t: TargetPos, equity: float) -> bool:
        now = time.time()
        until = float(self._dust_skip_until.get(t.coin) or 0.0)
        if until > now:
            return False
        try:
            sz, notional = self._size_for_target(t, equity)
        except Exception as exc:
            self.log.warning("Skip open %s — market resolve failed: %s", t.coin, exc)
            self._dust_skip_until[t.coin] = now + 300.0
            return False
        if sz <= 0 or notional <= 0:
            self.log.info("Skip open %s — size floors to 0 (dust notional)", t.coin)
            self._dust_skip_until[t.coin] = now + 600.0
            return False
        if notional < float(self.cfg.MIN_ORDER_NOTIONAL_USD):
            self.log.info("Skip open %s — notional $%.2f below min", t.coin, notional)
            self._dust_skip_until[t.coin] = now + 300.0
            return False
        mark = self.client.get_mark_price()
        spread = self._spread_pct()
        max_spread = float(getattr(self.cfg, "MAX_SPREAD_PCT", 0.0) or 0.0)
        if max_spread > 0 and spread > max_spread:
            self.log.warning(
                "Skip open %s — spread %.3f%% > max %.3f%%",
                t.coin,
                spread * 100.0,
                max_spread * 100.0,
            )
            return False
        if self.paper is not None:
            self.paper.apply_open(t.coin, t.side, sz, mark, t.leverage)
            return True
        is_buy = t.side == "long"
        try:
            self.client.set_leverage(t.leverage, is_cross=bool(self.cfg.USE_CROSS_MARGIN) and not self.client.only_isolated)
        except Exception as exc:
            self.log.warning("Leverage set failed %s: %s — continue", t.coin, exc)
        avail = self.client.get_available_margin(force=True)
        need = notional / max(1, t.leverage)
        if avail < need * 0.9:
            self.log.warning(
                "Skip open %s — need margin $%.2f have $%.2f (HIP-3/manual pots can be empty)",
                t.coin,
                need,
                avail,
            )
            return False
        try:
            if bool(self.cfg.USE_MARKET_ORDERS):
                self.log.info("Market %s %s sz=%.6f notional=$%.2f lev=%sx", t.side, t.coin, sz, notional, t.leverage)
                result = self.client.place_market_open(
                    is_buy, sz, slippage=float(self.cfg.MARKET_ORDER_SLIPPAGE)
                )
                self.log.info("Open result %s: %s", t.coin, result)
            else:
                raise RuntimeError("limit path not used")
        except Exception as exc:
            self.log.error("Open %s failed: %s", t.coin, exc)
            return False
        time.sleep(1.2)
        pos = self.client.get_position(force=True)
        if pos is None:
            self.log.error("Open %s — no position after order", t.coin)
            return False
        return True

    def _resize(self, t: TargetPos, have: OurPos, equity: float) -> bool:
        try:
            want_sz, want_n = self._size_for_target(t, equity)
        except Exception as exc:
            self.log.warning("Skip resize %s: %s", t.coin, exc)
            return False
        mark = self.client.get_mark_price()
        if want_sz <= 0:
            return self._close(t.coin, have.size)
        delta = want_sz - have.size
        step = 10 ** (-self.client.sz_decimals)
        if abs(delta) < step * 1.1:
            return True
        if self.paper is not None:
            # Rebuild paper position at mark for the new size.
            self.paper.apply_close(t.coin, mark)
            if want_sz > 0:
                self.paper.apply_open(t.coin, t.side, want_sz, mark, t.leverage)
            return True
        if delta < 0:
            close_sz = round_size(abs(delta), self.client.sz_decimals)
            try:
                self.client.place_market_close(sz=close_sz)
            except Exception as exc:
                self.log.error("Reduce %s failed: %s", t.coin, exc)
                return False
            return True
        add_sz = round_size(delta, self.client.sz_decimals)
        try:
            self.client.set_leverage(t.leverage, is_cross=bool(self.cfg.USE_CROSS_MARGIN) and not self.client.only_isolated)
            self.client.place_market_open(
                t.side == "long",
                add_sz,
                slippage=float(self.cfg.MARKET_ORDER_SLIPPAGE),
            )
        except Exception as exc:
            self.log.error("Add %s failed: %s", t.coin, exc)
            return False
        return True

    def run(
        self,
        targets: list[TargetPos],
        managed: set[str],
        last_rebalance_at: float,
        now: float,
    ) -> tuple[set[str], bool]:
        try:
            ours, equity = self.current_book()
        except Exception as exc:
            self.log.error("Could not read our book: %s", exc)
            return managed, False
        if equity <= 0:
            self.log.warning("Equity is 0 — skip rebalance")
            return managed, False
        have = {p.coin: p for p in ours}
        actions = plan_actions(ours, targets, equity, self.cfg, managed)
        if not actions:
            return managed, False
        cooldown = float(self.cfg.REBALANCE_COOLDOWN_S)
        on_cooldown = cooldown > 0 and (now - last_rebalance_at) < cooldown
        if on_cooldown:
            # Never block exits — cooldown only slows opens/resizes (anti-scalp).
            actions = [a for a in actions if a.kind == "close"]
            if not actions:
                return managed, False
        self.log.info(
            "Rebalance %s action(s) equity=$%.2f targets=%s",
            len(actions),
            equity,
            ", ".join(f"{t.side}:{t.coin}@{t.margin_pct:.1f}%" for t in targets) or "-",
        )
        target_map = {t.coin: t for t in targets}
        new_managed = set(managed)
        did = 0
        failed_close: set[str] = set()
        for act in actions:
            try:
                if act.kind == "close":
                    ok = self._close(act.coin, act.size)
                    if ok:
                        new_managed.discard(act.coin)
                        did += 1
                    else:
                        failed_close.add(act.coin)
                elif act.kind == "open":
                    if act.coin in failed_close:
                        self.log.warning("Skip open %s — close did not finish", act.coin)
                        continue
                    t = target_map.get(act.coin)
                    if t is None:
                        continue
                    ok = self._open(t, equity)
                    if ok:
                        new_managed.add(act.coin)
                        did += 1
                elif act.kind == "resize":
                    t = target_map.get(act.coin)
                    if t is None:
                        continue
                    ok = self._resize(t, have[act.coin], equity)
                    if ok:
                        new_managed.add(act.coin)
                        did += 1
            except Exception as exc:
                self.log.error("Action %s %s crashed: %s", act.kind, act.coin, exc)
        if did:
            self.log.info("Rebalance applied %s/%s", did, len(actions))
        # Only start the cooldown after a fill. Failed opens retry next tick.
        return new_managed, did > 0
