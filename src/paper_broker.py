"""
Paper-trading (simulation) broker for the EMA-deviation bot.

PaperHyperliquidClient subclasses the real HyperliquidClient but never sends a
real order. It keeps every read-only market-data path (candles, EMA, mids,
L2 book, market metadata) using the real Hyperliquid Info connection, so signals
are computed on live prices exactly like production. Only the account side is
simulated:

  - a persisted cash balance (USD)
  - one or more open paper positions (same-side DCA on a coin still stacks)
  - exchange-style market fills at the live mid price
  - simulated exchange TP/SL that auto-closes when the live mark crosses a level
  - taker fees on entry and exit (mirrors the backtest: 0.045% x2 by default)

State is persisted so a restart behaves like reconnecting to an exchange:
  data/paper_account.json   balance + open positions
  data/paper_trades.jsonl   one JSON line per closed trade

Nothing here touches real funds. Prices are read from mainnet for realism even
if the live bot is configured for testnet.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_files import append_jsonl
from .exchange_client import HyperliquidClient, Position
from .pricing import round_size


class PaperHyperliquidClient(HyperliquidClient):
    """Drop-in HyperliquidClient that simulates the account/order side."""

    def __init__(
        self,
        *args: Any,
        paper_data_dir: Path,
        paper_start_balance: float = 1000.0,
        paper_taker_fee_pct: float = 0.045,
        account_filename: str = "ema_dev_paper_account.json",
        trades_filename: str = "ema_dev_paper_trades.jsonl",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._paper_dir = Path(paper_data_dir)
        self._paper_dir.mkdir(parents=True, exist_ok=True)
        self._account_path = self._paper_dir / account_filename
        self._trades_path = self._paper_dir / trades_filename
        self._fee_frac = max(0.0, float(paper_taker_fee_pct)) / 100.0
        self._start_balance = max(0.0, float(paper_start_balance))
        self._paper_leverage = max(1, int(self.max_leverage))
        self.balance: float = self._start_balance
        # coin -> {coin, symbol, perp_dex, side, size, entry_px, leverage,
        #          tp_px, sl_px, entry_fee, opened_at}
        self.positions: dict[str, dict[str, Any]] = {}
        self._load_account()
        self.logger.info(
            "PAPER TRADING active — no real orders. balance=$%.2f fee=%.3f%%x2 "
            "| account=%s | trades=%s",
            self.balance,
            self._fee_frac * 100.0,
            self._account_path,
            self._trades_path,
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load_account(self) -> None:
        if not self._account_path.exists():
            self._save_account()
            return
        try:
            data = json.loads(self._account_path.read_text(encoding="utf-8"))
            self.balance = float(data.get("balance", self._start_balance))
            self.positions = {}
            raw_many = data.get("positions")
            if isinstance(raw_many, dict) and raw_many:
                for coin, pos in raw_many.items():
                    if isinstance(pos, dict) and pos.get("side"):
                        pos = dict(pos)
                        pos.setdefault("coin", coin)
                        self.positions[str(pos.get("coin") or coin)] = pos
            else:
                pos = data.get("position")
                if isinstance(pos, dict) and pos.get("side"):
                    self.positions[str(pos.get("coin") or self.coin)] = pos
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            self.logger.warning("Paper account file unreadable — starting fresh")
            self.balance = self._start_balance
            self.positions = {}

    def _save_account(self) -> None:
        payload = {
            "balance": round(self.balance, 8),
            "updated_at": time.time(),
            "updated_at_iso": datetime.now(timezone.utc).isoformat(),
            "positions": self.positions,
            "position": self.positions.get(self.coin),
        }
        tmp = self._account_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._account_path)

    @property
    def position(self) -> dict[str, Any] | None:
        return self.positions.get(self.coin)

    def _pos_obj(self, pos: dict[str, Any]) -> Position:
        return Position(
            side=pos["side"],
            size=float(pos["size"]),
            entry_price=float(pos["entry_px"]),
        )

    def _record_trade(self, row: dict[str, Any]) -> None:
        try:
            append_jsonl(self._trades_path, row, logger=self.logger)
        except OSError as exc:
            self.logger.warning("Paper trade log write failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Price helpers
    # ------------------------------------------------------------------ #
    def _paper_mark_for(self, coin: str, symbol: str, dex: str) -> float | None:
        """Live mid for an arbitrary coin (may differ from the active coin)."""
        try:
            mids = self.info.all_mids(dex=dex or "")
        except Exception as exc:
            self.logger.debug("paper mark lookup failed for %s: %s", coin, exc)
            return None
        keys = [coin, symbol]
        if dex:
            keys.append(f"{dex}:{symbol}")
        for key in keys:
            if key and key in mids:
                try:
                    return float(mids[key])
                except (TypeError, ValueError):
                    return None
        return None

    # ------------------------------------------------------------------ #
    # TP/SL simulation
    # ------------------------------------------------------------------ #
    def _exit_reason_for(self, pos: dict[str, Any], mark: float) -> str | None:
        side = pos["side"]
        tp = pos.get("tp_px")
        sl = pos.get("sl_px")
        if side == "long":
            # Conservative: if a single tick spans both, treat as stop first.
            if sl is not None and mark <= sl:
                return "stop_loss"
            if tp is not None and mark >= tp:
                return "take_profit"
        else:
            if sl is not None and mark >= sl:
                return "stop_loss"
            if tp is not None and mark <= tp:
                return "take_profit"
        return None

    def _settle(self) -> None:
        """Mark-to-market every paper position; auto-close on TP/SL cross."""
        if not self.positions:
            return
        for coin in list(self.positions.keys()):
            pos = self.positions.get(coin)
            if not pos:
                continue
            if pos.get("tp_px") is None and pos.get("sl_px") is None:
                continue
            mark = self._paper_mark_for(pos["coin"], pos["symbol"], pos.get("perp_dex", ""))
            if mark is None or mark <= 0:
                continue
            reason = self._exit_reason_for(pos, mark)
            if reason is None:
                continue
            fill_px = pos["tp_px"] if reason == "take_profit" else pos["sl_px"]
            self._close_position(float(fill_px), reason, coin=coin)

    def _close_position(self, exit_px: float, reason: str, coin: str | None = None) -> None:
        key = coin or self.coin
        pos = self.positions.get(key)
        if not pos:
            return
        side = pos["side"]
        size = float(pos["size"])
        entry_px = float(pos["entry_px"])
        direction = 1.0 if side == "long" else -1.0
        gross = (exit_px - entry_px) * size * direction
        exit_fee = exit_px * size * self._fee_frac
        entry_fee = float(pos.get("entry_fee", 0.0))
        net = gross - exit_fee  # entry_fee already deducted at open
        self.balance += net
        opened_at = float(pos.get("opened_at", time.time()))
        now = time.time()
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "coin": pos["coin"],
            "side": side,
            "reason": reason,
            "size": size,
            "leverage": pos.get("leverage"),
            "entry_px": entry_px,
            "exit_px": float(exit_px),
            "gross_pnl": round(gross, 6),
            "entry_fee": round(entry_fee, 6),
            "exit_fee": round(exit_fee, 6),
            "net_pnl": round(gross - entry_fee - exit_fee, 6),
            "hold_seconds": round(max(0.0, now - opened_at), 1),
            "balance_after": round(self.balance, 6),
        }
        self.positions.pop(key, None)
        self._save_account()
        self._record_trade(row)
        self.logger.info(
            "PAPER close %s %s @ %.8f (%s) | net=$%.4f balance=$%.2f",
            pos["coin"],
            side,
            exit_px,
            reason,
            row["net_pnl"],
            self.balance,
        )

    # ------------------------------------------------------------------ #
    # Account queries (overrides)
    # ------------------------------------------------------------------ #
    def get_user_abstraction(self, *, force: bool = False) -> str:
        return "paper"

    def uses_unified_collateral(self) -> bool:
        return False

    def _unrealized_gross(self) -> float:
        total = 0.0
        for pos in self.positions.values():
            mark = self._paper_mark_for(pos["coin"], pos["symbol"], pos.get("perp_dex", ""))
            if mark is None or mark <= 0:
                continue
            direction = 1.0 if pos["side"] == "long" else -1.0
            total += (mark - float(pos["entry_px"])) * float(pos["size"]) * direction
        return total

    def _margin_used(self) -> float:
        used = 0.0
        for pos in self.positions.values():
            notional = float(pos["entry_px"]) * float(pos["size"])
            lev = max(1, int(pos.get("leverage", self._paper_leverage)))
            used += notional / lev
        return used

    def get_account_value(self, *, force: bool = False) -> float:
        self._settle()
        return max(0.0, self.balance + self._unrealized_gross())

    def get_available_margin(self, *, force: bool = False) -> float:
        self._settle()
        return max(0.0, self.balance + self._unrealized_gross() - self._margin_used())

    # ------------------------------------------------------------------ #
    # Position queries (overrides)
    # ------------------------------------------------------------------ #
    def invalidate_user_state(self) -> None:  # no cached user state in paper mode
        return

    def _paper_position_obj(self) -> Position | None:
        pos = self.positions.get(self.coin)
        if not pos:
            return None
        return self._pos_obj(pos)

    def _iter_clearinghouse_states(self) -> list[dict[str, Any]]:
        self._settle()
        assets = []
        for pos in self.positions.values():
            szi = float(pos["size"]) * (1.0 if pos["side"] == "long" else -1.0)
            assets.append(
                {
                    "position": {
                        "coin": pos["coin"],
                        "szi": szi,
                        "entryPx": float(pos["entry_px"]),
                    }
                }
            )
        return [{"assetPositions": assets}]

    def get_position(self, *, force: bool = False) -> Position | None:
        self._settle()
        pos = self.positions.get(self.coin)
        if not pos:
            return None
        if not self._matches_active_coin(pos["coin"]):
            return None
        return self._pos_obj(pos)

    def fetch_open_positions(
        self, *, force: bool = False
    ) -> tuple[bool, list[tuple[str, Position]]]:
        self._settle()
        out: list[tuple[str, Position]] = []
        for pos in self.positions.values():
            out.append((pos["coin"], self._pos_obj(pos)))
        return True, out

    def has_any_open_position(self, *, force: bool = False) -> bool:
        self._settle()
        return bool(self.positions)

    # ------------------------------------------------------------------ #
    # Leverage / orders (overrides — never hit the real exchange)
    # ------------------------------------------------------------------ #
    def set_leverage(self, leverage: int, is_cross: bool = True) -> None:
        requested = max(1, int(leverage))
        self._paper_leverage = min(requested, max(1, int(self.max_leverage)))

    @staticmethod
    def _fill_result(sz: float, px: float) -> dict[str, Any]:
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"filled": {"totalSz": str(sz), "avgPx": str(px), "oid": 0}}
                    ]
                }
            },
        }

    def place_market_open(
        self, is_buy: bool, sz: float, *, slippage: float = 0.05
    ) -> dict[str, Any]:
        sz = round_size(sz, self.sz_decimals)
        if sz <= 0:
            return {"status": "err", "response": "zero size"}
        mark = self.get_mark_price()
        if mark <= 0:
            return {"status": "err", "response": "no mark price"}
        side = "long" if is_buy else "short"
        existing = self.positions.get(self.coin)
        # Same-side add = DCA (matches live place_market_open stacking).
        if existing is not None:
            pos = existing
            if pos["side"] != side:
                self.logger.warning(
                    "PAPER open refused — opposite-side position already open on %s",
                    self.coin,
                )
                return {"status": "err", "response": "opposite position exists"}
            old_sz = float(pos["size"])
            old_px = float(pos["entry_px"])
            entry_fee = mark * sz * self._fee_frac
            self.balance -= entry_fee
            new_sz = old_sz + float(sz)
            avg_px = (old_px * old_sz + mark * float(sz)) / new_sz
            pos["size"] = new_sz
            pos["entry_px"] = float(avg_px)
            pos["entry_fee"] = float(pos.get("entry_fee", 0.0)) + float(entry_fee)
            pos["tp_px"] = None
            pos["sl_px"] = None
            self._save_account()
            self.logger.info(
                "PAPER DCA add %s %s +%s @ %.8f avg=%.8f fee=$%.4f balance=$%.2f",
                self.coin,
                side,
                sz,
                mark,
                avg_px,
                entry_fee,
                self.balance,
            )
            return self._fill_result(sz, mark)

        entry_fee = mark * sz * self._fee_frac
        self.balance -= entry_fee
        self.positions[self.coin] = {
            "coin": self.coin,
            "symbol": self.market.symbol,
            "perp_dex": self.perp_dex or "",
            "side": side,
            "size": float(sz),
            "entry_px": float(mark),
            "leverage": int(self._paper_leverage),
            "tp_px": None,
            "sl_px": None,
            "entry_fee": float(entry_fee),
            "opened_at": time.time(),
        }
        self._save_account()
        self.logger.info(
            "PAPER open %s %s size=%s @ %.8f fee=$%.4f balance=$%.2f",
            self.coin,
            side,
            sz,
            mark,
            entry_fee,
            self.balance,
        )
        return self._fill_result(sz, mark)

    def place_market_close(self, sz: float | None = None) -> dict[str, Any]:
        pos = self.positions.get(self.coin)
        if not pos:
            return {"status": "ok", "response": {"data": {"statuses": []}}}
        mark = self._paper_mark_for(pos["coin"], pos["symbol"], pos.get("perp_dex", ""))
        if mark is None or mark <= 0:
            mark = self.get_mark_price()
        close_sz = float(pos["size"]) if sz is None else min(float(sz), float(pos["size"]))
        self._close_position(float(mark), "manual_close", coin=self.coin)
        return self._fill_result(close_sz, mark)

    def place_limit(
        self,
        is_buy: bool,
        sz: float,
        limit_px: float,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Simulate a maker limit as an immediate fill at the limit price."""
        sz = round_size(sz, self.sz_decimals)
        if sz <= 0:
            return {"status": "err", "response": "zero size"}
        if reduce_only:
            if self.coin in self.positions:
                self._close_position(float(limit_px), "manual_close", coin=self.coin)
            return self._fill_result(sz, limit_px)
        side = "long" if is_buy else "short"
        existing = self.positions.get(self.coin)
        if existing is not None:
            if existing["side"] != side:
                return {"status": "err", "response": "opposite position exists"}
            old_sz = float(existing["size"])
            old_px = float(existing["entry_px"])
            entry_fee = float(limit_px) * sz * self._fee_frac
            self.balance -= entry_fee
            new_sz = old_sz + float(sz)
            avg_px = (old_px * old_sz + float(limit_px) * float(sz)) / new_sz
            existing["size"] = new_sz
            existing["entry_px"] = float(avg_px)
            existing["entry_fee"] = float(existing.get("entry_fee", 0.0)) + float(entry_fee)
            existing["tp_px"] = None
            existing["sl_px"] = None
            self._save_account()
            return self._fill_result(sz, limit_px)
        entry_fee = float(limit_px) * sz * self._fee_frac
        self.balance -= entry_fee
        self.positions[self.coin] = {
            "coin": self.coin,
            "symbol": self.market.symbol,
            "perp_dex": self.perp_dex or "",
            "side": side,
            "size": float(sz),
            "entry_px": float(limit_px),
            "leverage": int(self._paper_leverage),
            "tp_px": None,
            "sl_px": None,
            "entry_fee": float(entry_fee),
            "opened_at": time.time(),
        }
        self._save_account()
        return self._fill_result(sz, limit_px)

    # ------------------------------------------------------------------ #
    # TP/SL attach + order bookkeeping (overrides)
    # ------------------------------------------------------------------ #
    def attach_position_tpsl(
        self,
        position: Position,
        take_profit_pct: float,
        stop_loss_pct: float,
        *,
        max_attempts: int = 3,
    ) -> bool:
        pos = self.positions.get(self.coin)
        if not pos:
            self.logger.warning("PAPER attach_tpsl skipped — no open position")
            return False
        entry = float(pos["entry_px"])
        tp_px, sl_px = self.tp_sl_prices_for_entry(
            pos["side"], entry, take_profit_pct, stop_loss_pct
        )
        pos["tp_px"] = float(tp_px)
        pos["sl_px"] = float(sl_px)
        self._save_account()
        self.logger.info(
            "PAPER TP/SL set %s %s entry=%.8f tp=%.8f sl=%.8f",
            pos["coin"],
            pos["side"],
            entry,
            tp_px,
            sl_px,
        )
        # A trigger could already be in the money at attach time.
        self._settle()
        return True

    def protect_ema_maker(
        self,
        take_profit_pct: float,
        stop_loss_pct: float,
        *,
        max_attempts: int = 3,
    ) -> str:
        pos = self.positions.get(self.coin)
        if not pos:
            return "flat"
        entry = float(pos["entry_px"])
        tp_px, sl_px = self.tp_sl_prices_for_entry(
            pos["side"], entry, take_profit_pct, stop_loss_pct
        )
        pos["tp_px"] = float(tp_px)
        pos["sl_px"] = float(sl_px)
        self._save_account()
        self.logger.info(
            "PAPER maker TP + SL %s %s entry=%.8f tp=%.8f sl=%.8f",
            pos["coin"],
            pos["side"],
            entry,
            tp_px,
            sl_px,
        )
        self._settle()
        if self.coin not in self.positions:
            return "flat"
        return "ok"

    def has_exchange_tpsl(self) -> bool:
        pos = self.positions.get(self.coin)
        return bool(pos and pos.get("tp_px") is not None and pos.get("sl_px") is not None)

    def has_exchange_sl(self) -> bool:
        pos = self.positions.get(self.coin)
        return bool(pos and pos.get("sl_px") is not None)

    def has_resting_tp_limit(self) -> bool:
        pos = self.positions.get(self.coin)
        return bool(pos and pos.get("tp_px") is not None)

    def resting_tp_px(self) -> float | None:
        pos = self.positions.get(self.coin)
        if not pos or pos.get("tp_px") is None:
            return None
        return float(pos["tp_px"])

    def has_open_entry_orders(self) -> bool:
        return False

    def _all_frontend_orders(self) -> list[dict]:
        return []

    def _frontend_orders_for_coin(self) -> list[dict]:
        return []

    def _reduce_only_triggers(self) -> list[dict]:
        return []

    def sweep_orphan_orders(self, keep_coins: set[str] | None = None) -> int:
        return 0

    def cancel_all_orders_for_coin(self) -> None:
        return

    def cancel_oid(self, oid: int | None) -> None:
        return

    def working_limit_quotes(self) -> tuple[dict | None, dict | None]:
        return None, None

    def cancel_all_orders_for_coin_named(self, coin: str) -> None:
        return

    def cancel_open_orders_for_coin(self) -> None:
        return

    def cancel_entry_orders_for_coin(self) -> None:
        return

    def cancel_tp_triggers_for_coin(self) -> None:
        return

    def cancel_sl_triggers_for_coin(self) -> None:
        return

    def cancel_reduce_only_limits_for_coin(self, *, keep_oid: int | None = None) -> None:
        return
