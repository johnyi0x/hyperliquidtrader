"""Active trade TP/SL levels (persisted across restarts)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ActiveTrade:
    coin: str
    side: str
    entry_price: float
    size: float
    take_profit_price: float
    stop_loss_price: float
    opened_at: float
    equity_at_entry: float = 0.0
    take_profit_pct: float = 0.0
    stop_loss_pct: float = 0.0
    initial_size: float = 0.0  # size of the first fill (DCA adds scale off this)
    dca_adds: int = 0  # DCA adds executed so far this trade

    def check_exit(self, mark: float) -> str | None:
        if self.side == "long":
            if mark >= self.take_profit_price:
                return "take_profit"
            if mark <= self.stop_loss_price:
                return "stop_loss"
        else:
            if mark <= self.take_profit_price:
                return "take_profit"
            if mark >= self.stop_loss_price:
                return "stop_loss"
        return None


class TradeStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.trade: ActiveTrade | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.trade = ActiveTrade(
                coin=str(data.get("coin", "")),
                side=str(data["side"]),
                entry_price=float(data["entry_price"]),
                size=float(data["size"]),
                take_profit_price=float(data["take_profit_price"]),
                stop_loss_price=float(data["stop_loss_price"]),
                opened_at=float(data.get("opened_at", time.time())),
                equity_at_entry=float(data.get("equity_at_entry", 0.0)),
                take_profit_pct=float(data.get("take_profit_pct", 0.0)),
                stop_loss_pct=float(data.get("stop_loss_pct", 0.0)),
                initial_size=float(data.get("initial_size", data.get("size", 0.0))),
                dca_adds=int(data.get("dca_adds", 0)),
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            self.trade = None

    def _save(self) -> None:
        if self.trade is None:
            if self.path.exists():
                try:
                    self.path.unlink()
                except OSError:
                    pass
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        t = self.trade
        payload = {
            "coin": t.coin,
            "side": t.side,
            "entry_price": t.entry_price,
            "size": t.size,
            "take_profit_price": t.take_profit_price,
            "stop_loss_price": t.stop_loss_price,
            "opened_at": t.opened_at,
            "equity_at_entry": t.equity_at_entry,
            "take_profit_pct": t.take_profit_pct,
            "stop_loss_pct": t.stop_loss_pct,
            "initial_size": t.initial_size,
            "dca_adds": t.dca_adds,
        }
        tmp = self.path.with_suffix(".tmp")
        last_err: OSError | None = None
        for attempt in range(5):
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                # Windows: replace can hit AccessDenied if AV/indexer locks the file.
                os.replace(tmp, self.path)
                return
            except OSError as exc:
                last_err = exc
                time.sleep(0.05 * (attempt + 1))
        if last_err is not None:
            raise last_err

    @staticmethod
    def _tp_sl_prices(
        side: str,
        entry: float,
        take_profit_pct: float,
        stop_loss_pct: float,
    ) -> tuple[float, float]:
        tp_mult = take_profit_pct / 100.0
        sl_mult = stop_loss_pct / 100.0
        if side == "long":
            return entry * (1 + tp_mult), entry * (1 - sl_mult)
        return entry * (1 - tp_mult), entry * (1 + sl_mult)

    def open_trade(
        self,
        coin: str,
        side: str,
        entry_price: float,
        size: float,
        take_profit_pct: float,
        stop_loss_pct: float,
        *,
        equity_at_entry: float = 0.0,
    ) -> ActiveTrade:
        tp, sl = self._tp_sl_prices(side, entry_price, take_profit_pct, stop_loss_pct)
        self.trade = ActiveTrade(
            coin=coin,
            side=side,
            entry_price=entry_price,
            size=size,
            take_profit_price=tp,
            stop_loss_price=sl,
            opened_at=time.time(),
            equity_at_entry=equity_at_entry,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            initial_size=size,
            dca_adds=0,
        )
        self._save()
        return self.trade

    def record_dca_add(self, new_avg_entry: float, new_size: float) -> None:
        """After a DCA fill: re-center TP/SL on the new average entry."""
        if self.trade is None:
            return
        t = self.trade
        tp, sl = self._tp_sl_prices(
            t.side, new_avg_entry, t.take_profit_pct, t.stop_loss_pct
        )
        t.entry_price = new_avg_entry
        t.size = new_size
        t.take_profit_price = tp
        t.stop_loss_price = sl
        t.dca_adds += 1
        self._save()

    def clear(self) -> None:
        self.trade = None
        self._save()

    def sync_from_exchange(
        self,
        coin: str,
        side: str,
        entry_price: float,
        size: float,
        take_profit_pct: float,
        stop_loss_pct: float,
    ) -> None:
        """Rebuild TP/SL from exchange position after restart."""
        if self.trade is not None and self.trade.side == side and self.trade.coin == coin:
            return
        self.open_trade(coin, side, entry_price, size, take_profit_pct, stop_loss_pct)

    def soft_adopt(
        self,
        coin: str,
        side: str,
        entry_price: float,
        size: float,
        take_profit_pct: float,
        stop_loss_pct: float,
        *,
        equity_at_entry: float = 0.0,
    ) -> ActiveTrade:
        """
        Attach local state to an exchange position without resetting the hold clock
        when the same coin+side was already tracked (survives script restart).
        If coin/side is new or unknown, starts the clock from now.
        """
        preserved_opened: float | None = None
        preserved_equity = equity_at_entry
        preserved_initial = size
        preserved_dca = 0
        if (
            self.trade is not None
            and self.trade.coin == coin
            and self.trade.side == side
        ):
            preserved_opened = self.trade.opened_at
            if preserved_equity <= 0:
                preserved_equity = self.trade.equity_at_entry
            preserved_initial = self.trade.initial_size or size
            preserved_dca = self.trade.dca_adds
        tp, sl = self._tp_sl_prices(side, entry_price, take_profit_pct, stop_loss_pct)
        self.trade = ActiveTrade(
            coin=coin,
            side=side,
            entry_price=entry_price,
            size=size,
            take_profit_price=tp,
            stop_loss_price=sl,
            opened_at=float(preserved_opened) if preserved_opened else time.time(),
            equity_at_entry=preserved_equity,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            initial_size=preserved_initial,
            dca_adds=preserved_dca,
        )
        self._save()
        return self.trade
