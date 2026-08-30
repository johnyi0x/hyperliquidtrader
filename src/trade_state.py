"""Active trade TP/SL levels (persisted across restarts). Supports multiple coins."""

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


def _trade_to_dict(t: ActiveTrade) -> dict:
    return {
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


def _trade_from_dict(data: dict) -> ActiveTrade:
    return ActiveTrade(
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


class TradeStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.trades: dict[str, ActiveTrade] = {}
        self._load()

    @property
    def trade(self) -> ActiveTrade | None:
        """Oldest tracked trade (compat). Prefer get(coin)."""
        if not self.trades:
            return None
        return next(iter(self.trades.values()))

    def get(self, coin: str) -> ActiveTrade | None:
        return self.trades.get(str(coin))

    def coins(self) -> list[str]:
        return list(self.trades.keys())

    def _load(self) -> None:
        self.trades = {}
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            raw_trades = data.get("trades")
            if isinstance(raw_trades, dict) and raw_trades:
                for coin, row in raw_trades.items():
                    if isinstance(row, dict) and "side" in row:
                        row = dict(row)
                        row.setdefault("coin", coin)
                        self.trades[str(row.get("coin") or coin)] = _trade_from_dict(row)
            elif "side" in data:
                t = _trade_from_dict(data)
                if t.coin:
                    self.trades[t.coin] = t
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            self.trades = {}

    def _save(self) -> None:
        if not self.trades:
            if self.path.exists():
                try:
                    self.path.unlink()
                except OSError:
                    pass
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 2,
            "trades": {coin: _trade_to_dict(t) for coin, t in self.trades.items()},
        }
        tmp = self.path.with_suffix(".tmp")
        last_err: OSError | None = None
        for attempt in range(5):
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
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
        t = ActiveTrade(
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
        self.trades[str(coin)] = t
        self._save()
        return t

    def record_dca_add(self, new_avg_entry: float, new_size: float, coin: str | None = None) -> None:
        """After a DCA fill: re-center TP/SL on the new average entry."""
        t = self.get(coin) if coin else self.trade
        if t is None:
            return
        tp, sl = self._tp_sl_prices(
            t.side, new_avg_entry, t.take_profit_pct, t.stop_loss_pct
        )
        t.entry_price = new_avg_entry
        t.size = new_size
        t.take_profit_price = tp
        t.stop_loss_price = sl
        t.dca_adds += 1
        self._save()

    def clear(self, coin: str | None = None) -> None:
        if coin is None:
            self.trades = {}
        else:
            self.trades.pop(str(coin), None)
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
        existing = self.get(coin)
        if existing is not None and existing.side == side and existing.coin == coin:
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
        existing = self.get(coin)
        if existing is not None and existing.side == side:
            preserved_opened = existing.opened_at
            if preserved_equity <= 0:
                preserved_equity = existing.equity_at_entry
            preserved_initial = existing.initial_size or size
            preserved_dca = existing.dca_adds
        tp, sl = self._tp_sl_prices(side, entry_price, take_profit_pct, stop_loss_pct)
        t = ActiveTrade(
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
        self.trades[str(coin)] = t
        self._save()
        return t
