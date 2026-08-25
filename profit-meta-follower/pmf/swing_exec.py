"""Holder-meta field + indicator swing trader. Shared by backtest and live.

Meta (holder crowd refine) decides WHICH coins and the crowd bias.
Indicators from PriceEngine (marks + 1m/15m/1h candles) decide WHEN to open and
WHEN to close: entry trigger + take-profit / stop-loss / RSI revert / max hold.

Unlike price gates (which only veto crowd votes) this owns entry and exit timing,
so it actually round-trips market fluctuation instead of holding forever.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .price_engine import PriceEngine
from .types import CoinVote

ENTRY_KINDS = ("rsi_dip", "ema_pullback", "breakout", "range_dip")
SIGNAL_TFS = ("1m", "15m", "1h")


@dataclass
class _SwingPos:
    side: str  # long | short
    entry_px: float
    entry_ts: float


def _side_sign(side: str) -> int:
    return 1 if str(side).lower() == "long" else -1


def _copy_vote(v: CoinVote, side: str) -> CoinVote:
    return replace(v, side=side)


@dataclass(frozen=True)
class SwingCfg:
    mode: str  # follow | reverse
    entry: str
    tf: str
    rsi_buy: float
    rsi_sell: float
    band_pct: float
    break_pct: float
    lookback_s: float
    tp_pct: float
    sl_pct: float
    max_hold_s: float
    exit_rsi: float
    reentry_s: float


def swing_cfg_from(cfg: Any) -> SwingCfg:
    mode = str(getattr(cfg, "SWING_META_MODE", "follow") or "follow").strip().lower()
    if mode not in ("follow", "reverse"):
        mode = "follow"
    entry = str(getattr(cfg, "SWING_ENTRY", "rsi_dip") or "rsi_dip").strip().lower()
    if entry not in ENTRY_KINDS:
        entry = "rsi_dip"
    tf = str(getattr(cfg, "SWING_TF", "15m") or "15m").strip().lower()
    if tf not in SIGNAL_TFS:
        tf = "15m"
    return SwingCfg(
        mode=mode,
        entry=entry,
        tf=tf,
        rsi_buy=float(getattr(cfg, "SWING_RSI_BUY", 35.0) or 35.0),
        rsi_sell=float(getattr(cfg, "SWING_RSI_SELL", 65.0) or 65.0),
        band_pct=abs(float(getattr(cfg, "SWING_BAND_PCT", 0.008) or 0.008)),
        break_pct=abs(float(getattr(cfg, "SWING_BREAK_PCT", 0.010) or 0.010)),
        lookback_s=float(getattr(cfg, "SWING_LOOKBACK_S", 1800.0) or 1800.0),
        tp_pct=abs(float(getattr(cfg, "SWING_TP_PCT", 1.2) or 1.2)),
        sl_pct=abs(float(getattr(cfg, "SWING_SL_PCT", 1.8) or 1.8)),
        max_hold_s=float(getattr(cfg, "SWING_MAX_HOLD_S", 14400.0) or 14400.0),
        exit_rsi=float(getattr(cfg, "SWING_EXIT_RSI", 0.0) or 0.0),
        reentry_s=max(0.0, float(getattr(cfg, "SWING_REENTRY_S", 900.0) or 0.0)),
    )


class SwingTrader:
    """Stateful indicator trader inside the holder-meta coin field."""

    def __init__(self) -> None:
        self.positions: dict[str, _SwingPos] = {}
        # coin → ts of last exit; blocks same-tick re-entry after TP/SL.
        self.last_exit: dict[str, float] = {}

    def dump(self) -> dict[str, Any]:
        return {
            "positions": {
                c: {"side": p.side, "entry_px": p.entry_px, "entry_ts": p.entry_ts}
                for c, p in self.positions.items()
            },
            "last_exit": dict(self.last_exit),
        }

    @classmethod
    def from_dump(cls, raw: Any) -> "SwingTrader":
        out = cls()
        if not isinstance(raw, dict):
            return out
        for coin, row in (raw.get("positions") or {}).items():
            if not isinstance(row, dict):
                continue
            side = str(row.get("side") or "")
            if side not in ("long", "short"):
                continue
            out.positions[str(coin)] = _SwingPos(
                side=side,
                entry_px=float(row.get("entry_px") or 0),
                entry_ts=float(row.get("entry_ts") or 0),
            )
        for coin, ts in (raw.get("last_exit") or {}).items():
            try:
                out.last_exit[str(coin)] = float(ts)
            except (TypeError, ValueError):
                continue
        return out

    def entry_ok(self, coin: str, want: int, asof: float, price: PriceEngine, sc: SwingCfg) -> bool:
        """Indicator timing for the desired side (+1 long / −1 short)."""
        if sc.entry == "rsi_dip":
            rsi = price.rsi(coin, asof, tf=sc.tf)
            return rsi <= sc.rsi_buy if want > 0 else rsi >= sc.rsi_sell
        if sc.entry == "ema_pullback":
            bias = price.ema_bias(coin, asof, tf=sc.tf)
            return bias <= -sc.band_pct if want > 0 else bias >= sc.band_pct
        if sc.entry == "breakout":
            ret = price.ret(coin, sc.lookback_s, asof)
            return ret >= sc.break_pct if want > 0 else ret <= -sc.break_pct
        # range_dip: buy after a sell-off from the local high, sell after a pop
        rd = price.range_dump(coin, sc.lookback_s, asof, tf="1m")
        if want > 0:
            return rd <= -sc.band_pct
        return price.ret(coin, sc.lookback_s, asof) >= sc.break_pct

    def exit_reason(
        self,
        coin: str,
        pos: _SwingPos,
        asof: float,
        price: PriceEngine,
        sc: SwingCfg,
    ) -> str | None:
        if sc.max_hold_s > 0 and (asof - pos.entry_ts) >= sc.max_hold_s:
            return "max_hold"
        px = price.price_at(coin, asof)
        if px <= 0 or pos.entry_px <= 0:
            return None
        move = (px / pos.entry_px - 1.0) * 100.0
        if pos.side == "short":
            move = -move
        if sc.tp_pct > 0 and move >= sc.tp_pct:
            return "take_profit"
        if sc.sl_pct > 0 and move <= -sc.sl_pct:
            return "stop_loss"
        if sc.exit_rsi > 0:
            rsi = price.rsi(coin, asof, tf=sc.tf)
            if pos.side == "long" and rsi >= sc.exit_rsi:
                return "rsi_revert"
            if pos.side == "short" and rsi <= (100.0 - sc.exit_rsi):
                return "rsi_revert"
        return None

    def apply(
        self,
        meta: list[CoinVote],
        *,
        managed: set[str],
        cfg: Any,
        now: float,
        price: PriceEngine | None,
        max_n: int,
    ) -> list[CoinVote]:
        """Return the desired book: swing trades inside the holder-meta field."""
        for coin in list(self.positions):
            if coin not in managed:
                self.positions.pop(coin, None)
        if price is None:
            return []

        sc = swing_cfg_from(cfg)
        field = {v.coin: v for v in meta}
        picked: list[CoinVote] = []
        # Keep the cooldown map bounded to the current field (live runs for weeks).
        for coin in list(self.last_exit):
            if coin not in field and coin not in self.positions:
                self.last_exit.pop(coin, None)

        for coin, pos in list(self.positions.items()):
            v = field.get(coin)
            if v is None:
                self.positions.pop(coin, None)
                self.last_exit[coin] = float(now)
                continue
            if self.exit_reason(coin, pos, now, price, sc) is not None:
                self.positions.pop(coin, None)
                self.last_exit[coin] = float(now)
                continue
            picked.append(_copy_vote(v, pos.side))

        held = {p.coin for p in picked}
        for v in meta:
            if len(picked) >= max_n:
                break
            if v.coin in held:
                continue
            last_exit = self.last_exit.get(v.coin)
            # Always wait at least one tick after an exit so a stop-out can't re-arm instantly.
            if last_exit is not None and (float(now) - last_exit) <= sc.reentry_s:
                continue
            crowd = _side_sign(v.side)
            want = crowd if sc.mode == "follow" else -crowd
            if not self.entry_ok(v.coin, want, now, price, sc):
                continue
            px = price.price_at(v.coin, now)
            if px <= 0:
                continue
            side = "long" if want > 0 else "short"
            self.positions[v.coin] = _SwingPos(side=side, entry_px=float(px), entry_ts=float(now))
            picked.append(_copy_vote(v, side))

        return picked[:max_n]
