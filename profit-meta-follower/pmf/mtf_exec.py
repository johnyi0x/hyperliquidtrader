"""Holder-meta + original multi-candle engine (no DCA). Shared by backtest and live.

Meta (holder crowd refine) picks the coin field + crowd side.
Original MTF (src.mtf / src.engine, closed bars, 1m+15m+1h) times entries/exits
inside that field. Follow = trade with crowd side; reverse = fade it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .price_engine import PriceEngine
from .types import CoinVote

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.engine import exit_signal  # noqa: E402
from src.mtf import prepare_interval_biases  # noqa: E402
from src.registry import STRATEGY_BY_ID  # noqa: E402

# Cross-combo MTF entry signal cache (shared across MtfTrader instances / tune jobs
# in the same process). Key includes every knob that feeds entry_signal.
_ENTRY_CACHE: dict[tuple[Any, ...], int] = {}
# Multi-TF EMA bias maps keyed by (coin, bar-open stamps, ema) — shared across presets.
_BIAS_CACHE: dict[tuple[Any, ...], dict[str, tuple[Any, Any]]] = {}


def clear_mtf_entry_cache() -> None:
    _ENTRY_CACHE.clear()
    _BIAS_CACHE.clear()

# Compact original-bot families (no DCA). sid/p0/p1/p2/aux match src.registry ENTRY_GRIDS.
MTF_PRESETS: dict[str, tuple[int, float, float, float, float]] = {
    "rsi_long_30": (0, 30.0, 0.0, 0.0, 14.0),
    "rsi_long_35": (0, 35.0, 0.0, 0.0, 14.0),
    "rsi_short_70": (1, 70.0, 0.0, 0.0, 14.0),
    "ema_x_long": (4, 0.0, 0.0, 0.0, 0.0),
    "ema_x_short": (5, 0.0, 0.0, 0.0, 0.0),
    "bb_bounce_long": (6, 0.0, 2.0, 0.0, 0.0),
    "zscore_long": (8, 2.0, 0.0, 0.0, 20.0),
    "dump_bounce": (14, 1.2, 0.0, 0.0, 0.0),
    "pump_fade": (15, 1.2, 0.0, 0.0, 0.0),
}

MTF_EXITS: dict[str, tuple[int, float, float]] = {
    "none": (-1, 0.0, 0.0),
    "rsi_55": (0, 55.0, 14.0),
    "ema_0.5": (1, 0.5, 50.0),
    "profit_1.0": (2, 1.0, 0.0),
}


@dataclass
class _Pos:
    side: str  # long | short
    entry_px: float
    entry_ts: float


def resolve_mtf_setup(cfg: Any) -> SimpleNamespace:
    """Build the original-bot setup object from PMF cfg / tune params."""
    preset = str(getattr(cfg, "MTF_PRESET", "rsi_long_30") or "rsi_long_30")
    if preset not in MTF_PRESETS:
        preset = "rsi_long_30"
    sid, p0, p1, p2, aux = MTF_PRESETS[preset]
    spec = STRATEGY_BY_ID.get(sid)
    side = int(spec.side) if spec is not None else 1
    exit_name = str(getattr(cfg, "MTF_EXIT", "rsi_55") or "rsi_55")
    if exit_name not in MTF_EXITS:
        exit_name = "rsi_55"
    eid, ex_p0, ex_aux = MTF_EXITS[exit_name]
    return SimpleNamespace(
        coin="",
        interval=str(getattr(cfg, "MTF_EXEC_IV", "1m") or "1m"),
        sid=int(sid),
        name=preset,
        side=side,
        p0=float(p0),
        p1=float(p1),
        p2=float(p2),
        aux=float(aux),
        is_mtf=True,
        mode="mtf",
        mtf_ema=int(getattr(cfg, "MTF_EMA", 50) or 50),
        mtf_min_agree=int(getattr(cfg, "MTF_MIN_AGREE", 2) or 2),
        mtf_min_score=float(getattr(cfg, "MTF_MIN_SCORE", 0.30) or 0.30),
        mtf_weight_power=float(getattr(cfg, "MTF_WEIGHT_POWER", 0.5) or 0.5),
        use_exit_signal=eid >= 0,
        exit_eid=int(eid),
        ex_p0=float(ex_p0),
        ex_aux=float(ex_aux),
        dca_enabled=False,
        dca_max_adds=0,
    )


def _meta_side_sign(side: str) -> int:
    return 1 if str(side).lower() == "long" else -1


def _copy_vote(v: CoinVote, side: str) -> CoinVote:
    return CoinVote(
        coin=v.coin,
        side=side,
        wallets_long=v.wallets_long,
        wallets_short=v.wallets_short,
        voters=v.voters,
        agreement=v.agreement,
        avg_conviction=v.avg_conviction,
        median_leverage=v.median_leverage,
        score=v.score,
        flow=v.flow,
        raw_flow=v.raw_flow,
        persist_s=v.persist_s,
        ema=v.ema,
        raw_conviction=v.raw_conviction,
        mean_leverage=v.mean_leverage,
        avg_margin_pct=v.avg_margin_pct,
    )


class MtfTrader:
    """Stateful in-field trader: original MTF timing on holder-meta coins."""

    def __init__(self) -> None:
        self.positions: dict[str, _Pos] = {}
        self._sig_cache: dict[tuple[str, int, str], int] = {}

    def dump(self) -> dict[str, Any]:
        return {
            "positions": {
                c: {"side": p.side, "entry_px": p.entry_px, "entry_ts": p.entry_ts}
                for c, p in self.positions.items()
            }
        }

    @classmethod
    def from_dump(cls, raw: Any) -> "MtfTrader":
        out = cls()
        if not isinstance(raw, dict):
            return out
        for coin, row in (raw.get("positions") or {}).items():
            if not isinstance(row, dict):
                continue
            side = str(row.get("side") or "")
            if side not in ("long", "short"):
                continue
            out.positions[str(coin)] = _Pos(
                side=side,
                entry_px=float(row.get("entry_px") or 0),
                entry_ts=float(row.get("entry_ts") or 0),
            )
        return out

    def candles_by_interval(self, price: PriceEngine, coin: str, asof: float) -> dict[str, list[dict[str, Any]]]:
        return {
            iv: price.candles_as_hl_dicts(coin, iv, asof, max_bars=160 if iv == "1m" else 96)
            for iv in ("1m", "15m", "1h")
        }

    def mtf_entry_side(self, coin: str, asof: float, price: PriceEngine, setup: Any) -> int:
        """Original mtf_entry_signal_now: +1 / −1 / 0 on last closed bar ≤ asof."""
        from src.mtf import mtf_entry_signal_now

        exec_iv = str(setup.interval)
        end = price._bar_end(coin, exec_iv, asof)
        bars = price._bars.get(coin, {}).get(exec_iv) or []
        last_t = int(bars[end - 1].t_open_ms) if end > 0 else 0
        cache_key = (
            coin,
            last_t,
            str(setup.name),
            int(getattr(setup, "mtf_ema", 0) or 0),
            int(getattr(setup, "mtf_min_agree", 0) or 0),
            round(float(getattr(setup, "mtf_min_score", 0) or 0), 4),
            round(float(getattr(setup, "mtf_weight_power", 0) or 0), 4),
            exec_iv,
        )
        if last_t:
            hit = _ENTRY_CACHE.get(cache_key)
            if hit is not None:
                return hit
            if cache_key in self._sig_cache:
                return self._sig_cache[cache_key]
        by_iv = self.candles_by_interval(price, coin, asof)
        ema = int(getattr(setup, "mtf_ema", 50) or 50)
        # Bias map depends on candle ends + ema only (not trigger preset).
        ends = tuple(
            int((by_iv.get(iv) or [{}])[-1].get("t") or 0) if by_iv.get(iv) else 0
            for iv in ("1m", "15m", "1h")
        )
        bias_key = (coin, ends, ema)
        biases = _BIAS_CACHE.get(bias_key)
        if biases is None:
            biases = prepare_interval_biases(by_iv, ema_period=ema)
            _BIAS_CACHE[bias_key] = biases
        sig = int(mtf_entry_signal_now(setup, by_iv, biases=biases) or 0)
        if last_t:
            _ENTRY_CACHE[cache_key] = sig
            self._sig_cache[cache_key] = sig
        return sig

    def mtf_should_exit(
        self,
        coin: str,
        asof: float,
        price: PriceEngine,
        setup: Any,
        pos: _Pos,
    ) -> bool:
        exec_iv = str(setup.interval)
        candles = price.candles_as_hl_dicts(coin, exec_iv, asof, max_bars=160)
        pos_side = 1 if pos.side == "long" else -1
        return bool(
            exit_signal(
                setup,
                candles,
                avg_entry_px=float(pos.entry_px or 0),
                position_side=pos_side,
            )
        )

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
        """Return desired book: MTF trades inside the holder-meta field."""
        for coin in list(self.positions):
            if coin not in managed:
                self.positions.pop(coin, None)

        if price is None:
            return []

        setup = resolve_mtf_setup(cfg)
        mode = str(getattr(cfg, "MTF_META_MODE", "follow") or "follow").strip().lower()
        if mode not in ("follow", "reverse"):
            mode = "follow"
        max_hold = float(getattr(cfg, "MTF_MAX_HOLD_S", 14400.0) or 14400.0)
        mtf_side = int(setup.side)  # +1 long family / −1 short family
        want_side = "long" if mtf_side > 0 else "short"

        field = {v.coin: v for v in meta}
        picked: list[CoinVote] = []

        for coin, pos in list(self.positions.items()):
            v = field.get(coin)
            if v is None:
                self.positions.pop(coin, None)
                continue
            hold_s = float(now) - float(pos.entry_ts)
            if max_hold > 0 and hold_s >= max_hold:
                self.positions.pop(coin, None)
                continue
            if setup.use_exit_signal and self.mtf_should_exit(coin, now, price, setup, pos):
                self.positions.pop(coin, None)
                continue
            picked.append(_copy_vote(v, pos.side))

        held = {p.coin for p in picked}
        for v in meta:
            if v.coin in held or len(picked) >= max_n:
                continue
            crowd = _meta_side_sign(v.side)
            aligned = (crowd == mtf_side) if mode == "follow" else (crowd != mtf_side)
            if not aligned:
                continue
            sig = self.mtf_entry_side(v.coin, now, price, setup)
            if sig != mtf_side:
                continue
            px = price.price_at(v.coin, now)
            if px <= 0:
                continue
            self.positions[v.coin] = _Pos(side=want_side, entry_px=float(px), entry_ts=float(now))
            picked.append(_copy_vote(v, want_side))

        return picked[:max_n]
