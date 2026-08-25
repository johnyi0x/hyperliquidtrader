"""Map backtest strategy names → live/offline execution knobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .price_engine import CANDLE_INTERVALS


@dataclass(frozen=True)
class StrategySpec:
    name: str
    style: str  # refine | direct | flow | logged | mtf_meta | swing_meta
    wallet_mode: str  # holders | all
    gate: str | None  # dump | trend | vol | btcdump | rsi | None
    needs_candles: bool
    candle_intervals: tuple[str, ...]
    filter_mode: str  # holder | off


def parse_strategy(name: str | None) -> StrategySpec:
    raw = str(name or "").strip().lower() or "cloud_holders"
    filter_mode = "off" if raw.endswith("_all") or raw in {"all", "no_filter", "filter_off"} else "holder"
    wallet_mode = "all" if filter_mode == "off" else "holders"

    # Meta+timing styles own their entries/exits — indicator names in them are not gates.
    if raw.startswith("mtf") or raw.startswith("swing"):
        style = "mtf_meta" if raw.startswith("mtf") else "swing_meta"
        return StrategySpec(
            name=raw,
            style=style,
            wallet_mode=wallet_mode,
            gate=None,
            needs_candles=True,
            candle_intervals=CANDLE_INTERVALS,
            filter_mode=filter_mode,
        )

    gate: str | None = None
    if "btcdump" in raw:
        gate = "btcdump"
    elif "dump" in raw:
        gate = "dump"
    elif "trend" in raw:
        gate = "trend"
    elif "rsi" in raw:
        gate = "rsi"
    elif "_vol_" in raw or raw.startswith("crowd_vol") or raw.endswith("_vol"):
        gate = "vol"

    if raw.startswith("direct"):
        style = "direct"
    elif raw.startswith("flow"):
        style = "flow"
    elif raw in {"logged_trade", "logged"}:
        style = "logged"
    else:
        style = "refine"

    needs_candles = gate is not None
    intervals = CANDLE_INTERVALS if needs_candles else ()
    return StrategySpec(
        name=raw,
        style=style,
        wallet_mode=wallet_mode,
        gate=gate,
        needs_candles=needs_candles,
        candle_intervals=intervals,
        filter_mode=filter_mode,
    )


def strategy_from_cfg(cfg: Any) -> StrategySpec:
    name = (
        getattr(cfg, "BACKTEST_LIVE_STRATEGY", None)
        or getattr(cfg, "TUNED_STRATEGY", None)
        or "cloud_holders"
    )
    return parse_strategy(str(name))
