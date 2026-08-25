"""Shared crowd strategy execution — identical for backtest and live."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .consensus import BookEngine
from .mtf_exec import MtfTrader
from .price_engine import PriceEngine
from .price_gates import price_gate_ok
from .strategy_spec import StrategySpec, parse_strategy
from .swing_exec import SwingTrader
from .types import CoinVote, MarketCtx


def merge_cfg(base: Any, overrides: dict[str, Any] | None = None) -> Any:
    """Shallow cfg view: overrides win, else getattr(base)."""
    ov = dict(overrides or {})
    keys = set(ov)
    if base is not None:
        for name in dir(base):
            if name.isupper():
                keys.add(name)
    ns = SimpleNamespace()
    for k in keys:
        if k in ov:
            setattr(ns, k, ov[k])
        elif base is not None:
            setattr(ns, k, getattr(base, k))
    return ns


def pick_trade_votes(
    raw: list[CoinVote],
    *,
    book: BookEngine,
    markets: dict[str, MarketCtx],
    managed: set[str],
    cfg: Any,
    now: float,
    spec: StrategySpec | str | None,
    price: PriceEngine | None,
    log: Any = None,
    mtf_trader: MtfTrader | None = None,
    swing_trader: SwingTrader | None = None,
) -> list[CoinVote]:
    """Apply backtest strategy style + price gates. Same code path live and offline."""
    if not isinstance(spec, StrategySpec):
        spec = parse_strategy(str(spec or getattr(cfg, "BACKTEST_LIVE_STRATEGY", "") or "cloud_holders"))
    max_n = int(getattr(cfg, "MAX_COINS_IN_BOOK", 3) or 3)

    if spec.style == "direct":
        picked = list(raw[:max_n])
    elif spec.style == "flow":
        refined = book.refine(raw, markets=markets, managed=managed, cfg=cfg, now=now, log=log)
        enter_flow = float(getattr(cfg, "MIN_ENTRY_FLOW", 0.0) or 0.0)
        exit_raw = float(getattr(cfg, "EXIT_RAW_FLOW", -0.02) or -0.02)
        picked = []
        for v in refined:
            rflow = float(getattr(v, "raw_flow", 0.0) or 0.0)
            flow = float(getattr(v, "flow", 0.0) or 0.0)
            if v.coin in managed:
                if exit_raw < 0 and rflow <= exit_raw:
                    continue
                picked.append(v)
            elif flow >= enter_flow:
                picked.append(v)
        picked = picked[:max_n]
    elif spec.style == "mtf_meta":
        meta = book.refine(raw, markets=markets, managed=managed, cfg=cfg, now=now, log=log)
        trader = mtf_trader if mtf_trader is not None else MtfTrader()
        picked = trader.apply(
            meta, managed=managed, cfg=cfg, now=now, price=price, max_n=max_n
        )
    elif spec.style == "swing_meta":
        meta = book.refine(raw, markets=markets, managed=managed, cfg=cfg, now=now, log=log)
        swing = swing_trader if swing_trader is not None else SwingTrader()
        picked = swing.apply(
            meta, managed=managed, cfg=cfg, now=now, price=price, max_n=max_n
        )
    else:
        picked = book.refine(raw, markets=markets, managed=managed, cfg=cfg, now=now, log=log)

    if spec.gate and price is not None:
        asof = float(now)

        def _ret(coin: str, look: float) -> float:
            return price.ret(coin, look, asof)

        def _bias(coin: str) -> float:
            return price.ema_bias(coin, asof)

        def _atr(coin: str) -> float:
            return price.atr_pct(coin, asof)

        def _rsi(coin: str) -> float:
            return price.rsi(coin, asof)

        def _rdump(coin: str, look: float) -> float:
            return price.range_dump(coin, look, asof)

        picked = [
            v
            for v in picked
            if price_gate_ok(
                gate=spec.gate,
                coin=v.coin,
                side=str(v.side),
                managed=managed,
                cfg=cfg,
                ret=_ret,
                ema_bias=_bias,
                atr_pct=_atr,
                rsi=_rsi,
                range_dump=_rdump,
                has_btc=lambda: price.has_coin("BTC"),
            )
        ]
    return picked[:max_n]
