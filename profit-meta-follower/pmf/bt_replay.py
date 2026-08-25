"""Replay crowd strategies on research data → per-tick target matrices."""

from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Any

import numpy as np

from .consensus import BookEngine, build_votes
from .price_engine import PriceEngine
from .strategy_exec import pick_trade_votes
from .mtf_exec import MtfTrader
from .swing_exec import SwingTrader
from .strategy_spec import StrategySpec, parse_strategy
from .research_load import BookRow, ResearchDataset, TickRow
from .types import CoinVote, MarketCtx, WalletSnapshot

# Process-local: reuse raw votes when only refine-flow / MTF / swing / gate knobs change.
_RAW_VOTE_CACHE: dict[tuple[Any, ...], list[CoinVote]] = {}


def clear_replay_caches() -> None:
    _RAW_VOTE_CACHE.clear()


def _raw_vote_key(
    tick_i: int,
    wallet_mode: str,
    listed: int,
    cfg: Any,
) -> tuple[Any, ...]:
    return (
        tick_i,
        wallet_mode,
        int(listed),
        round(float(getattr(cfg, "MIN_SIDE_AGREEMENT", 0) or 0), 6),
        round(float(getattr(cfg, "EXIT_SIDE_AGREEMENT", 0) or 0), 6),
        round(float(getattr(cfg, "MIN_AVG_CONVICTION", 0) or 0), 6),
        round(float(getattr(cfg, "MIN_WALLET_CONVICTION", 0) or 0), 6),
        round(float(getattr(cfg, "STALE_SNAPSHOT_S", 0) or 0), 3),
    )


def _copy_votes(votes: list[CoinVote]) -> list[CoinVote]:
    # Shallow copy: refine may set flow/ema on the instance.
    return [copy(v) for v in votes]


def _cfg_from(base: Any, overrides: dict[str, Any], *, listed: int) -> SimpleNamespace:
    keys = [
        "DEX_SCOPE",
        "ALLOW_COINS",
        "DENY_COINS",
        "STALE_SNAPSHOT_S",
        "MIN_WALLET_CONVICTION",
        "MIN_LIVE_VOTERS_PCT",
        "MIN_LIVE_VOTERS",
        "MIN_WALLETS_ON_COIN_PCT",
        "MIN_WALLETS_ON_COIN",
        "MIN_SIDE_AGREEMENT",
        "EXIT_SIDE_AGREEMENT",
        "MIN_AVG_CONVICTION",
        "MAX_COINS_IN_BOOK",
        "MAX_LIVE_EQUITY_DROP",
        "PREFERRED_COINS",
        "MIN_WALLETS_OTHER_PCT",
        "FLOW_EMA_ALPHA",
        "OPEN_CONFIRM_S",
        "EXIT_FLOW",
        "EXIT_RAW_FLOW",
        "EXIT_AGREEMENT_GIVEBACK",
        "CONV_GIVEBACK",
        "MIN_ENTRY_FLOW",
        "EXIT_AVG_CONVICTION",
        "STICKY_BOOK_SLOTS",
        "TRADE_MODE",
        "MIN_COIN_DAY_VOLUME",
        "MAX_HOSTILE_FUNDING",
        "MAX_BASIS_ABS",
        "MTF_PRESET",
        "MTF_EMA",
        "MTF_MIN_AGREE",
        "MTF_MIN_SCORE",
        "MTF_WEIGHT_POWER",
        "MTF_EXIT",
        "MTF_META_MODE",
        "MTF_MAX_HOLD_S",
        "MTF_EXEC_IV",
        "SWING_META_MODE",
        "SWING_ENTRY",
        "SWING_TF",
        "SWING_RSI_BUY",
        "SWING_RSI_SELL",
        "SWING_BAND_PCT",
        "SWING_BREAK_PCT",
        "SWING_LOOKBACK_S",
        "SWING_TP_PCT",
        "SWING_SL_PCT",
        "SWING_MAX_HOLD_S",
        "SWING_EXIT_RSI",
        "SWING_REENTRY_S",
    ]
    cfg = SimpleNamespace()
    for k in keys:
        setattr(cfg, k, getattr(base, k, None))
    cfg.BASKET_SIZE = int(listed)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _votes_from_compact(raw: list[dict[str, Any]], listed: int) -> list[CoinVote]:
    votes: list[CoinVote] = []
    for r in raw:
        coin = str(r.get("c") or "")
        if not coin:
            continue
        side = str(r.get("s") or "long")
        wl = int(r.get("wl") or 0)
        ws = int(r.get("ws") or 0)
        conv = float(r.get("conv") or r.get("ema") or 0)
        ema = float(r.get("ema") or conv)
        agr = float(r.get("agr") or 0)
        side_n = wl if side == "long" else ws
        v = CoinVote(
            coin=coin,
            side=side,
            wallets_long=wl,
            wallets_short=ws,
            voters=max(1, listed),
            agreement=agr,
            avg_conviction=ema,
            median_leverage=int(r.get("lev") or 10),
            score=abs(conv) * agr * max(1, side_n),
            mean_leverage=float(r.get("lev") or 10),
        )
        v.raw_conviction = conv
        v.ema = ema
        v.flow = float(r.get("flow") or 0)
        v.raw_flow = float(r.get("rflow") or 0)
        v.persist_s = float(r.get("pers") or 0)
        votes.append(v)
    votes.sort(key=lambda x: x.score, reverse=True)
    return votes


def _crowd(ds: ResearchDataset, wallet_mode: str) -> tuple[set[str], int]:
    """Live cloud: holder basket + listed=len(tracker.addrs). All-pool uses gather size."""
    if wallet_mode == "all":
        return set(), max(int(ds.pool_size), 1)
    holders = ds.live_holder_addrs or ds.holder_addrs
    listed = int(getattr(ds, "live_listed", 0) or 0) or len(holders)
    return holders, max(listed, 1)


def _filter_snaps(snaps: list[WalletSnapshot], holders: set[str], mode: str) -> list[WalletSnapshot]:
    if mode == "holders":
        return [s for s in snaps if s.address in holders]
    if mode == "all":
        return snaps
    return snaps


def _markets_for_row(
    book: BookRow | None,
    coin_index: dict[str, int],
    *,
    price: PriceEngine | None = None,
    asof: float = 0.0,
) -> dict[str, MarketCtx]:
    """Build MarketCtx exactly like live (mark + funding + oi + basis + day_vol)."""
    out: dict[str, MarketCtx] = {}
    big = 5_000_000.0
    t = float(asof)

    if book is not None and book.mkt:
        for coin, vec in book.mkt.items():
            if not vec:
                continue
            mark = float(vec[0] or 0)
            funding = float(vec[1] or 0) if len(vec) > 1 else 0.0
            oi = float(vec[2] or 0) if len(vec) > 2 else 0.0
            basis = float(vec[3] or 0) if len(vec) > 3 else 0.0
            day_vol = float(vec[4] or 0) if len(vec) > 4 else 0.0
            out[str(coin)] = MarketCtx(
                str(coin),
                day_volume=day_vol if day_vol > 0 else big,
                funding=funding,
                open_interest=oi if oi > 0 else big,
                basis=basis,
                mark=mark,
            )
    elif book is not None:
        for coin, px in book.marks.items():
            out[str(coin)] = MarketCtx(str(coin), big, 0.0, big, 0.0, mark=float(px))

    # Fill / enrich from PriceEngine (marks.jsonl + book mkt history).
    for coin in coin_index:
        if price is not None:
            ctx = price.market_ctx_at(coin, t, default_day_vol=big)
            if coin in out:
                prev = out[coin]
                # Prefer non-zero book snapshot; else engine asof.
                out[coin] = MarketCtx(
                    coin,
                    day_volume=prev.day_volume if prev.day_volume not in (0.0, big) else ctx.day_volume,
                    funding=prev.funding if prev.funding != 0.0 else ctx.funding,
                    open_interest=prev.open_interest if prev.open_interest not in (0.0, big) else ctx.open_interest,
                    basis=prev.basis if prev.basis != 0.0 else ctx.basis,
                    mark=prev.mark if prev.mark > 0 else ctx.mark,
                )
            else:
                out[coin] = ctx
        elif coin not in out:
            out[coin] = MarketCtx(coin, big, 0.0, big, 0.0)
    return out


def _targets_from_votes(
    votes: list[CoinVote],
    coin_index: dict[str, int],
    max_slots: int,
) -> tuple[list[int], list[int]]:
    coins: list[int] = [-1] * max_slots
    sides: list[int] = [0] * max_slots
    for i, v in enumerate(votes[:max_slots]):
        idx = coin_index.get(v.coin)
        if idx is None:
            continue
        coins[i] = idx
        sides[i] = 1 if str(v.side).lower() == "long" else -1
    return coins, sides


def replay_cloud_refine(
    ds: ResearchDataset,
    base_cfg: Any,
    overrides: dict[str, Any],
    *,
    wallet_mode: str = "holders",
    max_slots: int = 3,
    gate: str | None = None,
    style: str = "refine",
    progress_cb: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Same pick_trade_votes path as live (BookEngine + optional price gates)."""
    holders, listed = _crowd(ds, wallet_mode)
    cfg = _cfg_from(base_cfg, overrides, listed=listed)
    # merge indicator overrides onto cfg for gates
    for k, v in overrides.items():
        setattr(cfg, k, v)
    eng = BookEngine()
    mtf_trader = MtfTrader() if style == "mtf_meta" else None
    swing_trader = SwingTrader() if style == "swing_meta" else None
    managed: set[str] = set()
    n = ds.n_ticks
    target_coin = np.full((n, max_slots), -1, dtype=np.int32)
    target_side = np.zeros((n, max_slots), dtype=np.int8)
    price_eng: PriceEngine | None = getattr(ds, "price_engine", None)
    panel = getattr(ds, "ind_panel", None)
    if price_eng is not None and panel is not None:
        from .bt_panels import PanelPrice

        price: PriceEngine | PanelPrice | None = PanelPrice(price_eng, panel, ds)
    else:
        price = price_eng
    filter_mode = "off" if wallet_mode == "all" else "holder"
    owns_timing = style in ("mtf_meta", "swing_meta")
    needs_candles = gate is not None or owns_timing
    spec = StrategySpec(
        name=f"replay_{style}_{wallet_mode}_{gate or 'none'}",
        style=style,
        wallet_mode=wallet_mode,
        gate=gate,
        needs_candles=needs_candles,
        candle_intervals=("1m", "15m", "1h") if needs_candles else (),
        filter_mode=filter_mode,
    )

    report_every = max(1, n // 40) if n else 1
    markets_cache = getattr(ds, "markets_by_tick", None)
    set_tick = getattr(price, "set_tick", None)
    for i in range(n):
        ts = float(ds.ts[i])
        if set_tick is not None:
            set_tick(i)
        raw_votes: list[CoinVote] = []
        book = ds.books[i]
        tick = ds.ticks[i]
        if markets_cache is not None and i < len(markets_cache):
            markets = markets_cache[i]
        else:
            markets = _markets_for_row(book, ds.coin_index, price=price_eng, asof=ts)

        if book is not None:
            vk = _raw_vote_key(i, wallet_mode, listed, cfg)
            cached = _RAW_VOTE_CACHE.get(vk)
            if cached is not None:
                raw_votes = _copy_votes(cached)
            else:
                snaps = _filter_snaps(book.wallets, holders, wallet_mode)
                raw_votes = build_votes(snaps, set(), cfg, ts, listed=listed)
                _RAW_VOTE_CACHE[vk] = _copy_votes(raw_votes)
        elif tick is not None:
            # Compact ticks are already independent of tunable refine floors.
            vk = (i, wallet_mode, "compact", listed)
            cached = _RAW_VOTE_CACHE.get(vk)
            if cached is not None:
                raw_votes = _copy_votes(cached)
            else:
                if wallet_mode == "holders":
                    raw_votes = _votes_from_compact(tick.holder_votes, listed)
                elif wallet_mode == "all":
                    raw_votes = _votes_from_compact(tick.all_votes, listed)
                else:
                    raw_votes = _votes_from_compact(tick.trade_votes, listed)
                _RAW_VOTE_CACHE[vk] = _copy_votes(raw_votes)

        if not raw_votes:
            managed = set()
            if mtf_trader is not None:
                mtf_trader.positions.clear()
            if swing_trader is not None:
                swing_trader.positions.clear()
            if progress_cb is not None and (i % report_every == 0 or i + 1 >= n):
                progress_cb(i + 1, n)
            continue

        refined = pick_trade_votes(
            raw_votes,
            book=eng,
            markets=markets,
            managed=managed,
            cfg=cfg,
            now=ts,
            spec=spec,
            price=price,
            log=None,
            mtf_trader=mtf_trader,
            swing_trader=swing_trader,
        )
        managed = {v.coin for v in refined}
        coins, sides = _targets_from_votes(refined, ds.coin_index, max_slots)
        for j in range(max_slots):
            target_coin[i, j] = coins[j]
            target_side[i, j] = sides[j]
        if progress_cb is not None and (i % report_every == 0 or i + 1 >= n):
            progress_cb(i + 1, n)

    return target_coin, target_side


def replay_direct_raw(
    ds: ResearchDataset,
    base_cfg: Any,
    overrides: dict[str, Any],
    *,
    wallet_mode: str = "holders",
    max_slots: int = 3,
    progress_cb: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top raw votes — same pick_trade_votes(style=direct) as live."""
    return replay_cloud_refine(
        ds,
        base_cfg,
        overrides,
        wallet_mode=wallet_mode,
        max_slots=max_slots,
        style="direct",
        progress_cb=progress_cb,
    )


def replay_flow_momentum(
    ds: ResearchDataset,
    base_cfg: Any,
    overrides: dict[str, Any],
    *,
    wallet_mode: str = "holders",
    max_slots: int = 3,
    progress_cb: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flow filter — same pick_trade_votes(style=flow) as live."""
    return replay_cloud_refine(
        ds,
        base_cfg,
        overrides,
        wallet_mode=wallet_mode,
        max_slots=max_slots,
        style="flow",
        progress_cb=progress_cb,
    )


def _wrap_replay(fn: Any, **fixed: Any) -> Any:
    def _call(ds: Any, cfg: Any, ov: Any, progress_cb: Any = None) -> Any:
        return fn(ds, cfg, ov, progress_cb=progress_cb, **fixed)

    return _call


STRATEGY_REPLAYERS = {
    "cloud_holders": _wrap_replay(replay_cloud_refine, wallet_mode="holders"),
    "cloud_all": _wrap_replay(replay_cloud_refine, wallet_mode="all"),
    "direct_holders": _wrap_replay(replay_direct_raw, wallet_mode="holders"),
    "flow_holders": _wrap_replay(replay_flow_momentum, wallet_mode="holders"),
    # Meta field + original multi-candle MTF (no DCA). Replaces unused logged_trade.
    "mtf_meta_holders": _wrap_replay(
        replay_cloud_refine, wallet_mode="holders", style="mtf_meta"
    ),
    "mtf_meta_all": _wrap_replay(
        replay_cloud_refine, wallet_mode="all", style="mtf_meta"
    ),
    # Meta field + indicator swing timing (TP/SL/RSI/max-hold). Replaces direct_all.
    "swing_meta_holders": _wrap_replay(
        replay_cloud_refine, wallet_mode="holders", style="swing_meta"
    ),
    "swing_meta_all": _wrap_replay(
        replay_cloud_refine, wallet_mode="all", style="swing_meta"
    ),
    # Crowd + full price engine gates (marks + 1m/15m/1h) — same pick_trade_votes as live.
    "crowd_dump_holders": _wrap_replay(replay_cloud_refine, wallet_mode="holders", gate="dump"),
    "crowd_dump_all": _wrap_replay(replay_cloud_refine, wallet_mode="all", gate="dump"),
    "crowd_trend_holders": _wrap_replay(replay_cloud_refine, wallet_mode="holders", gate="trend"),
    "crowd_trend_all": _wrap_replay(replay_cloud_refine, wallet_mode="all", gate="trend"),
    "crowd_vol_holders": _wrap_replay(replay_cloud_refine, wallet_mode="holders", gate="vol"),
    "crowd_vol_all": _wrap_replay(replay_cloud_refine, wallet_mode="all", gate="vol"),
    "crowd_btcdump_holders": _wrap_replay(
        replay_cloud_refine, wallet_mode="holders", gate="btcdump"
    ),
    "crowd_btcdump_all": _wrap_replay(replay_cloud_refine, wallet_mode="all", gate="btcdump"),
    "crowd_rsi_holders": _wrap_replay(replay_cloud_refine, wallet_mode="holders", gate="rsi"),
    "crowd_rsi_all": _wrap_replay(replay_cloud_refine, wallet_mode="all", gate="rsi"),
}
