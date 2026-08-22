"""Replay daily telemetry ticks to score strategy parameters (no size/leverage)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from .consensus import BookEngine
from .types import CoinVote, MarketCtx


SIZE_LEV_KEYS = frozenset(
    {
        "OUR_GROSS_MARGIN_PCT",
        "MAX_MARGIN_PER_COIN_PCT",
        "SINGLE_NAME_SIZE_MULT",
        "OUR_MIN_LEVERAGE",
        "OUR_MAX_LEVERAGE",
        "LEVERAGE_MODE",
        "SIZE_MODE",
        "COPY_MARGIN_CAP_PCT",
    }
)

# Params safe to sweep in backtest (signal / timing only).
BACKTEST_PARAM_KEYS = (
    "FLOW_EMA_ALPHA",
    "EXIT_FLOW",
    "EXIT_RAW_FLOW",
    "EXIT_AGREEMENT_GIVEBACK",
    "CONV_GIVEBACK",
    "OPEN_CONFIRM_S",
    "MIN_SIDE_AGREEMENT",
    "EXIT_SIDE_AGREEMENT",
    "MIN_AVG_CONVICTION",
    "EXIT_AVG_CONVICTION",
    "MIN_ENTRY_FLOW",
    "REBALANCE_COOLDOWN_S",
)


@dataclass
class SimState:
    managed: set[str] = field(default_factory=set)
    entries: int = 0
    exits: int = 0
    hold_ticks: int = 0
    flat_ticks: int = 0


@dataclass
class BacktestResult:
    params: dict[str, Any]
    entries: int
    exits: int
    avg_hold_ticks: float
    time_in_market_pct: float
    score: float


def _default_markets(coins: Iterable[str]) -> dict[str, MarketCtx]:
    big = 5_000_000.0
    return {c: MarketCtx(c, big, 0.00001, big, 0.0) for c in coins}


def _row_to_votes(raw: list[dict[str, Any]], listed: int) -> list[CoinVote]:
    votes: list[CoinVote] = []
    for r in raw:
        coin = str(r.get("c") or "")
        if not coin:
            continue
        side = str(r.get("s") or "long")
        wl = int(r.get("wl") or 0)
        ws = int(r.get("ws") or 0)
        conv = float(r.get("conv") or 0)
        agr = float(r.get("agr") or 0)
        side_n = wl if side == "long" else ws
        votes.append(
            CoinVote(
                coin=coin,
                side=side,
                wallets_long=wl,
                wallets_short=ws,
                voters=max(1, listed),
                agreement=agr,
                avg_conviction=conv,
                median_leverage=10,
                score=abs(conv) * agr * max(1, side_n),
                mean_leverage=10.0,
            )
        )
    votes.sort(key=lambda x: x.score, reverse=True)
    return votes


def load_tick_files(telemetry_dir: Path, *, days: int | None = None) -> list[Path]:
    if not telemetry_dir.exists():
        return []
    day_dirs = sorted(p for p in telemetry_dir.iterdir() if p.is_dir())
    if days is not None and days > 0:
        day_dirs = day_dirs[-days:]
    files: list[Path] = []
    for d in day_dirs:
        p = d / "ticks.jsonl"
        if p.exists():
            files.append(p)
    return files


def iter_ticks(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row


def replay(cfg: Any, ticks: Iterable[dict[str, Any]]) -> SimState:
    eng = BookEngine()
    st = SimState()
    markets: dict[str, MarketCtx] = {}
    for row in ticks:
        ts = float(row.get("ts") or 0)
        listed = int(row.get("listed") or 50)
        raw = row.get("raw") or []
        if not raw:
            continue
        votes_in = _row_to_votes(raw, listed)
        for v in votes_in:
            markets.setdefault(v.coin, MarketCtx(v.coin, 5e6, 0.00001, 5e6, 0.0))
        out = eng.refine(
            votes_in,
            markets=markets,
            managed=set(st.managed),
            cfg=cfg,
            now=ts,
            log=None,
        )
        trade_coins = {v.coin for v in out}
        # Entry: newly qualified
        for c in trade_coins - st.managed:
            st.entries += 1
            st.managed.add(c)
        # Exit: dropped from trade book
        for c in list(st.managed):
            if c not in trade_coins:
                st.exits += 1
                st.managed.discard(c)
        if st.managed:
            st.hold_ticks += 1
        else:
            st.flat_ticks += 1
    return st


def score_sim(st: SimState, *, target_hold_ticks: float = 120.0) -> float:
    total = st.hold_ticks + st.flat_ticks
    if total <= 0:
        return -1e9
    in_mkt = st.hold_ticks / total
    avg_hold = st.hold_ticks / max(1, st.entries)
    # Prefer ~30m–3h holds: penalize scalp (too many exits) and sticky (never exits).
    exit_penalty = max(0, st.exits - st.entries) * 2.0
    hold_penalty = abs(math.log(max(1.0, avg_hold / target_hold_ticks)))
    churn_penalty = (st.entries + st.exits) * 0.15
    return in_mkt * 10.0 - hold_penalty - exit_penalty - churn_penalty


def run_backtest(base_cfg: Any, telemetry_dir: Path, overrides: dict[str, Any], *, days: int = 7) -> BacktestResult:
    cfg = SimpleNamespace(**{k: getattr(base_cfg, k) for k in dir(base_cfg) if k.isupper()})
    for k, v in overrides.items():
        if k in SIZE_LEV_KEYS:
            continue
        setattr(cfg, k, v)
    paths = load_tick_files(telemetry_dir, days=days)
    st = replay(cfg, iter_ticks(paths))
    sc = score_sim(st)
    avg_hold = st.hold_ticks / max(1, st.entries)
    total = st.hold_ticks + st.flat_ticks
    in_mkt = (st.hold_ticks / total * 100.0) if total else 0.0
    return BacktestResult(
        params=overrides,
        entries=st.entries,
        exits=st.exits,
        avg_hold_ticks=avg_hold,
        time_in_market_pct=in_mkt,
        score=sc,
    )


def grid_search(
    base_cfg: Any,
    telemetry_dir: Path,
    grid: dict[str, list[Any]],
    *,
    days: int = 7,
    top_n: int = 10,
) -> list[BacktestResult]:
    keys = [k for k in grid if k not in SIZE_LEV_KEYS]
    if not keys:
        return []

    results: list[BacktestResult] = []

    def _walk(i: int, cur: dict[str, Any]) -> None:
        if i >= len(keys):
            results.append(run_backtest(base_cfg, telemetry_dir, cur, days=days))
            return
        k = keys[i]
        for v in grid[k]:
            cur[k] = v
            _walk(i + 1, cur)

    _walk(0, {})
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]
