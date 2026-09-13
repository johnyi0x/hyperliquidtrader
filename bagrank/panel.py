"""Load collector meta_index hours into a dense rank panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .store import load_meta_rows
from .timeutil import to_unix

SID_LONG = 1
SID_SHORT = -1

EXTRA = (
    "conviction",
    "notional",
    "long_n",
    "short_n",
    "funding",
    "oi",
    "premium",
    "volume",
    "prev_day",
)


@dataclass
class RankPanel:
    cycle_unix: np.ndarray  # (T,) int64
    coins: list[str]
    rank: np.ndarray  # (T, C) int32  0 = off board
    side: np.ndarray  # (T, C) int8
    wallets: np.ndarray
    hold_pct: np.ndarray
    agreement: np.ndarray
    mean_leverage: np.ndarray
    marks: np.ndarray
    conviction: np.ndarray
    notional: np.ndarray
    long_n: np.ndarray
    short_n: np.ndarray
    funding: np.ndarray
    oi: np.ndarray
    premium: np.ndarray
    volume: np.ndarray
    prev_day: np.ndarray

    @property
    def n_times(self) -> int:
        return int(self.cycle_unix.shape[0])

    @property
    def n_coins(self) -> int:
        return len(self.coins)

    def take(self, idx: np.ndarray) -> "RankPanel":
        return RankPanel(
            cycle_unix=self.cycle_unix[idx],
            coins=self.coins,
            rank=self.rank[idx],
            side=self.side[idx],
            wallets=self.wallets[idx],
            hold_pct=self.hold_pct[idx],
            agreement=self.agreement[idx],
            mean_leverage=self.mean_leverage[idx],
            marks=self.marks[idx],
            conviction=self.conviction[idx],
            notional=self.notional[idx],
            long_n=self.long_n[idx],
            short_n=self.short_n[idx],
            funding=self.funding[idx],
            oi=self.oi[idx],
            premium=self.premium[idx],
            volume=self.volume[idx],
            prev_day=self.prev_day[idx],
        )


def _zeros(t_n: int, c_n: int) -> dict[str, np.ndarray]:
    z64 = np.zeros((t_n, c_n), dtype=np.float64)
    return {name: z64.copy() for name in EXTRA}


def panel_from_meta_rows(
    rows: list[dict[str, Any]],
    *,
    step_hours: int = 1,
) -> RankPanel:
    _ = step_hours
    if not rows:
        empty = np.zeros((0, 0), dtype=np.float64)
        extras = {name: empty.copy() for name in EXTRA}
        return RankPanel(
            cycle_unix=np.zeros(0, dtype=np.int64),
            coins=[],
            rank=np.zeros((0, 0), dtype=np.int32),
            side=np.zeros((0, 0), dtype=np.int8),
            wallets=np.zeros((0, 0), dtype=np.int32),
            hold_pct=empty.copy(),
            agreement=empty.copy(),
            mean_leverage=empty.copy(),
            marks=empty.copy(),
            **extras,
        )
    times: list[int] = []
    seen_t: set[int] = set()
    coins_set: set[str] = set()
    for row in rows:
        ts = to_unix(row["cycle_ts"])
        if ts not in seen_t:
            seen_t.add(ts)
            times.append(ts)
        coin = str(row.get("coin") or "")
        if coin:
            coins_set.add(coin)
    times.sort()
    coins = sorted(coins_set)
    t_ix = {t: i for i, t in enumerate(times)}
    c_ix = {c: i for i, c in enumerate(coins)}
    t_n = len(times)
    c_n = len(coins)
    rank = np.zeros((t_n, c_n), dtype=np.int32)
    side = np.zeros((t_n, c_n), dtype=np.int8)
    wallets = np.zeros((t_n, c_n), dtype=np.int32)
    hold_pct = np.zeros((t_n, c_n), dtype=np.float64)
    agreement = np.zeros((t_n, c_n), dtype=np.float64)
    mean_lev = np.zeros((t_n, c_n), dtype=np.float64)
    extras = _zeros(t_n, c_n)
    for row in rows:
        ts = to_unix(row["cycle_ts"])
        if ts not in t_ix:
            continue
        coin = str(row.get("coin") or "")
        if coin not in c_ix:
            continue
        i = t_ix[ts]
        j = c_ix[coin]
        rank[i, j] = int(row.get("rank") or 0)
        sd = str(row.get("side") or "").strip().lower()
        side[i, j] = SID_LONG if sd == "long" else SID_SHORT if sd == "short" else 0
        wallets[i, j] = int(row.get("wallets") or 0)
        hold_pct[i, j] = float(row.get("hold_pct") or 0)
        agreement[i, j] = float(row.get("agreement") or 0)
        mean_lev[i, j] = float(row.get("mean_leverage") or row.get("median_leverage") or 1)
        extras["conviction"][i, j] = float(row.get("avg_conviction") or 0)
        extras["notional"][i, j] = float(row.get("notional_usd") or 0)
        extras["long_n"][i, j] = float(row.get("long_n") or 0)
        extras["short_n"][i, j] = float(row.get("short_n") or 0)
    return RankPanel(
        cycle_unix=np.asarray(times, dtype=np.int64),
        coins=coins,
        rank=rank,
        side=side,
        wallets=wallets,
        hold_pct=hold_pct,
        agreement=agreement,
        mean_leverage=mean_lev,
        marks=np.zeros((t_n, c_n), dtype=np.float64),
        **extras,
    )


def resample_closed(panel: RankPanel, step_hours: int) -> RankPanel:
    """Keep the last 1h snapshot in each N-hour bucket (closed bar)."""
    step = max(1, int(step_hours))
    if step <= 1 or panel.n_times == 0:
        return panel
    step_s = step * 3600
    unix = panel.cycle_unix
    keep: list[int] = []
    i = 0
    n = panel.n_times
    while i < n:
        bucket = int(unix[i]) // step_s
        j = i
        while j + 1 < n and int(unix[j + 1]) // step_s == bucket:
            j += 1
        keep.append(j)
        i = j + 1
    return panel.take(np.asarray(keep, dtype=np.int64))


def panel_from_sqlite(conn: Any, *, venue: str, step_hours: int = 1) -> RankPanel:
    _ = step_hours
    return panel_from_meta_rows(load_meta_rows(conn, venue=venue), step_hours=1)
